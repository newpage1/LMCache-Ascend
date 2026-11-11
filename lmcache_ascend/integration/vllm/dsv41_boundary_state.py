# SPDX-License-Identifier: Apache-2.0
"""Engine-side codec and validity checks for DSV4.1 boundary snapshots.

The wire descriptor type lives in
:mod:`lmcache.v1.multiprocess.layerwise_protocol` (single source of truth).
This module adds the engine-side rules that cannot live on the wire:

- Row provenance: ``row_indices`` must come from the real slot mapping of
  the boundary step. Rows guessed from hashes, modulo arithmetic, or any
  other synthetic scheme describe scratch rows that were never written and
  are rejected before anything is restored.
- Boundary validity: schema/plan/layout/generation agreement, source ring
  not yet overwritten (cursor window), TTL, and phase consistency.
- Fallback selection: when the newest boundary of a key is invalid, the
  previous *complete* boundary of the same key may serve the request; if
  none survives, the lookup is a miss (never a partial restore).

All functions are pure; no module-level mutable state.
"""

# Standard
from dataclasses import dataclass

# Third Party
from lmcache.v1.multiprocess.layerwise_protocol import (
    BOUNDARY_STATE_SCHEMA_VERSION,
    BoundaryStateDescriptor,
    LayerwiseSnapshotCommit,
)

#: Ring planes of the compressor scratch buffer. A snapshot must cover one
#: row set per plane; fewer planes means the snapshot is structurally
#: incomplete and must not restore.
COMPRESSOR_RING_PLANES = 2

#: How far (ring length units) a source ring may have advanced past a
#: snapshot cursor before the captured rows are considered overwritten.
RING_OVERWRITE_TOLERANCE = 0


class DSV41BoundaryStateError(ValueError):
    """A boundary descriptor failed engine-side validation."""


@dataclass(frozen=True)
class BoundaryValidityContext:
    """Facts of the *current* engine state a boundary is checked against."""

    plan_hash: str
    layout_hash: str
    schema_version: int = BOUNDARY_STATE_SCHEMA_VERSION
    server_generation: int | None = None
    current_ring_cursor: int | None = None
    ring_length: int | None = None
    now_monotonic: float | None = None
    snapshot_ttl_s: float | None = None
    committed_at: dict[str, float] | None = None

    def ring_distance(self, snapshot_cursor: int) -> int | None:
        """Signed distance the source ring advanced past a snapshot cursor."""
        if self.current_ring_cursor is None or self.ring_length is None:
            return None
        return (self.current_ring_cursor - snapshot_cursor) % self.ring_length


def validate_row_provenance(
    row_indices: tuple[int, ...], slot_mapping: frozenset[int]
) -> None:
    """Reject boundary rows that were not read from the real slot mapping.

    Args:
        row_indices: Rows captured by the boundary descriptor.
        slot_mapping: The physical scratch rows the engine actually wrote
            for the boundary step.

    Raises:
        DSV41BoundaryStateError: If any row is absent from ``slot_mapping``
            (e.g. hash-derived or modulo-guessed rows) or the mapping is
            empty.
    """
    if not slot_mapping:
        raise DSV41BoundaryStateError("slot mapping is empty for this boundary")
    synthetic = [row for row in row_indices if row not in slot_mapping]
    if synthetic:
        raise DSV41BoundaryStateError(
            f"boundary rows {synthetic} were never written by the slot mapping; "
            "hash-derived or modulo-guessed row indices are rejected"
        )


def validate_boundary_descriptor(
    descriptor: BoundaryStateDescriptor,
    *,
    slot_mapping: frozenset[int] | None = None,
    ring_length: int | None = None,
) -> None:
    """Full engine-side validation of one boundary descriptor.

    One descriptor covers exactly one ring plane: its ``compression_phase``
    identifies which plane, and its ``row_indices`` are that plane's rows.
    Snapshot-level plane completeness (one descriptor per plane) is enforced
    by :func:`boundary_snapshot_is_valid`, not here.

    Args:
        descriptor: Wire descriptor (checksum already enforced on decode).
        slot_mapping: When given, enforces row provenance.
        ring_length: When given, enforces ``ring_cursor < ring_length``.

    Raises:
        DSV41BoundaryStateError: On schema, cursor, or provenance
            violations.
    """
    if descriptor.schema_version != BOUNDARY_STATE_SCHEMA_VERSION:
        raise DSV41BoundaryStateError(
            f"boundary schema {descriptor.schema_version} != engine schema "
            f"{BOUNDARY_STATE_SCHEMA_VERSION}"
        )
    if ring_length is not None and descriptor.ring_cursor >= ring_length:
        raise DSV41BoundaryStateError(
            f"ring cursor {descriptor.ring_cursor} exceeds ring length {ring_length}"
        )
    if slot_mapping is not None:
        validate_row_provenance(descriptor.row_indices, slot_mapping)


def boundary_snapshot_is_valid(
    commit: LayerwiseSnapshotCommit,
    context: BoundaryValidityContext,
    *,
    required_planes: int = COMPRESSOR_RING_PLANES,
) -> tuple[bool, str]:
    """Check one published snapshot against the current engine state.

    Args:
        commit: The snapshot commit whose boundaries are checked.
        context: Current engine facts (hashes, generation, ring, TTL).
        required_planes: Number of distinct ring planes (``compression_phase``
            values) the boundaries must cover — one descriptor per plane.

    Returns:
        ``(valid, reason)``; ``reason`` is empty when valid and names the
        first violated invariant otherwise. Never raises: an invalid
        snapshot is a cache miss, not an engine error.
    """
    if commit.plan_hash != context.plan_hash:
        return False, "plan_hash_mismatch"
    if commit.layout_hash != context.layout_hash:
        return False, "layout_hash_mismatch"
    if context.server_generation is not None and (
        commit.server_generation != context.server_generation
    ):
        return False, "server_generation_changed"
    if not commit.boundaries:
        return False, "no_boundary_state"
    covered_planes = {boundary.compression_phase for boundary in commit.boundaries}
    if covered_planes != set(range(required_planes)):
        return False, "incomplete_planes"
    for boundary in commit.boundaries:
        if boundary.schema_version != context.schema_version:
            return False, "schema_version_mismatch"
        distance = context.ring_distance(boundary.ring_cursor)
        if distance is not None and distance > RING_OVERWRITE_TOLERANCE:
            return False, "source_ring_overwritten"
    if context.now_monotonic is not None and context.snapshot_ttl_s is not None:
        committed_at = (context.committed_at or {}).get(commit.snapshot_id)
        if committed_at is None:
            return False, "commit_time_unknown"
        if context.now_monotonic - committed_at > context.snapshot_ttl_s:
            return False, "ttl_expired"
    return True, ""


def choose_fallback_snapshot(
    candidates: tuple[LayerwiseSnapshotCommit, ...],
    context: BoundaryValidityContext,
    *,
    required_planes: int = COMPRESSOR_RING_PLANES,
) -> LayerwiseSnapshotCommit | None:
    """Pick the newest still-valid complete boundary of the key.

    Args:
        candidates: Keyed snapshots ordered oldest -> newest.
        context: Current engine facts for validity checks.

    Returns:
        The newest valid snapshot, or ``None`` for a full miss. Callers must
        treat ``None`` as hit=0 and fall back to computing from scratch —
        never a partial restore.
    """
    for commit in reversed(candidates):
        valid, _ = boundary_snapshot_is_valid(
            commit, context, required_planes=required_planes
        )
        if valid:
            return commit
    return None
