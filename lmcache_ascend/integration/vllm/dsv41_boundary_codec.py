# SPDX-License-Identifier: Apache-2.0
"""C2 boundary-state codec for the DSV4.1 layerwise transfer (S1).

The S0 transport moved raw token-range KV bytes but left snapshot
boundaries empty, so a lookup could never hit. S1 adds the *boundary
state* half of the contract: alongside the token-range bytes, the owning
worker encodes the boundary rows of the engine-side ring buffer (the rows
the engine must restore to resume KV production at a token boundary)
into a self-describing payload, described by a checksummed
:class:`BoundaryStateDescriptor` (see
``lmcache.v1.multiprocess.layerwise_protocol``).

Wire format (``encode_boundary_rows``):

    [4-byte big-endian header length][JSON header][row bytes]

Two header versions exist:

``v1`` (legacy, sparse rows)
    The header is ``{"v": 1, "dtype": ..., "shape": [rows, cols],
    "rows": [...]}`` and the body is the selected rows concatenated in
    row order. Sparse payloads carry no ring geometry, so the transport
    restore path *rejects* them (a sparse snapshot cannot be rebased
    onto a new request page without writing wrong rows or leaving stale
    slots). Pure round-trip decoding stays supported so engine-side
    bridges built against v1 keep working.

``v2`` (schema v2, full-page with ring integrity)
    The header is ``{"v": 2, "dtype": ..., "shape": [ring_length, cols],
    "ring_length": R, "page_offset": P, "ring_cursor": C,
    "compression_phase": ph, "written_rows": [...], "boundary_token": t}``
    and the body is the **whole ring page** (``R`` rows starting at
    absolute row ``P``), with rows the engine did not write in the
    captured step normalized to zero. Per the layerwise plan (§6.2) the
    first version copies the complete ring page per request and
    normalizes invalid rows, so checksums stay stable and a restore can
    overwrite every target slot — including the ring tail — instead of
    leaving another request's bytes behind. ``page_offset`` is capture
    provenance only: restores never write back at the captured physical
    page, they rebase onto the target request's page.

The codec is deliberately strict: rows must be real indices into the
tensor the engine hands over (unique, in range), never guesses — the
protocol docstring already states that guessed row indices are rejected
by this codec.
"""

# Standard
from typing import Any

# Third Party
from lmcache.v1.multiprocess.layerwise_protocol import (
    BOUNDARY_COMPRESSION_PHASES,
    BOUNDARY_STATE_SCHEMA_VERSION,
    BoundaryStateDescriptor,
    compute_boundary_checksum,
)

_HEADER_VERSION = 1
_HEADER_VERSION_V2 = 2


def _torch():
    # Third Party
    import torch

    return torch


def _normalize_rows(rows: tuple[int, ...]) -> tuple[int, ...]:
    if not rows:
        raise ValueError("boundary row indices must not be empty")
    if len(set(rows)) != len(rows):
        raise ValueError(f"boundary row indices must be unique, got {rows}")
    if any(index < 0 for index in rows):
        raise ValueError(f"boundary row indices must be non-negative, got {rows}")
    return rows


def _page_block(tensor: Any, page_start: int, ring_length: int) -> Any:
    """Slice ``[page_start, page_start + ring_length)`` as a 2D block."""
    torch = _torch()
    if tensor.dim() < 2:
        raise ValueError(
            f"boundary tensor must be at least 2D, got shape {tuple(tensor.shape)}"
        )
    if page_start < 0 or page_start + ring_length > tensor.shape[0]:
        raise ValueError(
            f"boundary ring page [{page_start}, {page_start + ring_length}) is "
            f"out of range for {tensor.shape[0]} rows"
        )
    page = tensor[page_start : page_start + ring_length]
    cols = 1
    for dim in page.shape[1:]:
        cols *= int(dim)
    return page.reshape(ring_length, cols).detach().to("cpu").contiguous(), cols


def encode_boundary_rows(
    tensor: Any,
    row_indices: Any,
    *,
    snapshot_id: str,
    resource_id: str,
    boundary_token: str,
    ring_cursor: int,
    compression_phase: int,
    ring_length: int | None = None,
    page_offset: int | None = None,
) -> tuple[BoundaryStateDescriptor, bytes]:
    """Encode boundary rows of ``tensor`` for the wire.

    v2 full-page mode (preferred; pass ``ring_length``)
        ``tensor`` may be any >=2D tensor whose dim 0 is the absolute
        row space of the engine ring (extra trailing dimensions are
        flattened for the wire and recorded in the header shape).
        ``row_indices`` are the absolute rows the engine actually wrote
        for the captured request; they must all live inside one ring
        page. The payload carries the **entire** page with the
        unwritten rows normalized to zero, plus the ring integrity
        fields (``ring_length``, ``page_offset``, ``ring_cursor``,
        ``compression_phase``, ``written_rows``). The descriptor's
        ``row_indices`` identity spans the full page, so a restore can
        verify ring integrity before rebasing.

    v1 sparse mode (legacy; ``ring_length`` omitted)
        ``tensor`` must be 2D and the selected rows travel as-is with a
        v1 header. Kept for engine-side bridges; the transport restore
        path rejects v1 payloads.

    Args:
        tensor: Engine-side tensor holding the boundary rows.
        row_indices: Absolute indices of the rows the engine wrote.
            Must be a non-empty sequence of unique in-range indices; a
            guess or duplicate is rejected (ValueError) rather than
            silently encoded.
        snapshot_id: Snapshot the rows belong to.
        resource_id: Boundary-state resource the rows are stored under.
        boundary_token: Token boundary the rows represent (string form,
            as carried by the protocol descriptor).
        ring_cursor: Engine ring-buffer cursor at capture time.
        compression_phase: C2 compression phase (0 or 1).
        ring_length: Rows per ring page; enables v2 full-page framing.
        page_offset: Absolute row of the page start. Required when it
            cannot be derived (``row_indices[0] // ring_length`` is used
            when omitted); validated against ``row_indices``.

    Returns:
        ``(descriptor, payload)`` where the descriptor checksums the full
        identity and ``payload`` is the self-describing wire bytes.

    Raises:
        ValueError: On empty/duplicate/out-of-range row indices, an
            unknown compression phase, shape/geometry mismatches, or
            rows spanning more than one ring page.
    """
    if compression_phase not in BOUNDARY_COMPRESSION_PHASES:
        raise ValueError(f"unknown compression phase {compression_phase}")
    rows = _normalize_rows(tuple(int(index) for index in row_indices))
    torch = _torch()
    import json

    if ring_length is None:
        # Legacy sparse framing (v1). Kept byte-compatible with the S1
        # bridges; the transport restore path rejects these payloads.
        if tensor.dim() != 2:
            raise ValueError(
                f"boundary tensor must be 2D, got shape {tuple(tensor.shape)}"
            )
        if max(rows) >= tensor.shape[0]:
            raise ValueError(
                f"boundary row index {max(rows)} out of range for "
                f"{tensor.shape[0]} rows"
            )
        block = tensor[list(rows)].detach().to("cpu").contiguous()
        header = {
            "v": _HEADER_VERSION,
            "dtype": str(block.dtype).removeprefix("torch."),
            "shape": [int(block.shape[0]), int(block.shape[1])],
            "rows": list(rows),
        }
        page_rows: tuple[int, ...] = rows
        ring_length_out = int(block.shape[0])
    else:
        ring_length_out = int(ring_length)
        if ring_length_out <= 0:
            raise ValueError(f"ring_length must be positive, got {ring_length}")
        page_start = (
            int(page_offset)
            if page_offset is not None
            else rows[0] // ring_length_out * ring_length_out
        )
        page_start = page_start // ring_length_out * ring_length_out
        if page_start < 0:
            raise ValueError(f"page_offset {page_offset} is negative")
        page_end = page_start + ring_length_out
        if any(row < page_start or row >= page_end for row in rows):
            raise ValueError(
                "boundary rows must stay inside one ring page: rows "
                f"{rows} vs page [{page_start}, {page_end})"
            )
        block, _cols = _page_block(tensor, page_start, ring_length_out)
        written_local = sorted(row - page_start for row in rows)
        # Normalize the rows the engine did not write this step so the
        # payload never ships uninitialized bytes (plan §6.2); a restore
        # then overwrites every target slot including the ring tail.
        page = block.clone()
        written_mask = torch.zeros(ring_length_out, dtype=torch.bool)
        written_mask[list(written_local)] = True
        page[~written_mask] = 0
        header = {
            "v": _HEADER_VERSION_V2,
            "dtype": str(page.dtype).removeprefix("torch."),
            "shape": [int(page.shape[0]), int(page.shape[1])],
            "ring_length": ring_length_out,
            "page_offset": int(page_start),
            "ring_cursor": int(ring_cursor),
            "compression_phase": int(compression_phase),
            "written_rows": list(written_local),
            "boundary_token": str(boundary_token),
        }
        block = page
        # Full-page identity: absolute rows of the captured page.
        page_rows = tuple(range(page_start, page_end))

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if block.numel():
        flat = block.view(torch.uint8).reshape(-1)
        body_bytes = bytes(flat.tolist())
    else:
        body_bytes = b""
    payload = len(header_bytes).to_bytes(4, "big") + header_bytes + body_bytes

    descriptor = BoundaryStateDescriptor(
        snapshot_id=snapshot_id,
        resource_id=resource_id,
        boundary_token=str(boundary_token),
        ring_cursor=int(ring_cursor),
        compression_phase=int(compression_phase),
        row_indices=page_rows,
        schema_version=BOUNDARY_STATE_SCHEMA_VERSION,
        checksum=compute_boundary_checksum(
            snapshot_id,
            resource_id,
            str(boundary_token),
            int(ring_cursor),
            int(compression_phase),
            page_rows,
            BOUNDARY_STATE_SCHEMA_VERSION,
        ),
    )
    return descriptor, payload


def decode_boundary_rows(payload: bytes) -> dict[str, Any]:
    """Decode boundary wire bytes back into ``(header, rows_tensor)``.

    Both header versions decode (``header["v"]`` distinguishes them) so
    engine-side round-trips keep working; the *transport restore* path
    additionally rejects v1 payloads because sparse rows cannot be
    rebased onto a new request page.

    Returns:
        Dict with ``header`` (parsed JSON) and ``rows`` (2D tensor with
        the encoded dtype/shape restored).

    Raises:
        ValueError: On truncated payloads, unknown header versions or
            dtypes this process cannot reconstruct.
    """
    import json

    if len(payload) < 4:
        raise ValueError("boundary payload truncated before header length")
    header_len = int.from_bytes(payload[:4], "big")
    if len(payload) < 4 + header_len:
        raise ValueError("boundary payload truncated inside header")
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    version = header.get("v")
    if version not in (_HEADER_VERSION, _HEADER_VERSION_V2):
        raise ValueError(f"unsupported boundary payload version {version}")
    torch = _torch()
    rows = int(header["shape"][0])
    cols = int(header["shape"][1])
    dtype = getattr(torch, str(header["dtype"]), None)
    if dtype is None:
        raise ValueError(f"unknown boundary dtype {header['dtype']}")
    body = payload[4 + header_len :]
    expected = rows * cols * torch.empty((0,), dtype=dtype).element_size()
    if len(body) != expected:
        raise ValueError(
            f"boundary body is {len(body)} bytes, expected {expected}"
        )
    block = (
        torch.frombuffer(bytearray(body), dtype=dtype)
        if body
        else torch.empty((rows, cols), dtype=dtype)
    )
    return {"header": header, "rows": block.reshape(rows, cols)}
