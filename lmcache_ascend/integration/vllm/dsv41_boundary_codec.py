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

The header is ``{"v": 1, "dtype": ..., "shape": [rows, cols], "rows": [...]}``
and the body is the selected rows concatenated in row order. Decoding
(``decode_boundary_rows``) validates the version, rebuilds the row block
and returns it; the descriptor checksum binds the *content identity*
(snapshot, resource, boundary token, ring cursor, phase, row indices).

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


def _torch():
    # Third Party
    import torch

    return torch


def encode_boundary_rows(
    tensor: Any,
    row_indices: Any,
    *,
    snapshot_id: str,
    resource_id: str,
    boundary_token: str,
    ring_cursor: int,
    compression_phase: int,
) -> tuple[BoundaryStateDescriptor, bytes]:
    """Encode the selected boundary rows of ``tensor`` for the wire.

    Args:
        tensor: 2D engine-side tensor holding the boundary rows (any
            dtype; rows are read as-is and restored with the same dtype).
        row_indices: Indices of the boundary rows inside ``tensor``.
            Must be a non-empty sequence of unique in-range indices; a
            guess or duplicate is rejected (ValueError) rather than
            silently encoded.
        snapshot_id: Snapshot the rows belong to.
        resource_id: Boundary-state resource the rows are stored under.
        boundary_token: Token boundary the rows represent (string form,
            as carried by the protocol descriptor).
        ring_cursor: Engine ring-buffer cursor at capture time.
        compression_phase: C2 compression phase (0 or 1).

    Returns:
        ``(descriptor, payload)`` where the descriptor checksums the full
        identity and ``payload`` is the self-describing wire bytes.

    Raises:
        ValueError: On empty/duplicate/out-of-range row indices, a
            non-2D tensor, or an unknown compression phase.
    """
    if compression_phase not in BOUNDARY_COMPRESSION_PHASES:
        raise ValueError(f"unknown compression phase {compression_phase}")
    rows = tuple(int(index) for index in row_indices)
    if not rows:
        raise ValueError("boundary row indices must not be empty")
    if len(set(rows)) != len(rows):
        raise ValueError(f"boundary row indices must be unique, got {rows}")
    if any(index < 0 for index in rows):
        raise ValueError(f"boundary row indices must be non-negative, got {rows}")
    if tensor.dim() != 2:
        raise ValueError(
            f"boundary tensor must be 2D, got shape {tuple(tensor.shape)}"
        )
    if max(rows) >= tensor.shape[0]:
        raise ValueError(
            f"boundary row index {max(rows)} out of range for "
            f"{tensor.shape[0]} rows"
        )

    torch = _torch()
    import json

    block = tensor[list(rows)].detach().to("cpu").contiguous()
    header = {
        "v": _HEADER_VERSION,
        "dtype": str(block.dtype).removeprefix("torch."),
        "shape": [int(block.shape[0]), int(block.shape[1])],
        "rows": list(rows),
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if block.numel():
        flat = block.view(torch.uint8).reshape(-1)
        body_bytes = bytes(flat.tolist())
    else:
        body_bytes = b""
    payload = (
        len(header_bytes).to_bytes(4, "big") + header_bytes + body_bytes
    )

    descriptor = BoundaryStateDescriptor(
        snapshot_id=snapshot_id,
        resource_id=resource_id,
        boundary_token=str(boundary_token),
        ring_cursor=int(ring_cursor),
        compression_phase=int(compression_phase),
        row_indices=rows,
        schema_version=BOUNDARY_STATE_SCHEMA_VERSION,
        checksum=compute_boundary_checksum(
            snapshot_id,
            resource_id,
            str(boundary_token),
            int(ring_cursor),
            int(compression_phase),
            rows,
            BOUNDARY_STATE_SCHEMA_VERSION,
        ),
    )
    return descriptor, payload


def decode_boundary_rows(payload: bytes) -> dict[str, Any]:
    """Decode boundary wire bytes back into ``(header, rows_tensor)``.

    Args:
        payload: Bytes produced by :func:`encode_boundary_rows`.

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
    if header.get("v") != _HEADER_VERSION:
        raise ValueError(f"unsupported boundary payload version {header.get('v')}")
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
