# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layerwise v2 schema-2 boundary codec.

The engine side (vLLM-Ascend ``dsv41_lw_v2_boundary``) owns the live compressor
ring and calls this codec at the three boundary group heads (G2, G8, G14):

* :meth:`DSV41LwV2BoundaryCodec.capture_layer_boundary` freezes one immutable
  full ring page after the source forward. It copies the page on the device
  stream, records the copy event and hands it to the transport adapter; the
  device-to-host copy and the schema-2 encoding happen on the adapter worker,
  never on the forward thread.
* :meth:`DSV41LwV2BoundaryCodec.restore_layer_boundary` validates a retrieved
  payload end-to-end (schema, identity, geometry, written rows, phase, digest),
  relocates it from the capture page to the current request's live page and
  writes both planes (KV and scores) before the kernel consumes them.

Wire format (schema 2 only)::

    b"DSV2BC1\\n" | u32 header_len | UTF-8 JSON header | kv bytes | scores bytes | sha256

The digest covers the canonical JSON header and both payload planes. The old
``compute_boundary_checksum`` does not cover geometry, written rows or payload
bytes and is therefore not used as an end-to-end proof here.

A missing binding or payload on a real boundary hit is a hard error: this
codec never returns silently as if the page had been restored.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
import hashlib
import json
import struct

# Third Party
import torch

# First Party
from lmcache_ascend.integration.vllm.dsv41_lw_v2.adapter import (
    BOUNDARY_PAYLOAD_MAGIC,
    DSV41LwV2GroupAdapter,
)

BOUNDARY_SCHEMA_VERSION = 2
BOUNDARY_NOT_BOUND = "BOUNDARY_NOT_BOUND"
BOUNDARY_PAYLOAD_MISSING = "BOUNDARY_PAYLOAD_MISSING"
BOUNDARY_PAYLOAD_CORRUPT = "BOUNDARY_PAYLOAD_CORRUPT"
BOUNDARY_PREFIX_MISMATCH = "BOUNDARY_PREFIX_MISMATCH"
BOUNDARY_PAGE_MISMATCH = "BOUNDARY_PAGE_MISMATCH"
BOUNDARY_GEOMETRY_MISMATCH = "BOUNDARY_GEOMETRY_MISMATCH"
BOUNDARY_SCHEMA_UNSUPPORTED = "BOUNDARY_SCHEMA_UNSUPPORTED"


class DSV41BoundaryStateError(RuntimeError):
    """A boundary payload or binding failed schema-2 validation."""

    def __init__(self, message: str, *, error_code: str) -> None:
        super().__init__(f"{message} (error_code={error_code})")
        self.error_code = error_code


class DSV41BoundaryRestoreError(DSV41BoundaryStateError):
    """A boundary restore could not write the live page; recompute required."""


class DSV41BoundaryNotBoundError(DSV41BoundaryStateError):
    """No codec is installed for a real boundary hit."""


@dataclass(frozen=True)
class BoundaryEnvelope:
    """Decoded schema-2 boundary envelope."""

    header: dict[str, Any]
    kv: torch.Tensor
    scores: torch.Tensor


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping.get(name)
    if dtype is None:
        raise DSV41BoundaryStateError(
            f"unsupported boundary plane dtype {name!r}",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    return dtype


def _canonical_header(header: Mapping[str, Any]) -> bytes:
    return json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def encode_boundary_schema2(
    header: Mapping[str, Any],
    kv: torch.Tensor,
    scores: torch.Tensor,
) -> bytes:
    """Encode a full-page schema-2 boundary payload.

    Args:
        header: Canonical boundary metadata (already validated by the codec).
        kv: KV plane of shape ``(ring_rows, head_size)`` on CPU.
        scores: scores plane of the same shape on CPU.

    Returns:
        The self-describing payload bytes.

    Raises:
        DSV41BoundaryStateError: if a plane is not CPU contiguous or the
            declared geometry does not match the tensors.
    """
    if kv.device.type != "cpu" or scores.device.type != "cpu":
        raise DSV41BoundaryStateError(
            "boundary payload encoding requires CPU planes",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    kv = kv.contiguous()
    scores = scores.contiguous()
    header = dict(header)
    header["schema"] = BOUNDARY_SCHEMA_VERSION
    header["kv"] = {
        "dtype": str(kv.dtype).removeprefix("torch."),
        "shape": [int(dim) for dim in kv.shape],
        "bytes": kv.numel() * kv.element_size(),
    }
    header["scores"] = {
        "dtype": str(scores.dtype).removeprefix("torch."),
        "shape": [int(dim) for dim in scores.shape],
        "bytes": scores.numel() * scores.element_size(),
    }
    header_bytes = _canonical_header(header)
    kv_bytes = _tensor_bytes(kv)
    score_bytes = _tensor_bytes(scores)
    digest = hashlib.sha256(
        header_bytes + kv_bytes + score_bytes
    ).digest()
    return (
        BOUNDARY_PAYLOAD_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + kv_bytes
        + score_bytes
        + digest
    )


def decode_boundary_schema2(payload: bytes) -> BoundaryEnvelope:
    """Decode and integrity-check a schema-2 boundary payload.

    Raises:
        DSV41BoundaryStateError: on schema-1 input, truncation, trailing
            bytes, digest mismatch or geometry corruption.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise DSV41BoundaryStateError(
            "boundary payload is not bytes",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    payload = bytes(payload)
    if not payload.startswith(BOUNDARY_PAYLOAD_MAGIC):
        raise DSV41BoundaryStateError(
            "boundary payload is not schema 2 (schema 1 is not accepted on "
            "the v2 path)",
            error_code=BOUNDARY_SCHEMA_UNSUPPORTED,
        )
    offset = len(BOUNDARY_PAYLOAD_MAGIC)
    if len(payload) < offset + 4 + 32:
        raise DSV41BoundaryStateError(
            "boundary payload is truncated",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    (header_length,) = struct.unpack(">I", payload[offset : offset + 4])
    offset += 4
    if header_length > 1 << 20:
        raise DSV41BoundaryStateError(
            "boundary header is unreasonably large",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    if len(payload) < offset + header_length + 32:
        raise DSV41BoundaryStateError(
            "boundary header is truncated",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    header_bytes = payload[offset : offset + header_length]
    offset += header_length
    digest = payload[-32:]
    body = payload[:-32]
    expected = hashlib.sha256(body[len(BOUNDARY_PAYLOAD_MAGIC) + 4 :]).digest()
    if not _constant_time_equal(digest, expected):
        raise DSV41BoundaryStateError(
            "boundary payload digest does not match",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DSV41BoundaryStateError(
            f"boundary header is not valid JSON: {exc}",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        ) from None
    if not isinstance(header, dict) or header.get("schema") != 2:
        raise DSV41BoundaryStateError(
            "boundary header schema is not 2",
            error_code=BOUNDARY_SCHEMA_UNSUPPORTED,
        )
    kv_header = header.get("kv")
    score_header = header.get("scores")
    if not isinstance(kv_header, dict) or not isinstance(score_header, dict):
        raise DSV41BoundaryStateError(
            "boundary header has no plane descriptors",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    kv_bytes = _read_plane(payload, offset, kv_header, "kv")
    offset += kv_bytes[0]
    score_bytes = _read_plane(payload, offset, score_header, "scores")
    offset += score_bytes[0]
    if offset != len(body):
        raise DSV41BoundaryStateError(
            "boundary payload has trailing bytes",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    kv = _tensor_from_plane(kv_bytes[1], kv_header)
    scores = _tensor_from_plane(score_bytes[1], score_header)
    if tuple(kv.shape) != tuple(scores.shape):
        raise DSV41BoundaryStateError(
            "boundary KV and score planes have different shapes",
            error_code=BOUNDARY_GEOMETRY_MISMATCH,
        )
    return BoundaryEnvelope(header=header, kv=kv, scores=scores)


def _constant_time_equal(left: bytes, right: bytes) -> bool:
    if len(left) != len(right):
        return False
    result = 0
    for a, b in zip(left, right):
        result |= a ^ b
    return result == 0


def _read_plane(
    payload: bytes, offset: int, header: Mapping[str, Any], name: str
) -> tuple[int, bytes]:
    dtype_name = header.get("dtype")
    _torch_dtype(dtype_name)
    shape = tuple(int(dim) for dim in header.get("shape", ()))
    if len(shape) != 2 or shape[0] < 0 or shape[1] <= 0:
        raise DSV41BoundaryStateError(
            f"boundary {name} plane declares invalid shape {shape}",
            error_code=BOUNDARY_GEOMETRY_MISMATCH,
        )
    byte_count = int(header.get("bytes", -1))
    if byte_count < 0 or len(payload) < offset + byte_count:
        raise DSV41BoundaryStateError(
            f"boundary {name} plane is truncated",
            error_code=BOUNDARY_PAYLOAD_CORRUPT,
        )
    return byte_count, payload[offset : offset + byte_count]


def _tensor_from_plane(
    raw: bytes, header: Mapping[str, Any]
) -> torch.Tensor:
    dtype = _torch_dtype(str(header.get("dtype")))
    shape = tuple(int(dim) for dim in header.get("shape", ()))
    expected = dtype.itemsize
    for dim in shape:
        expected *= dim
    if expected != len(raw):
        raise DSV41BoundaryStateError(
            f"boundary plane declares {len(raw)} bytes but shape {shape} "
            f"needs {expected}",
            error_code=BOUNDARY_GEOMETRY_MISMATCH,
        )
    if dtype == torch.bfloat16:
        return torch.frombuffer(bytearray(raw), dtype=torch.int16).view(
            torch.bfloat16
        ).reshape(shape)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().tobytes()
    return tensor.numpy().tobytes()


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


class DSV41LwV2BoundaryCodec:
    """Schema-2 boundary codec wired to a group transport adapter.

    Args:
        adapter: Transport adapter that stages captured payloads and serves
            retrieved boundary bytes.
        allowed_resource_ids: Boundary resource ids the plan may carry; a
            capture/restore for any other resource is rejected.
    """

    def __init__(
        self,
        adapter: DSV41LwV2GroupAdapter,
        *,
        allowed_resource_ids: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(adapter, DSV41LwV2GroupAdapter):
            raise TypeError("adapter must be a DSV41LwV2GroupAdapter")
        allowed = (
            tuple(adapter.boundary_resource_ids)
            if allowed_resource_ids is None
            else tuple(allowed_resource_ids)
        )
        if not allowed:
            raise ValueError("the plan has no boundary resources")
        for resource_id in allowed:
            if not resource_id.endswith(".compressor.state_cache"):
                raise ValueError(
                    f"boundary resource {resource_id!r} is not a compressor "
                    "state cache"
                )
        self._adapter = adapter
        self._allowed = frozenset(allowed)

    @property
    def adapter(self) -> DSV41LwV2GroupAdapter:
        return self._adapter

    def capture_layer_boundary(
        self,
        request_view: Any,
        tensor_view: Any,
        *,
        stream: Any = None,
    ) -> None:
        """Freeze one full ring page for a real request boundary.

        Args:
            request_view: Engine ``BoundaryRequestView`` (duck typed).
            tensor_view: Engine ``BoundaryTensorView`` with the live ring.
            stream: Optional device stream the engine captured on.

        Raises:
            DSV41BoundaryStateError: on an unknown resource, bad page or
                geometry, or when the capture cannot be staged.
        """
        resource_id = str(getattr(request_view, "resource_id", ""))
        self._require_resource(resource_id)
        ring_rows, head_size = self._geometry(tensor_view)
        request_index = int(request_view.request_index)
        page_ids = tensor_view.page_ids
        if request_index < 0 or request_index >= len(page_ids):
            raise DSV41BoundaryStateError(
                f"request index {request_index} has no ring page",
                error_code=BOUNDARY_PAGE_MISMATCH,
            )
        capture_page = int(page_ids[request_index])
        if capture_page < 0 or capture_page >= int(tensor_view.ring.shape[0]):
            raise DSV41BoundaryStateError(
                f"capture page {capture_page} is outside the ring",
                error_code=BOUNDARY_PAGE_MISMATCH,
            )
        if int(request_view.ring_page) != capture_page:
            raise DSV41BoundaryStateError(
                f"capture view page {request_view.ring_page} does not match "
                f"the live page {capture_page}",
                error_code=BOUNDARY_PAGE_MISMATCH,
            )
        self._validate_view(request_view)
        page = tensor_view.ring[capture_page]
        if page.ndim != 2 or page.shape[0] != ring_rows:
            raise DSV41BoundaryStateError(
                f"ring page shape {tuple(page.shape)} does not match "
                f"({ring_rows}, 2*{head_size})",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        if page.shape[1] != 2 * head_size:
            raise DSV41BoundaryStateError(
                f"ring page width {page.shape[1]} is not "
                f"2*{head_size}",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        device = page.device
        copy_stream = stream
        if device.type == "cuda":
            copy_stream = stream or torch.cuda.current_stream(device)
            with torch.cuda.stream(copy_stream):
                device_page = page.clone()
                event = torch.cuda.Event()
                event.record(copy_stream)
        else:
            device_page = page.clone()
            event = None
        header = self._capture_header(request_view)
        adapter = self._adapter

        def encode(cpu_page: torch.Tensor) -> bytes:
            kv = cpu_page[:, :head_size].detach().contiguous()
            scores = cpu_page[:, head_size:].detach().contiguous()
            return encode_boundary_schema2(header, kv, scores)

        try:
            adapter.stage_boundary_capture(
                str(request_view.request_id),
                resource_id,
                int(getattr(request_view, "boundary_end", 0)),
                device_page,
                event,
                encode,
                stream=copy_stream,
            )
        except (ValueError, KeyError) as exc:
            raise DSV41BoundaryStateError(
                f"boundary capture could not be staged: {exc}",
                error_code=BOUNDARY_PAYLOAD_CORRUPT,
            ) from None
        adapter.record_boundary_staged()

    def restore_layer_boundary(
        self,
        request_view: Any,
        tensor_view: Any,
        *,
        stream: Any = None,
    ) -> None:
        """Validate and write a retrieved full page into the live ring.

        The stored endpoint is the capture-time ``boundary_end``; the lookup
        endpoint is the hit ``prefix_end``. A page captured in one request's
        ring page is relocated to this request's live page.

        Raises:
            DSV41BoundaryRestoreError: when the payload is missing, corrupt,
                mismatched or cannot be written. There is no silent no-op.
        """
        resource_id = str(getattr(request_view, "resource_id", ""))
        self._require_resource(resource_id)
        self._validate_view(request_view)
        prefix_end = int(request_view.prefix_end)
        payload = self._adapter.take_boundary_payload(
            str(request_view.request_id), resource_id, prefix_end
        )
        if payload is None:
            self._adapter.record_boundary_restore_failed()
            raise DSV41BoundaryRestoreError(
                "no boundary payload was staged for this hit",
                error_code=BOUNDARY_PAYLOAD_MISSING,
            )
        try:
            envelope = decode_boundary_schema2(payload)
            self._validate_restore(request_view, tensor_view, envelope)
        except DSV41BoundaryStateError:
            self._adapter.record_boundary_restore_failed()
            raise
        ring_rows, head_size = self._geometry(tensor_view)
        request_index = int(request_view.request_index)
        page_ids = tensor_view.page_ids
        if request_index < 0 or request_index >= len(page_ids):
            self._adapter.record_boundary_restore_failed()
            raise DSV41BoundaryRestoreError(
                f"request index {request_index} has no live ring page",
                error_code=BOUNDARY_PAGE_MISMATCH,
            )
        target_page = int(page_ids[request_index])
        if target_page < 0 or target_page >= int(tensor_view.ring.shape[0]):
            self._adapter.record_boundary_restore_failed()
            raise DSV41BoundaryRestoreError(
                f"live page {target_page} is outside the ring",
                error_code=BOUNDARY_PAGE_MISMATCH,
            )
        ring = tensor_view.ring
        try:
            kv_target = ring[target_page][:, :head_size]
            scores_target = ring[target_page][:, head_size:]
            if tuple(kv_target.shape) != (ring_rows, head_size):
                raise ValueError("live ring geometry changed")
            kv_source = envelope.kv.to(
                device=ring.device, dtype=ring.dtype, non_blocking=False
            )
            scores_source = envelope.scores.to(
                device=ring.device, dtype=ring.dtype, non_blocking=False
            )
            if device_is_cuda(ring):
                write_stream = stream or torch.cuda.current_stream(ring.device)
                with torch.cuda.stream(write_stream):
                    kv_target.copy_(kv_source)
                    scores_target.copy_(scores_source)
                write_stream.synchronize()
            else:
                kv_target.copy_(kv_source)
                scores_target.copy_(scores_source)
        except (RuntimeError, ValueError) as exc:
            self._adapter.record_boundary_restore_failed()
            raise DSV41BoundaryRestoreError(
                f"boundary restore could not write the live ring: {exc}",
                error_code=BOUNDARY_PAGE_MISMATCH,
            ) from None
        self._adapter.record_boundary_restored()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _require_resource(self, resource_id: str) -> None:
        if resource_id not in self._allowed:
            raise DSV41BoundaryStateError(
                f"resource {resource_id!r} is not a boundary resource of this "
                f"plan (allowed: {sorted(self._allowed)!r})",
                error_code=BOUNDARY_SCHEMA_UNSUPPORTED,
            )

    def _validate_view(self, view: Any) -> None:
        ratio = int(view.compression_ratio)
        phase = int(view.compression_phase)
        if ratio < 1 or not 0 <= phase < ratio:
            raise DSV41BoundaryStateError(
                f"invalid compression phase {phase}/{ratio}",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        ring_length = int(view.ring_length)
        if ring_length < 1:
            raise DSV41BoundaryStateError(
                "ring length must be positive",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        if int(view.next_slot) != int(view.prefix_end) % ring_length:
            raise DSV41BoundaryStateError(
                f"next slot {view.next_slot} does not match prefix_end "
                f"{view.prefix_end} modulo ring length {ring_length}",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        page = int(view.ring_page)
        if page < 0:
            raise DSV41BoundaryStateError(
                f"ring page {page} must be non-negative",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        # The engine DTO is the wire source of truth for ``written_rows``:
        # rows are *absolute* in the physical ring, i.e. the live-page window
        # is [ring_page * ring_length, (ring_page + 1) * ring_length).  Page
        # 0 therefore looks relative, but page N must not be rejected.
        written = tuple(int(row) for row in view.written_rows)
        expected_rows = min(int(view.step_tokens), ring_length)
        if len(written) != expected_rows:
            raise DSV41BoundaryStateError(
                f"written rows {len(written)} do not match "
                f"min(step_tokens, ring_length)={expected_rows}",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        if len(set(written)) != len(written):
            raise DSV41BoundaryStateError(
                "written rows repeat",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        page_start = page * ring_length
        page_end = page_start + ring_length
        for row in written:
            if not page_start <= row < page_end:
                raise DSV41BoundaryStateError(
                    f"written row {row} is outside the live page "
                    f"[{page_start}, {page_end}) for ring page {page}",
                    error_code=BOUNDARY_GEOMETRY_MISMATCH,
                )
        control = tuple(view.position_control)
        if len(control) != 5 or any(
            type(value) is not int for value in control
        ):
            raise DSV41BoundaryStateError(
                "position control must be five ints",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )

    def _capture_header(self, view: Any) -> dict[str, Any]:
        return {
            "schema": BOUNDARY_SCHEMA_VERSION,
            "resource_id": str(view.resource_id),
            "request_id": str(view.request_id),
            "step_id": int(view.step_id),
            "registration_epoch": int(view.registration_epoch),
            "server_generation": int(view.server_generation),
            "dispatch_layer": str(view.dispatch_layer),
            "execution_layer": str(view.execution_layer),
            "prefix_end": int(view.prefix_end),
            "boundary_end": int(view.boundary_end),
            "step_tokens": int(view.step_tokens),
            "ring_length": int(view.ring_length),
            "ring_page": int(view.ring_page),
            "next_slot": int(view.next_slot),
            "compression_ratio": int(view.compression_ratio),
            "compression_phase": int(view.compression_phase),
            "position_control": [int(value) for value in view.position_control],
            "written_rows": [int(row) for row in view.written_rows],
        }

    def _validate_restore(
        self,
        view: Any,
        tensor_view: Any,
        envelope: BoundaryEnvelope,
    ) -> None:
        header = envelope.header
        # The capture records its logical end (``boundary_end``); a hit is
        # looked up by the prefix end it serves, so those two must agree.
        checks = (
            # A boundary page is shared by every request with the same
            # logical prefix, so the capture's request id is provenance only:
            # validating it would break the cross-request stock hit.
            ("resource_id", str(view.resource_id)),
            ("dispatch_layer", str(view.dispatch_layer)),
            ("execution_layer", str(view.execution_layer)),
            ("boundary_end", int(view.prefix_end)),
            ("ring_length", int(view.ring_length)),
            ("compression_ratio", int(view.compression_ratio)),
        )
        # ``next_slot`` / ``compression_phase`` / ``position_control`` are
        # live kernel state of the consuming request; the captured page is
        # shared across requests, so only the page geometry and endpoint are
        # identity-bearing.  The live view's values are what the kernel keeps.
        if int(header.get("compression_phase", -1)) < 0:
            raise DSV41BoundaryStateError(
                "boundary payload has no compression phase",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        for field_name, expected in checks:
            actual = header.get(field_name)
            if actual != expected:
                code = (
                    BOUNDARY_PREFIX_MISMATCH
                    if field_name == "boundary_end"
                    else BOUNDARY_GEOMETRY_MISMATCH
                )
                raise DSV41BoundaryStateError(
                    f"boundary payload {field_name} {actual!r} does not match "
                    f"the restore view {expected!r}",
                    error_code=code,
                )
        self._validate_header_rows(header)
        ring_rows, head_size = int(tensor_view.ring_rows), int(
            tensor_view.ring_head_size
        )
        if tuple(envelope.kv.shape) != (ring_rows, head_size):
            raise DSV41BoundaryStateError(
                f"boundary plane shape {tuple(envelope.kv.shape)} does not "
                f"match the live ring ({ring_rows}, {head_size})",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        ring = tensor_view.ring
        if tuple(ring.shape[1:]) != (ring_rows, 2 * head_size):
            raise DSV41BoundaryStateError(
                f"live ring shape {tuple(ring.shape)} is not "
                f"[pages, {ring_rows}, 2*{head_size}]",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        if envelope.kv.dtype != torch.float32:
            raise DSV41BoundaryStateError(
                f"boundary planes must be float32, got "
                f"{envelope.kv.dtype}",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )

    def _validate_header_rows(self, header: dict[str, Any]) -> None:
        """Validate the captured absolute rows against the captured page.

        The rows are provenance of the capture (which absolute ring rows that
        step wrote); a restore happens in a *different* request's live page and
        has its own ``written_rows``.  Comparing the two would reject every
        cross-page stock hit, so the header is validated against its own page
        window instead.
        """
        ring_length = header.get("ring_length")
        ring_page = header.get("ring_page")
        step_tokens = header.get("step_tokens")
        if not all(
            type(value) is int
            for value in (ring_length, ring_page, step_tokens)
        ):
            raise DSV41BoundaryStateError(
                "boundary payload is missing its ring geometry",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        if ring_length <= 0 or ring_page < 0 or step_tokens < 0:
            raise DSV41BoundaryStateError(
                f"boundary payload has invalid ring geometry "
                f"(page={ring_page}, length={ring_length}, "
                f"step_tokens={step_tokens})",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        rows = header.get("written_rows")
        if not isinstance(rows, list) or any(
            type(row) is not int for row in rows
        ):
            raise DSV41BoundaryStateError(
                "boundary payload written rows are not a list of ints",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        expected = min(step_tokens, ring_length)
        if len(rows) != expected:
            raise DSV41BoundaryStateError(
                f"boundary payload has {len(rows)} written rows, expected "
                f"min(step_tokens, ring_length)={expected}",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        if len(set(rows)) != len(rows):
            raise DSV41BoundaryStateError(
                "boundary payload written rows repeat",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        page_start = ring_page * ring_length
        page_end = page_start + ring_length
        for row in rows:
            if not page_start <= row < page_end:
                raise DSV41BoundaryStateError(
                    f"boundary payload written row {row} is outside its "
                    f"capture page [{page_start}, {page_end})",
                    error_code=BOUNDARY_GEOMETRY_MISMATCH,
                )

    @staticmethod
    def _geometry(tensor_view: Any) -> tuple[int, int]:
        ring_rows = int(tensor_view.ring_rows)
        head_size = int(tensor_view.ring_head_size)
        if ring_rows < 1 or head_size < 1:
            raise DSV41BoundaryStateError(
                "ring geometry must be positive",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        ring = tensor_view.ring
        if ring.dtype != torch.float32:
            raise DSV41BoundaryStateError(
                f"boundary ring dtype {ring.dtype} is not float32",
                error_code=BOUNDARY_GEOMETRY_MISMATCH,
            )
        return ring_rows, head_size


def device_is_cuda(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda"


def install_dsv41_lw_v2_boundary_codec(
    connector: Any,
    *,
    adapter: DSV41LwV2GroupAdapter,
    allowed_resource_ids: Sequence[str] | None = None,
) -> DSV41LwV2BoundaryCodec:
    """Create and install the production codec on a v2 connector.

    This is the single production entry point: the connector exposes the
    installed codec through ``connector.boundary_codec`` so the engine-side
    ``install_from_connector`` sees exactly one binding before the first
    forward.

    Args:
        connector: ``DSV41LwV2Connector`` instance.
        adapter: Adapter that stages captured pages and serves retrieves.
        allowed_resource_ids: Optional restricted boundary resource set.

    Returns:
        The installed codec.

    Raises:
        TypeError: when the connector has no v2 codec seam.
    """
    setter = getattr(connector, "set_boundary_codec", None)
    if setter is None:
        raise TypeError(
            "connector does not expose set_boundary_codec; is it a "
            "DSV41LwV2Connector?"
        )
    codec = DSV41LwV2BoundaryCodec(
        adapter, allowed_resource_ids=allowed_resource_ids
    )
    setter(codec)
    return codec
