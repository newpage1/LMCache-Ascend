# SPDX-License-Identifier: Apache-2.0
"""Diagnostic sampling for the DSV4.1 layerwise lane (t120 evidence run).

This recorder is OFF by default and must stay OFF in every default
deployment: it exists only for the t120 supplementary-sampling run
described in ``dsv41-layerwise/docs/t114-b0a-a0-evidence.md`` §4, whose
goal is to close the 24-layer observation gap with real
request-to-row mappings.

Enable (diagnostic runs only)::

    DSV41_SAMPLING=1
    DSV41_SAMPLING_DIR=/tmp/dsv41-sampling      # default
    DSV41_SAMPLING_MAX_EVENTS=200000            # per (tag, pid) file
    DSV41_SAMPLING_ROW_LIMIT=256                # max ints per tensor row
    DSV41_SAMPLING_SLOT_LIMIT=8192              # max ints for slot maps

One JSONL file is written per process per tag
(``sampling-<tag>-<pid>.jsonl``). Every writer failure (full disk,
permission, closed fd) is swallowed: diagnostics must never take down
serving. No prompt text is ever recorded; request ids are kept verbatim
per t120 contract.
"""

import hashlib
import json
import os
import threading
import time

__all__ = [
    "enabled",
    "emit",
    "row_limit",
    "slot_limit",
    "summarize_tensor",
    "tensor_ints",
    "tensor_digest",
]

_ENABLED = None
_ENABLED_LOCK = threading.Lock()
_WRITE_LOCK = threading.Lock()
_EVENT_COUNTS: dict[str, int] = {}


def enabled() -> bool:
    """Return True when DSV41_SAMPLING=1 (cached after first read)."""
    global _ENABLED
    if _ENABLED is None:
        with _ENABLED_LOCK:
            if _ENABLED is None:
                _ENABLED = os.environ.get("DSV41_SAMPLING", "").strip() == "1"
    return _ENABLED


def _max_events() -> int:
    try:
        return int(os.environ.get("DSV41_SAMPLING_MAX_EVENTS", "200000"))
    except ValueError:
        return 200000


def row_limit() -> int:
    """Max integers recorded per captured tensor row."""
    try:
        return int(os.environ.get("DSV41_SAMPLING_ROW_LIMIT", "256"))
    except ValueError:
        return 256


def slot_limit() -> int:
    """Max integers recorded for a full slot map capture."""
    try:
        return int(os.environ.get("DSV41_SAMPLING_SLOT_LIMIT", "8192"))
    except ValueError:
        return 8192


def _record(tag: str, event: str, fields: dict) -> None:
    count = _EVENT_COUNTS.get(tag, 0)
    if count >= _max_events():
        return
    directory = os.environ.get("DSV41_SAMPLING_DIR", "/tmp/dsv41-sampling")
    record = {"ts": time.time(), "pid": os.getpid(), "tag": tag}
    record.update(fields)
    record["event"] = event
    line = json.dumps(record, default=str, separators=(",", ":"))
    with _WRITE_LOCK:
        _EVENT_COUNTS[tag] = count + 1
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"sampling-{tag}-{os.getpid()}.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            # Diagnostics must never break serving: drop the record.
            pass


def emit(tag: str, event: str, **fields: object) -> None:
    """Append one sampling record when sampling is enabled.

    Cheap-field call sites may call unconditionally; call sites building
    tensor-derived fields must first check :func:`enabled`.
    """
    if not enabled():
        return
    _record(tag, event, fields)


def summarize_tensor(tensor: object) -> dict:
    """Summarize a tensor without copying its data.

    ``shape``/``stride`` may be tuples or bound methods (torch exposes
    them as methods); a callable is invoked once (the T56-01 lesson).
    """
    shape = getattr(tensor, "shape", None)
    if callable(shape):
        shape = shape()
    stride = getattr(tensor, "stride", None)
    if callable(stride):
        stride = stride()
    dtype = getattr(tensor, "dtype", None)
    data_ptr = getattr(tensor, "data_ptr", 0)
    if callable(data_ptr):
        data_ptr = data_ptr()
    return {
        "dtype": str(dtype),
        "shape": [int(dim) for dim in (shape or ())],
        "stride": [int(dim) for dim in (stride or ())],
        "data_ptr": hex(int(data_ptr or 0)),
    }


def tensor_ints(tensor: object, limit: int) -> list:
    """Read up to ``limit`` integer elements (bounded CPU copy).

    Returns [] when the tensor cannot be read (wrong dtype, device
    error, size 0); diagnostics record absence, never crash.
    """
    try:
        flat = tensor.reshape(-1)[:limit].detach().cpu().tolist()
        return [int(item) for item in flat]
    except Exception:  # noqa: BLE001 - diagnostics never raise
        return []


def tensor_digest(tensor: object, limit: int = 1 << 20) -> str | None:
    """SHA-256 over raw bytes (bounded to ``limit`` elements)."""
    try:
        raw = tensor.reshape(-1)[:limit].detach().cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    except Exception:  # noqa: BLE001 - diagnostics never raise
        return None
