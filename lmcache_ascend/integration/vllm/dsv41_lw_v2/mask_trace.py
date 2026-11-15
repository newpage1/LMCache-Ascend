# SPDX-License-Identifier: Apache-2.0
"""Default-off JSONL timeline for the layerwise v2 group transport.

The trace exists to quantify how much of a group load's wait is hidden behind
the previous layer's forward, so it must be *free* when disabled and must never
change the order of transport operations.  It is therefore a plain append-only
JSONL sink with three properties:

* ``DSV41_LW_V2_MASK_TRACE`` must be exactly ``1``; otherwise every entry
  point returns after one cached boolean check and no file is opened.
* Event timestamps use :func:`time.monotonic_ns`; nothing here calls a device
  synchronisation primitive or a blocking RPC.
* Writing is bounded twice: per request (env
  ``DSV41_LW_V2_MASK_TRACE_MAX_LINES_PER_REQUEST``, default 20000) and per
  process (``DSV41_LW_V2_MASK_TRACE_MAX_LINES``, default 2000000).  A request that exceeds its budget gets one ``trace_truncated``
  record and then stops; the process-wide budget stops the sink with a single
  warning.

The directory named by ``DSV41_LW_V2_MASK_TRACE_DIR`` (default
``/tmp/dsv41-lw-v2-mask-trace``) must already exist.  Missing/unwritable
directories disable tracing with one warning instead of raising: instrumentation
must never turn a serving request into a failure.

Each process writes ``mask-trace-<pid>.jsonl``.  The vllm-ascend seam writes
``mask-trace-<pid>-engine.jsonl`` from its own sink because it cannot import
this package; both use the same record schema, so B01 can concatenate them or
join on ``(request_id, step_id, group_id)``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_ENABLED = "DSV41_LW_V2_MASK_TRACE"
ENV_DIR = "DSV41_LW_V2_MASK_TRACE_DIR"
ENV_MAX_LINES = "DSV41_LW_V2_MASK_TRACE_MAX_LINES"
ENV_MAX_LINES_PER_REQUEST = "DSV41_LW_V2_MASK_TRACE_MAX_LINES_PER_REQUEST"

DEFAULT_DIR = "/tmp/dsv41-lw-v2-mask-trace"
DEFAULT_MAX_LINES = 2_000_000
DEFAULT_MAX_LINES_PER_REQUEST = 20_000

_lock = threading.Lock()
_enabled: bool | None = None
_stream: Any = None
_path: Path | None = None
_warned = False
_seq = 0
_lines = 0
_per_request: dict[str, int] = {}
_truncated: set[str] = set()
_stats: dict[str, int] = {
    "lines": 0,
    "dropped": 0,
    "truncated_requests": 0,
    "write_errors": 0,
}


def reset_state() -> None:
    """Forget the cached switch and close the sink (tests and re-exec)."""
    global _enabled, _stream, _path, _warned, _seq, _lines
    with _lock:
        stream = _stream
        _stream = None
        _path = None
        _enabled = None
        _warned = False
        _seq = 0
        _lines = 0
        _per_request.clear()
        _truncated.clear()
        for name in _stats:
            _stats[name] = 0
    if stream is not None:
        try:
            stream.close()
        except Exception:  # noqa: BLE001 - teardown only
            pass


def enabled() -> bool:
    """Whether the process opted into the timeline (single env read)."""
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get(ENV_ENABLED, "") == "1"
    return _enabled


def now_ns() -> int:
    """Monotonic timestamp used by every trace record."""
    return time.monotonic_ns()


def trace_path() -> Path | None:
    """Path of the open JSONL file, or ``None`` when tracing is off."""
    return _path


def stats() -> dict[str, int]:
    """Bounded-writer counters (``lines``/``dropped``/...)."""
    with _lock:
        return dict(_stats)


def _limit(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _open_locked() -> bool:
    global _stream, _path, _warned, _enabled

    directory = Path(os.environ.get(ENV_DIR, DEFAULT_DIR))
    if not directory.is_dir():
        if not _warned:
            _warned = True
            logger.warning(
                "dsv41 lw v2 mask trace disabled: %s=%s is not an existing "
                "directory",
                ENV_DIR,
                directory,
            )
        _enabled = False
        return False
    path = directory / f"mask-trace-{os.getpid()}.jsonl"
    try:
        stream = path.open("a", encoding="utf-8", buffering=1)
    except OSError as exc:
        if not _warned:
            _warned = True
            logger.warning(
                "dsv41 lw v2 mask trace disabled: cannot open %s: %s",
                path,
                exc,
            )
        _enabled = False
        return False
    _stream = stream
    _path = path
    return True


def emit(event: str, **fields: Any) -> None:
    """Append one record when tracing is enabled, else return immediately."""
    global _seq, _lines, _warned

    if not enabled():
        return
    with _lock:
        if _stream is None and not _open_locked():
            return
        request_id = fields.get("request_id")
        request_key = request_id if isinstance(request_id, str) else None
        if request_key is not None:
            per_request_limit = _limit(
                ENV_MAX_LINES_PER_REQUEST, DEFAULT_MAX_LINES_PER_REQUEST
            )
            count = _per_request.get(request_key, 0)
            if count >= per_request_limit:
                dropped = _truncated
                if request_key not in dropped:
                    dropped.add(request_key)
                    _stats["truncated_requests"] += 1
                    # One record explains why the request's timeline stops.
                    record = {
                        "ts_ns": time.monotonic_ns(),
                        "seq": _seq,
                        "pid": os.getpid(),
                        "event": "trace_truncated",
                        "phase": "trace",
                        "request_id": request_key,
                        "limit": per_request_limit,
                    }
                    _seq += 1
                    _write_locked(record)
                _stats["dropped"] += 1
                return
            _per_request[request_key] = count + 1
        total_limit = _limit(ENV_MAX_LINES, DEFAULT_MAX_LINES)
        if _lines >= total_limit:
            if not _warned:
                _warned = True
                logger.warning(
                    "dsv41 lw v2 mask trace reached %s=%s lines; "
                    "further records are dropped",
                    ENV_MAX_LINES,
                    total_limit,
                )
            _stats["dropped"] += 1
            return
        record = {
            "ts_ns": time.monotonic_ns(),
            "seq": _seq,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        _seq += 1
        _lines += 1
        _stats["lines"] += 1
        _write_locked(record)


def _write_locked(record: dict[str, Any]) -> None:
    global _stream
    stream = _stream
    if stream is None:
        return
    try:
        stream.write(json.dumps(record, separators=(",", ":"), default=str))
        stream.write("\n")
    except Exception:  # noqa: BLE001 - tracing must never break serving
        _stats["write_errors"] += 1
        try:
            stream.close()
        finally:
            _stream = None
