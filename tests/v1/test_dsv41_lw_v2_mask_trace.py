# SPDX-License-Identifier: Apache-2.0
"""Tests for the default-off mask timeline sink.

The sink is pure stdlib, so it is loaded by path without touching the
LMCache-Ascend package ``__init__`` (which needs the stock connector).  The
tests pin the three promises the transport relies on: disabled means no file
and one cached boolean; enabled means append-only JSONL with monotonic
timestamps; and both line budgets truncate explicitly instead of growing
without bound.
"""

# Future
from __future__ import annotations

# Standard
import importlib.util
import json
from pathlib import Path
import sys
import types

# Third Party
import pytest


ROOT = Path(__file__).resolve().parents[2]
TRACE_PATH = ROOT / "lmcache_ascend/integration/vllm/dsv41_lw_v2/mask_trace.py"
MODULE_NAME = "dsv41_lw_v2_mask_trace_under_test"


def _load_trace() -> types.ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(MODULE_NAME, TRACE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


trace = _load_trace()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(trace.ENV_ENABLED, raising=False)
    monkeypatch.delenv(trace.ENV_DIR, raising=False)
    monkeypatch.delenv(trace.ENV_MAX_LINES, raising=False)
    monkeypatch.delenv(trace.ENV_MAX_LINES_PER_REQUEST, raising=False)
    trace.reset_state()
    yield
    trace.reset_state()


def _read(directory: Path) -> list[dict]:
    files = list(directory.glob("mask-trace-*.jsonl"))
    assert len(files) == 1, files
    return [
        json.loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_disabled_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()

    assert trace.enabled() is False
    trace.emit("load_start", phase="load", request_id="r1")
    trace.emit("wait_enter", phase="wait", request_id="r1")

    assert list(tmp_path.iterdir()) == []
    assert trace.stats()["lines"] == 0


def test_enabled_writes_monotonic_records_that_can_be_joined(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()

    trace.emit(
        "load_start",
        phase="load",
        step_id=3,
        group_id="g2",
        request_id="r1",
        prefix_end=512,
    )
    trace.emit(
        "wait_enter", phase="wait", step_id=3, group_id="g2", request_id="r1"
    )
    trace.emit(
        "wait_exit",
        phase="wait",
        step_id=3,
        group_id="g2",
        request_id="r1",
        ok=True,
        elapsed_ns=7,
    )
    trace.emit(
        "load_ready",
        phase="load",
        step_id=3,
        group_id="g2",
        request_id="r1",
        status="ok",
    )

    records = _read(tmp_path)
    assert [record["event"] for record in records] == [
        "load_start",
        "wait_enter",
        "wait_exit",
        "load_ready",
    ]
    timestamps = [record["ts_ns"] for record in records]
    assert timestamps == sorted(timestamps)
    assert {record["request_id"] for record in records} == {"r1"}
    assert {record["group_id"] for record in records} == {"g2"}
    assert {record["step_id"] for record in records} == {3}
    assert records[2]["elapsed_ns"] == 7
    assert trace.stats()["lines"] == 4


def test_missing_directory_disables_tracing_with_one_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(missing))
    trace.reset_state()

    trace.emit("load_start", phase="load", request_id="r1")
    assert trace.enabled() is False
    trace.emit("load_ready", phase="load", request_id="r1")

    assert "disabled" in caplog.text
    assert not missing.exists()
    assert trace.stats()["lines"] == 0


def test_per_request_budget_truncates_once(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(trace.ENV_MAX_LINES_PER_REQUEST, "3")
    trace.reset_state()

    for index in range(5):
        trace.emit(
            "load_start", phase="load", request_id="r1", index=index
        )

    records = _read(tmp_path)
    assert len(records) == 4
    assert records[-1]["event"] == "trace_truncated"
    assert records[-1]["request_id"] == "r1"
    assert trace.stats()["truncated_requests"] == 1
    assert trace.stats()["dropped"] == 2


def test_total_budget_stops_the_writer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    monkeypatch.setenv(trace.ENV_MAX_LINES, "2")
    trace.reset_state()

    for index in range(3):
        trace.emit(
            "load_start", phase="load", request_id=f"r{index}", index=index
        )

    records = _read(tmp_path)
    assert len(records) == 2
    assert trace.stats()["dropped"] == 1
