# DSV4.1 layerwise v2 — repair batch 1 (t5 F1/F2/F3/F5)

Status: implementation + unit tests. Base: `b334e5f` on branch `dsv41_lw_v2`
(the t10 plan-consumer/docstring commit). Review source:
`dsv41-layerwise/notes/v2-review.md` (t5). Findings F4 and F6 are handled by
the separate repair batch 2 (`notes/v2-repair-2.md`), so they are out of scope
here. The legacy `dsv41_layerwise` branch and its modules are untouched.

Commits (same branch, in review order):

- `a62a780` — F1, boundary absolute ring rows
- `eeedf42` — F2/F3, connector terminal reporting + `take_finished` order
- `ed54a1e` — F5 worker side, `PLAN_MISMATCH` propagation test
- LMCache `d14dfb5` (signed-off) — F5 server side, `ERROR_PLAN_MISMATCH`

## F1 (blocker) — `written_rows` are absolute, not page-relative

Problem: the engine DTO (`vllm_ascend/attention/dsv41_lw_v2_boundary.py`,
`BoundaryRequestView.__post_init__` and `_written_rows`) emits absolute ring
rows in `[ring_page * ring_length, (ring_page + 1) * ring_length)`, while the
v2 codec (`boundary.py::_validate_view`) demanded `0 <= row < ring_length`.
Every real capture/restore on a page other than zero therefore failed with
`BOUNDARY_GEOMETRY_MISMATCH`.

Fix (`boundary.py`, LMCache-Ascend):

- `_validate_view` now validates rows against the view's own live-page window
  `[ring_page * ring_length, (ring_page + 1) * ring_length)` and rejects a
  negative page. The engine `BoundaryRequestView` remains the wire source of
  truth; no vllm-ascend file is changed.
- `_validate_restore` no longer compares the consuming view's rows with the
  captured rows. A restore happens in a different request's live page (that is
  the whole point of a shared stock hit), so the captured rows are validated
  against the capture page recorded in the payload (`_validate_header_rows`:
  int rows, `min(step_tokens, ring_length)` of them, unique, inside the
  capture page window). The consuming view is still validated by
  `_validate_view` (count, uniqueness, its own page).
- Error messages now name the live/capture page window.

Cross-repo evidence: `test_real_engine_view_absolute_rows_cross_page_roundtrip`
in `tests/v1/test_dsv41_lw_v2_adapter.py` loads the real vllm-ascend
`BoundaryRequestView`, `BoundaryTensorView` and `_written_rows` (module loaded
from `../vllm-ascend` with a package skeleton; no vLLM runtime), asserts the
kernel rows are `(15,)` on page 3, then runs the real v2 codec
capture -> store -> retrieve -> restore into ring page 7 (rows `(28,)`) and
checks both planes are relocated, page 6 is untouched, and
`boundary_captured == boundary_restored == 1`, `boundary_restore_failed == 0`.
The test skips only when the sibling vllm-ascend checkout is absent.

## F2 (high) — v2 terminal sets never reached the stock scheduler

Problem: `DSV41LwV2Connector` did not override `get_finished` /
`get_transfer_results`, so the adapter's `take_finished()` and
`take_failed_receiving()` had no consumer. A completed v2 transfer was never
reported, and a failed group load could not become `failed_recving`, so vLLM
would not take the `BYPASS_LMCACHE` recompute path.

Fix (`connector.py`):

- `get_finished(finished_req_ids)` calls the stock implementation, then merges
  the adapter's `(finished_sending, finished_receiving)`. A stock `None`
  ("nothing new") stays `None` when the adapter adds nothing.
- `get_transfer_results(finished_req_ids)` calls the stock implementation
  (whose own `get_finished` dispatch drains the adapter through the override)
  and merges `take_failed_receiving()` into `results.failed_recving`; failed
  requests are never added to `finished_recving`.
- Helper `_merge_id_sets(base, extra)` preserves the stock `None` semantics.

Tests (connector suite):

- `test_get_finished_merges_stock_and_adapter_terminal_sets` — both stock and
  adapter send/receive sets are unioned, and the adapter half is drained.
- `test_get_transfer_results_reports_failed_recving_for_bypass` — a failed
  receive appears in `failed_recving` (together with the stock failure) and
  not in `finished_recving`.
- The adapter suite already asserts a real missing resource ends in
  `take_failed_receiving() == {"r1"}` (that is the producer side of the same
  path).

## F3 (medium) — `take_finished` order did not match the contract

Problem: `adapter.take_finished()` returned `(finished_receiving,
finished_sending)`, while the contract and the new connector seam expect
`(finished_sending, finished_receiving)`.

Fix: return `finished_sending, finished_receiving`; the four adapter test
assertions were flipped accordingly (ready-load test, drop/stale negative,
store completion, store failure).

## F5 (low) — plan mismatch was reported as invalid spec

Problem: `modules/resource_block.py::_validate_spec` returned
`ERROR_INVALID_SPEC` when `spec.plan_hash` did not match the registered plan,
although `ERROR_PLAN_MISMATCH` exists in the wire contract.

Fix (LMCache): return `ERROR_PLAN_MISMATCH` for store and retrieve, keep
`ERROR_LAYOUT_MISMATCH` for layout. Adapter worker propagates the server code
unchanged, so this is an additive error-precision change.

Tests:

- `test_plan_hash_mismatch_is_plan_mismatch` (LMCache) — store and retrieve
  both return `STATUS_FAILED` + `ERROR_PLAN_MISMATCH`, retrieve payload is
  empty.
- `test_plan_mismatch_propagates_to_failed_recving` (LMCache-Ascend) — a
  server that rejects with `PLAN_MISMATCH` makes `wait_group` raise
  `DSV41GroupLoadError(error_code=PLAN_MISMATCH)` and reports the request in
  `take_failed_receiving()`.

## Commands and results

Environment: `/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python`
(Python 3.11.15, torch 2.15.0, msgspec 0.22.0; no vLLM, no accelerator), all
pytest runs with `--noconftest`, t5 rerun set.

```
# LMCache-Ascend v2 suite (plan + connector + adapter + production factory)
cd /Users/hexiaoying/workspace/repos/LMCache-Ascend
PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache \
  /Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest --noconftest -q \
  tests/v1/test_dsv41_lw_v2_plan.py \
  tests/v1/test_dsv41_lw_v2_connector.py \
  tests/v1/test_dsv41_lw_v2_adapter.py \
  tests/v1/test_dsv41_lw_v2_production_factory.py
# 149 passed in 1.28s

# LMCache resource blocks
cd /Users/hexiaoying/workspace/repos/LMCache
PYTHONPATH=$PWD /Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest --noconftest -q \
  tests/v1/multiprocess/test_resource_block_visibility.py
# 11 passed in 0.73s

# vllm-ascend v2 engine tests (unchanged by this repair)
cd /Users/hexiaoying/workspace/repos/vllm-ascend
/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest --noconftest -q \
  tests/ut/models/test_deepseek_v41_transfer_resources.py \
  tests/ut/attention/test_dsv41_lw_v2_hooks.py \
  tests/ut/attention/test_dsv41_lw_v2_boundary.py
# 81 passed in 0.69s

# legacy layerwise regression on the v2 branch
/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest --noconftest -q \
  tests/ut/attention/test_dsv41_layerwise_store_hook.py
# 5 passed in 0.05s
```

`python -m py_compile` passes for `boundary.py`, `connector.py`, `adapter.py`
and the LMCache `resource_block.py`. `ruff` is not installed in this venv.

## Not verified (same limits as t4)

- No hardware/T6 run: stock hit, four-group transfer, `boundary_restored > 0`
  and `SWA transfer == 0` still need the B01 TP8 run.
- No real vLLM startup: the stock connector's `get_finished` /
  `get_transfer_results` are exercised through the CPU stub, and vLLM's
  `KVConnectorTransferResults` mutability is assumed (it is a plain dataclass).
- No TP-consistent abort, PP/draft path, or real ZMQ round-trip with the
  resource-block module in this slice.
