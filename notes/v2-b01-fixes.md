# DSV4.1 layerwise v2 — B01 round-2 fixes (t14)

Scope: close the five findings from
`docs/../dsv41-layerwise/notes/v2-b01-validation-round2.md` (t13) that kept the
B01 run at zero stock hits and zero boundary captures.  No legacy
`dsv41_layerwise` module or branch was touched, and no fake boundary/hit path
was added.

## F1 (blocker) — one wire DTO definition

`adapter.py` used to declare its own `msgspec.Struct` copies of
`ResourceBlockRegistration/Spec/Result/Ack/...`.  A typed client decodes a
server reply into `lmcache.v1.multiprocess.resource_block_protocol.<X>`, so
`isinstance(reply, adapter.<X>)` was always false and every real request ended
as `INVALID_SPEC`.

Fix: the adapter now imports the server dataclasses
(`from lmcache.v1.multiprocess import resource_block_protocol as _wire`) and
uses them for construction, typing, and `isinstance`.  Constants/error codes
come from the same module (`QUEUE_FULL`/`INTERNAL_ERROR` stay adapter-local).

Test: `test_wire_dtos_are_shared_server_types_and_round_trip` asserts type
identity and `msgspec.msgpack` round-trip equality for a spec whose stock key
is a real `IPCCacheServerKey`.

## F2 (high) — resolve the model prefix from the live model

The engine seam hard-coded `DSV41_LW_V2_MODEL_PREFIX = "model"`, while B01
runs with `language_model.model`.  `install_from_connector` therefore built a
binding whose resource ids did not exist on the model and every boundary
resource check failed closed.

Fix (`vllm_ascend/attention/dsv41_lw_v2_boundary.py`): drop the constant and
add `_resolve_model_prefix(connector, model)`:

1. use `connector.grouped_plan.base.layer_order[0]` up to `".layers."`, else
2. scan `model.modules()` for `v41_impl.compressor_state_prefix`, else
3. raise `DSV41BoundaryBindingError`.

Tests: install with `prefix="language_model.model"` (plan-derived) asserting
binding prefix, group readiness layers, and the actual capture view/tensor
resource ids; plus plan-less model-derived prefix.

## F3 (high) — production cold STORE gets a stock key

The production factory never bound a stock-key factory, so a cold STORE threw
`DSV41LwV2NotReadyError: no stock key factory is bound` from
`_build_step_view` -> `_stock_key`.

Fix: `create_production_adapter` binds `connector.bind_stock_key_factory(
worker_adapter._create_key)`.  That is the worker's own
`IPCCacheServerKey` constructor, so the store key is byte-for-byte the key the
stock lookup uses (`num_kv_readers`, model, world, worker, token bytes,
start/end, request id, salt).

Test: `test_production_factory_serves_cold_store_with_the_stock_key` builds a
STORE-only step through `start_load_kv` and asserts the resulting stock key.

## F4 (medium) — do not release unfinished requests

`get_finished` merged every terminal adapter id, but vLLM frees blocks for
`finished_sending` and asserts the request is finished.  The stock connector
only reports ids from the scheduler's finished set.

Fix: `DSV41LwV2GroupAdapter.take_finished(finished_req_ids=...)` intersects
with the scheduler set and keeps the rest queued (`None` keeps the legacy
drain-everything behaviour); `DSV41LwV2Connector.get_finished` passes its
argument through.

Tests: merge test with the full finished set, plus
`test_get_finished_defers_terminal_ids_for_unfinished_requests` (id reported
only on the second call that contains it).

## F5 (blocker) — keep the live step until post-forward capture

Trace: `wait_for_save` submitted all groups, drained, and called
`finish_step()` *before* the forward; the post-forward
`capture_group_boundary` then saw `has_ops=False`.  `note_step` also began
with `finish_step()`, so a deferred `start_load_kv` could erase ops.

Fix (connector):

- `GroupTransferRouter.drain_pending_stores()` only drains.
- `GroupTransferRouter.wait_for_save()` only drains previous-forward stores;
  it no longer submits or finishes the live step.
- `GroupTransferRouter.flush_step()` submits the groups that still have ops,
  drains, then `finish_step()`.
- `connector.start_load_kv` flushes the *previous* step before creating the
  new one.
- `connector.wait_for_save` drains only; `_lw_v2_step_noted` and
  `_current_step` survive so later heads can capture.
- `connector.get_finished` flushes the current step, then clears the step
  fields and merges terminal ids.
- An empty metadata step clears the live step instead of leaving a stale one.
- A generation change now drops pending ops immediately via
  `GroupTransferRouter.set_generations` (previously the next `flush_step`
  cleared them first and no request reached `adapter.drop_request`).

Tests:
`test_capture_precedes_submit_and_wait_for_save_keeps_ops` follows the B01
sequence (capture g2 -> wait_for_save -> capture g8/g14 -> get_finished hands
off g20), and
`test_wait_for_save_drains_previous_stores_and_keeps_the_live_step` asserts
the live step and `_lw_v2_step_noted` survive the hook.  The router/connector
generation test still sees `("r1", "generation_changed")` in
`adapter.dropped`.

## Verification

Commands and results (venv `/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python`):

```
cd repos/LMCache-Ascend
PYTHONPATH=../LMCache python -m pytest --noconftest -q \
  tests/v1/test_dsv41_lw_v2_adapter.py \
  tests/v1/test_dsv41_lw_v2_connector.py \
  tests/v1/test_dsv41_lw_v2_plan.py \
  tests/v1/test_dsv41_lw_v2_production_factory.py
# 155 passed
python -m py_compile lmcache_ascend/integration/vllm/dsv41_lw_v2/*.py
# ok

cd repos/vllm-ascend
python -m pytest --noconftest -q \
  tests/ut/attention/test_dsv41_lw_v2_boundary.py \
  tests/ut/attention/test_dsv41_lw_v2_hooks.py
# 63 passed
python -m py_compile vllm_ascend/attention/dsv41_lw_v2_boundary.py
# ok
```

`git diff --check` clean in both repositories.

## Still not verified

- No B01 rerun in this change set: the real end-to-end evidence (stock hit,
  4 group transfers, `boundary_restored > 0`, `swa_transfer_attempts == 0`)
  still has to come from the next hardware round.
- The v2 engine seam must actually be bound in that run
  (`dsv41_lw_v2.lw_v2_engine()` non-None); the unit tests drive
  `save_kv_layer`/`wait_for_save`/`get_finished` in the vLLM order, not the
  model runner.
- `wait_for_save` is now a drain-only hook; the real scheduler must call
  `get_finished` (or the next `start_load_kv`) every step for the store
  batches and routing step to close.
