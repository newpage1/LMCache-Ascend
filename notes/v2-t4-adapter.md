# DSV4.1 layerwise v2 — T4 group transport and boundary codec

Status: implementation + unit tests. No vLLM, accelerator or hardware validation
in this slice; the engine-side seam is T3/T7 (`vllm-ascend`), the server-side
visibility contract is in `LMCache`.

## Scope

Contract: `docs/v2-api-contract.md` §4 (adapter/completion), §5 (engine-neutral
resource blocks), §6 (boundary codec), §8.3 (T4 evidence).

Deliverables:

- `lmcache_ascend/integration/vllm/dsv41_lw_v2/adapter.py`
- `lmcache_ascend/integration/vllm/dsv41_lw_v2/boundary.py`
- `lmcache_ascend/integration/vllm/dsv41_lw_v2/connector.py` (runtime wiring)
- `LMCache: lmcache/v1/multiprocess/resource_block_protocol.py`
- `LMCache: lmcache/v1/multiprocess/modules/resource_block.py`
- `LMCache: transport/base.py`, `engine_context.py`, `modules/lookup.py`
- tests: `tests/v1/test_dsv41_lw_v2_adapter.py` (LMCache-Ascend),
  `tests/v1/multiprocess/test_resource_block_visibility.py` (LMCache)

The legacy `dsv41_layerwise` branch and its modules are untouched.

## Group adapter

`DSV41LwV2GroupAdapter(plan, transport=..., instance_id=..., ...)` implements
the `DSV41LwV2Adapter` protocol used by `GroupTransferRouter`:

- `schedule_group_retrieves` queues one load job per group. All four group
  jobs are enqueued by the connector's single `start_load_kv`, so later groups
  prefetch while the current group is being waited on. Each group job submits
  one spec per token resource per chunk plus one boundary spec at
  `request.load_prefix_end`.
- A single worker thread performs device copies, payload encoding, RPC calls
  and result validation; engine threads only stage rows (`index_select/clone`)
  and wait on `GroupEvent`s.
- `wait_group` waits the event, attaches copy-stream events to the compute
  stream, and records `host_wait_fallback` when it had to block on an
  incomplete event. Failures raise `DSV41GroupLoadError` with
  request/group/resource/error code and are reported through
  `take_failed_receiving`.
- `submit_group_store` stages incremental rows from the local stored prefix to
  `store_prefix_end`; stores commit only after the server echoes `ok`. A
  failed store is `dropped/failed`, never committed.
- `drop_request`/`drop_stale_ops` mark terminal `drop`, cancel queued work and
  ignore late callbacks; stale epoch/generation changes are counted and the
  old operations are dropped.
- `take_finished` publishes a request once per direction after all of its
  group operations settled and buffers were released.

Wire DTOs in `adapter.py` mirror the server contract (`ResourceBlockLayout`,
`Registration`, `Spec`, `Result`, `LookupRequest/Result`, `Ack`) and require
the server capabilities `resource_blocks_v1`, `stock_prefix_visibility_v1`,
`boundary_schema2`.

## Boundary codec

`DSV41LwV2BoundaryCodec` implements the T7 hooks
(`capture_layer_boundary` / `restore_layer_boundary`):

- schema 2 only, full ring page, both planes (KV + scores) in one envelope;
  magic + JSON header + plane bytes + SHA-256 over the normalised header and
  payload. Truncation, trailing bytes, bad digest, schema 1, wrong resource,
  wrong ring geometry and wrong written rows are rejected.
- capture is staging-only on the engine thread; the encode/D2H happens in the
  adapter worker. `boundary_end = prefix_end + step_tokens` is the stored
  endpoint, so a later restore at that prefix can find it even when the
  consuming request has a different `request_id` and a different live page.
- restore validates the envelope before touching the live ring, relocates the
  source page to the current request's `ring_page`, writes both planes, and
  raises `DSV41BoundaryRestoreError` (counted as `boundary_restore_failed`)
  instead of silently doing nothing. Missing payloads are explicit
  `BOUNDARY_PAYLOAD_MISSING`.
- `install_dsv41_lw_v2_boundary_codec(connector, adapter=...)` is the
  production install entry point; `DSV41LwV2Connector.set_boundary_codec`
  enforces a single codec instance. `register_kv_caches` now runs the stock
  registration, then an optional adapter factory builds the group adapter over
  the edited caches and installs the codec, so `connector.current_step` /
  `connector.boundary_codec` are populated before `initialize_kv_cache`.

## Server-side visibility

New RPCs: `register_resource_blocks`, `store_resource_block`,
`retrieve_resource_block`, `lookup_resource_blocks`.

- token blocks are written to the stock L1 at resource-namespaced `ObjectKey`s
  derived from the stock chunk hash + plan/layout/resource id; boundary blocks
  are keyed by `<resource>@<logical prefix end>`. `store ok` means
  `finish_write` completed, i.e. the block is in the normal stock storage path.
- the module keeps a completeness index per (instance, plan, layout, model,
  world, salt). `lookup_resource_blocks` returns the longest prefix whose 8
  token resources are complete on every rank and whose 3 boundary resources
  are complete at the endpoint. Intermediate chunks need no boundary.
- `LookupModule` consults `ctx.resource_block_visibility` before building the
  prefetch and truncates the chunk list to that prefix, so stock read locks and
  the reported hit describe the same prefix. Without a provider the stock
  behaviour is unchanged.
- the legacy `store_resource`/`retrieve_resource` staging/commit semantics are
  untouched.

## Tests

LMCache-Ascend (fake server/transport, no vLLM):

```
PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache \
  /Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest \
  --noconftest -q tests/v1/test_dsv41_lw_v2_adapter.py
```

Result: 13 passed.

Existing v2 connector/plan/adapter suites:

```
... pytest --noconftest -q tests/v1/test_dsv41_lw_v2_connector.py \
  tests/v1/test_dsv41_lw_v2_plan.py tests/v1/test_dsv41_lw_v2_adapter.py
```

Result: 123 passed.

LMCache (fake token hasher/storage/context):

```
cd /Users/hexiaoying/workspace/repos/LMCache && PYTHONPATH=$PWD \
  /Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python -m pytest \
  --noconftest -q tests/v1/multiprocess/test_resource_block_visibility.py
```

Result: 10 passed (capacity echo, token+boundary completeness, missing
rank/resource, provider cap + retrieve, shared prefix/different suffix and
request id, salt/model isolation, schema-1 and tamper rejection, stale
generation, duplicate store idempotency, msgspec wire round-trip).

The LMCache test needs the native extension for `tests/v1/multiprocess/
__init__.py`; locally built once with
`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0 NO_GPU_EXT=1 python
setup.py build_ext --inplace`. The legacy `test_layerwise_s1_transport.py`
also imports the compute backend (`numba`) and therefore does not collect in
this bare venv; it was not part of this slice.

## Limits / not yet verified

- No vLLM and no accelerator: real `kv_caches` names/geometry, the adapter
  factory's transport construction and the TP8 path are unverified.
- The engine integration must bind a factory (`bind_adapter_factory`) that
  builds the adapter over the stock edited caches and the real message-queue
  client; this slice provides the seam, not the production factory.
- Full `LookupModule` end-to-end (prefetch + cap) is not exercised here; the
  provider contract and cap call site are unit-tested/compiled only, because
  the stock lookup module needs `numba`/`cryptography` in this environment.
- Hardware evidence (4-group transfer, boundary_restored > 0, SWA transfer 0,
  stock hit on a second shared-prefix request) remains T6.
