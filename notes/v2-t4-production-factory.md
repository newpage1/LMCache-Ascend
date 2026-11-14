# T4 production adapter factory / transport wiring

Status: implemented on `dsv41_lw_v2` (LMCache-Ascend). The v1
`dsv41_layerwise` branch and its modules are untouched.

## Why this slice exists

T4 (`9332bdd`) shipped the group data plane but left the connector waiting for
a caller to bind `bind_adapter_factory(...)`. Without that bind the connector
was silently inert and only failed closed once a *real* hit reached a hook.
This slice makes the normal vLLM worker startup build the real adapter on the
real stock registration, and makes every missing dependency fail closed at
`register_kv_caches` time instead of degrading to the stock full-tensor path.

## Production path

`DSV41LwV2Connector.register_kv_caches(kv_caches)` now:

1. runs the stock registration (`super().register_kv_caches`) so
   `worker_adapter.kv_caches` / `engine_group_infos` hold the *edited* caches
   and the layer layout is registered with the LMCache server;
2. returns early when an adapter already exists (repeated registration /
   heartbeat is idempotent);
3. resolves the adapter factory: the default is
   `production.DEFAULT_ADAPTER_FACTORY`; an explicit `bind_adapter_factory(None)`
   disables it and raises `DSV41LwV2NotReadyError` (never silent inert);
4. derives the grouped plan from the live registration when the connector has
   no plan bound (`production.build_grouped_plan_from_registration`);
5. calls `factory(plan, connector)`, binds the resulting adapter, and installs
   the schema-2 boundary codec so `connector.current_step` and
   `connector.boundary_codec` are already available when the engine
   `initialize_kv_cache` step runs `install_from_connector`.

`production.create_production_adapter` reads only stock worker-adapter fields
(no host, port or config literals):

| adapter argument | stock source |
| --- | --- |
| `kv_caches` / `cache_names` | `worker_adapter.kv_caches` (post stock edits) |
| `transport` | `worker_adapter.req_client` (the real multiprocess `RequestClient`) |
| `chunk_size` | `worker_adapter.lmcache_tokens_per_chunk` |
| `block_size` | `chunk_size // worker_adapter.blocks_in_chunk` |
| `model_name` | `worker_adapter.model_name` |
| `world_size` | `worker_adapter.world_size` |
| `worker_id` | `worker_adapter.parallel_strategy.kv_worker_id` |
| `instance_id` | `str(worker_adapter.instance_id)` |
| `registration_epoch` | connector registration epoch |
| `plan_hash` | `compute_plan_hash(plan)` (canonical JSON + sha256) |
| `layout_hash` | `compute_layout_hash(plan, worker_adapter.kv_caches)` |

After construction the adapter sends `register_resource_blocks` through
`MPResourceBlockTransport` and the connector's
`set_generations(server_generation=ack.server_generation)` is applied.

## Plan derivation

`build_grouped_plan_from_registration(kv_caches, engine_group_infos)` parses
the live cache names (`<prefix>.layers.<i>.self_attn.{swa_cache,
long_kv_cache,indexer.k_cache,compressor.state_cache}`) and enforces the fixed
DSV4.1 backbone before calling the T2 validator:

- exactly 40 layers 0..39, one `.swa_cache` recompute resource each;
- sources 2/8/14/20 each own `.long_kv_cache` + `.indexer.k_cache`; no other
  layer may register them, and the folded index cache is rejected;
- boundaries 2/8/14 own `.compressor.state_cache`; layer 20 and all other
  layers must not;
- each token resource's `engine_group_id` is the stock engine group whose
  `layer_indices` contain the source layer (missing/ambiguous group fails);
- consumers are the fixed DSV4.1 segments (source 2 -> 2..7, 8 -> 8..13,
  14 -> 14..19, 20 -> 20..39);
- the plan's tensor index order must match the live `cache_names` order; the
  factory rejects a bound plan whose resource id resolves to a different
  registered cache name.

## Transport

`MPResourceBlockTransport` now accepts both stock client shapes:

- the generated typed `ZmqMultiprocessClient` exposes the four
  `@rpc_method`-generated resource-block methods and has no
  `submit_request`; the transport calls those typed methods directly;
- a generic client exposing `submit_request(operation, payloads)` keeps the
  previous path (used by the T4 fake server/tests).

A partial typed interface is a `TypeError`, not a silent fallback.

## Fail-closed matrix

| missing dependency | error |
| --- | --- |
| no factory (`bind_adapter_factory(None)`) | `DSV41LwV2NotReadyError` naming the adapter factory |
| no `worker_adapter` (scheduler role) | `DSV41LwV2NotReadyError` naming `worker_adapter` |
| empty `worker_adapter.kv_caches` | `DSV41LwV2NotReadyError` naming `worker_adapter.kv_caches` |
| no `worker_adapter.req_client` / missing RPCs | `DSV41LwV2NotReadyError` naming `worker_adapter.req_client` |
| invalid chunk / block geometry | `DSV41LwV2NotReadyError` naming `lmcache_tokens_per_chunk` / `blocks_in_chunk` |
| missing model name / world size / worker id | `DSV41LwV2NotReadyError` naming the field |
| registration not the fixed 40-layer topology | `DSV41LwV2NotReadyError` wrapping the production detail |
| server lacks `resource_blocks_v1` capabilities | `DSV41GroupLoadError` from `adapter.register()` at startup |

## Verification (CPU host, no vLLM/hardware)

Environment: `/Users/hexiaoying/workspace/dsv41-layerwise/.venv/bin/python`
(Python 3.11.15, torch 2.15.0, msgspec 0.22.0).

```
cd repos/LMCache-Ascend
PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache \
  .venv/bin/python -m pytest --noconftest -q \
  tests/v1/test_dsv41_lw_v2_production_factory.py
# 17 passed

PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache \
  .venv/bin/python -m pytest --noconftest -q \
  tests/v1/test_dsv41_lw_v2_connector.py \
  tests/v1/test_dsv41_lw_v2_plan.py \
  tests/v1/test_dsv41_lw_v2_adapter.py \
  tests/v1/test_dsv41_lw_v2_production_factory.py
# 140 passed
```

The new test file covers: default factory selection without any manual bind;
adapter built on the worker registration (not the raw argument); typed-client
selection and generic-client fallback; partial-interface rejection; repeated
registration idempotency (one factory call, one adapter, one codec);
`current_step`/`boundary_codec` availability; plan derivation and hash
stability; folded-index and engine-group rejection; shuffled registration
order rejection; and each fail-closed dependency. A git-diff assertion also
verifies no v1 `dsv41_*` source was modified and the new modules do not import
the v1 runtime.

## Still unverified (T6 must prove on hardware)

- real vLLM startup path and `initialize_kv_cache` ordering on Ascend;
- real ZMQ `MessageQueueClient` round-trip with `ResourceBlockModule`
  (T4 server tests used a fake storage manager; no server process was started
  in this slice);
- stock lookup actually returns a hit after the group stores
  (`resource_blocks_v1` + provider cap);
- four group transfers per real prompt, `boundary_restored > 0`,
  `swa_transfer_attempts == 0`;
- TP8/PP and draft-model paths, abort propagation, failed-receiving ordering;
- the connector still requires the stock-key factory
  (`bind_stock_key_factory`) and `dsv41.lw_v2: true` from the launcher; this
  slice only supplies the adapter/transport half of the startup wiring.

The unit tests use a CPU fake request client to prove *which* stock fields and
RPC methods are used; they do not claim the server or hardware path works.
