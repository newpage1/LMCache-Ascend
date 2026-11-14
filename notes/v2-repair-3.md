# v2 repair 3 — production plan engine groups follow the registered tensor order

## Symptom

B01 TP8 (`/data3/v2-logs/v2-diag.jsonl`) stopped during startup, before the
first forward: the production adapter factory could not derive the grouped
plan from the live stock registration.

The registration contains 51 KV caches and 24 `EngineGroupInfo` records:

- tensor indices 0..7 — the four sources' `long_kv_cache` / `indexer.k_cache`
  caches, owned by engine group 0 (split over two physical-transfer infos);
- tensor indices 8..10 — `compressor.state_cache`; they belong to no engine
  group;
- tensor indices 11..50 — the 40 per-layer `swa_cache` caches, in groups
  2..11.

## Root cause

`production.py::_engine_group_for_source` matched the model layer number
(2/8/14/20) against `EngineGroupInfo.layer_indices`. Those indices are
`kv_caches` insertion positions, not model layer numbers — see
`lmcache/v1/multiprocess/group_view.py:26` ("Registered KV tensor indices")
and `lmcache/integration/vllm/kv_cache_groups.py:263`, which maps vLLM layer
names to registered tensor indices before grouping.

Consequently the source never matched group 0. With a diagnostic shim that
rewrote `layer_indices` to layer numbers, the source matched both group 0
(the token caches) and its own SWA group, producing:

```
KV source layer N belongs to several engine groups [0, 2]
```

The t8 CPU tests only used a synthetic single group over `range(40)`, so they
could not catch the live ordering.

## Fix

`_engine_group_for_source(parsed, engine_group_infos, source)` now resolves
the group from the tensor indices already produced by
`_parse_cache_names`:

1. look up every engine group covering the source's `long_kv_cache` index;
2. look up every engine group covering its `indexer.k_cache` index;
3. require both to be non-empty, unambiguous, and identical; otherwise raise
   `DSV41LwV2ProductionUnavailable` before a plan exists.

Model layer numbers are no longer involved in engine-group resolution.
Boundary `compressor.state_cache` resources keep `engine_group_id=None` and
now document why: boundary payloads are full ring pages addressed by request
and logical prefix, not by paged KV block ids.

## Tests

New regression coverage in `tests/v1/test_dsv41_lw_v2_production_factory.py`:

- `test_real_registration_order_maps_sources_to_token_group` — the real B01
  order (8 token, 3 boundary, 40 SWA caches) with all 24 engine-group infos
  (group 0 split over 0..7, groups 2..11 over 11..50, compressors absent):
  every token resource resolves to group 0, no "several groups" error, all
  boundary resources resolve to `None`;
- `test_source_token_caches_must_share_engine_group` — long-KV in group 0 and
  index-K in group 1 fail closed with "split across engine groups";
- `test_plan_derivation_requires_source_engine_group` — a missing token index
  still fails closed (now by tensor index, not layer number).

The previous synthetic single-group double stays as a minimal positive case
and is documented as not representing the live order.

Commands (venv `.venv/bin/python`, no vLLM on this host):

```
PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache \
  python -m pytest --noconftest -q \
  tests/v1/test_dsv41_lw_v2_plan.py \
  tests/v1/test_dsv41_lw_v2_connector.py \
  tests/v1/test_dsv41_lw_v2_adapter.py \
  tests/v1/test_dsv41_lw_v2_production_factory.py
# 151 passed

python -m py_compile lmcache_ascend/integration/vllm/dsv41_lw_v2/production.py
# ok
```

## Not verified

- A real B01 restart through this path: the tests only prove plan derivation
  from the recorded registration; they do not start vLLM or talk to the stock
  resource-block server.
- Real `EngineGroupInfo` producer and `kv_caches` keys on TP8/PP/draft.
- Forward, stock prefix hit, four-group transfer, `boundary_restored > 0` and
  `SWA transfer = 0` (t6 hardware acceptance).
- Any change to t5/t9/t10 semantics; those suites were re-run unchanged, and
  the old `dsv41_layerwise` branch is untouched.
