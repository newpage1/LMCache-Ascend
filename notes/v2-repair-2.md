# DSV4.1 layerwise v2 — repair batch 2 (t5 F4/F6)

Status: implementation + unit tests. Base: `8a5d010` on branch `dsv41_lw_v2`.
Review source: `dsv41-layerwise/notes/v2-review.md` (t5, attempt `552044fe`),
findings F4 (medium) and F6 (low). The legacy `dsv41_layerwise` branch is
untouched, and the repair does not change the T4 adapter/boundary code or the
LMCache server module.

## Scope

Repairs only:

- `lmcache_ascend/integration/vllm/dsv41_lw_v2/plan.py`
- `lmcache_ascend/integration/vllm/dsv41_lw_v2/connector.py` (docstring only)
- `tests/v1/test_dsv41_lw_v2_plan.py`

## F4 — fixed consumer segments are now validated exactly

Problem: the default-topology validator only checked that a group's first
token-range consumer was its head and that `consumer_layers` equalled the union
of the group's resource consumers. A plan whose long KV and index K resources
both shrunk consistently (e.g. source 2 covering layers 2–6) passed, as did a
plan that kept a group union of 2–7 while the long KV descriptor itself only
covered 2–6.

Fix:

- New `DSV41_LW_V2_CONSUMER_SEGMENTS` (plan.py, next to the other fixed
  topology constants) mirrors
  `vllm_ascend.models.deepseek_v41.topology.DSV41_LW_V2_CONSUMER_SEGMENTS`:
  `(2, 2–7)`, `(8, 8–13)`, `(14, 14–19)`, `(20, 20–39)`. The engine module
  remains the source of truth and already validates the same table when it
  builds the topology; the copy lets the stock-manifest validator enforce the
  layout without importing vllm.
- `_validate_default_dsv41_manifest` now builds the expected consumer layer
  names from `base.layer_order` and rejects:
  - a group whose `consumer_layers` is not exactly its fixed segment
    (`Group gN consumers must be the fixed DSV4.1 segment ...`);
  - a long KV descriptor whose own `consumer_layers` is not the whole segment,
    even when the group union happens to match
    (`Long KV resource of group gN must be consumed by the fixed DSV4.1
    segment ...`);
  - an index K descriptor with no consumers or a consumer outside the segment
    (`Index K resource of group gN must be consumed by the index-source subset
    of the fixed DSV4.1 segment ...`);
  - a `DSV41_LW_V2_CONSUMER_SEGMENTS` table whose source ordering drifted away
    from `DSV41_LW_V2_SOURCE_LAYERS`.
- The check runs in the same place as the other fixed-topology invariants, so
  every public construction path (`build_grouped_plan`, direct
  `GroupedTransferPlan(...)`) is covered, before any loader, RPC or forward.

Tests added to `tests/v1/test_dsv41_lw_v2_plan.py`:

- `test_f4_consumer_segment_constant_matches_fixed_table` — the constant is the
  documented table with tuple values (import-safe, no vllm dependency).
- `test_f4_consumer_segments_match_engine_topology_constant` — AST-reads
  `vllm-ascend/.../topology.py` and compares the table when the sibling
  checkout exists; skips otherwise.
- `test_f4_wrong_but_self_consistent_segment_rejected` — both resources of
  source 2 shrink to 2–6 (self-consistent union) and are rejected.
- `test_f4_long_kv_must_cover_the_whole_segment` — union is exactly 2–7, but
  layer 7 is only an index K consumer and the long KV descriptor is rejected.
- `test_f4_normal_plan_keeps_segment_while_index_stays_a_subset` — the normal
  registered shape still builds: long KV equals the full segment, index K is a
  strict subset, group consumers equal the segment.

Residual, documented on purpose: index K membership is checked against the
segment, not against the exact index-source set, because the engine derives
`index_source_layer_ids` from the model config
(`vllm_ascend/models/deepseek_v41/topology.py:363`); the plan cannot hard-code
that set. Descriptor identity against the registered base still pins whatever
the worker actually registered.

## F6 — registration docstring timing

`DSV41LwV2Connector.register_kv_caches` no longer claims the engine reads
`current_step`/`boundary_codec` "from here". It now states that registration
does not open a step: `current_step` is `None` during registration, is filled by
each step's `start_load_kv`, and is cleared by `wait_for_save`; the schema-2
`boundary_codec` is installed at the end of registration. Behaviour unchanged.

## Verification

Run from the `LMCache-Ascend` checkout root, CPU venv
(`/Users/hexiaoying/workspace/dsv41-layerwise/.venv`), no conftest:

```
PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache \
  python -m pytest --noconftest -q tests/v1/test_dsv41_lw_v2_plan.py
# 52 passed (47 pre-existing + 5 F4)

PYTHONPATH=/Users/hexiaoying/workspace/repos/LMCache:/Users/hexiaoying/workspace/repos/LMCache-Ascend \
  python -m pytest --noconftest -q \
    tests/v1/test_dsv41_lw_v2_plan.py tests/v1/test_dsv41_lw_v2_connector.py \
    tests/v1/test_dsv41_lw_v2_adapter.py tests/v1/test_dsv41_lw_v2_production_factory.py
# 145 passed (140 pre-existing + 5 F4), repeated 3× with the same result
```

This is unit evidence only: no vLLM, accelerator, TP or hardware validation in
this batch, and F1/F2/F3/F5 remain T6 release gates.
