# DSV4.1 Layerwise Registration Prototype

The `dsv41_layerwise` branches introduce a CPU-testable registration manifest for
LMCache `dev`, LMCache-Ascend `main` and vLLM-Ascend `main`. This is preparation
for `lmcache_driven` layerwise transfers, not a new runnable serving mode.

## Build an Inspectable Plan

After building the DSV4.1 topology and obtaining the original registered cache
dictionary plus its `EngineGroupInfo` metadata:

```python
from lmcache_ascend.integration.vllm.layerwise import build_dsv41_layerwise_plan

plan = build_dsv41_layerwise_plan(
    topology,
    tuple(kv_caches),
    engine_group_infos,
    prefix="model",
    replayed_cache_names=replayed_swa_names,
)
dependencies = plan.required_by("model.layers.26.self_attn.attn")
```

The model prefix must match the actual registration, including multimodal
wrappers. The adapter is explicit and is not installed as an import-time patch.
The current prototype covers the backbone only and rejects unexpected draft or
other cache entries rather than silently dropping them.

## Ownership and State

Long KV is owned by its KV source and remains live through the final long-KV
consumer. Index K/scale is owned by that source too, but is read only by layers
that actually compute new top-k. Top-k and candidate results are query-dependent
computed state and do not become persistent resources.

SWA is layer-local. C2 compressor state is represented as `boundary_state`, not
ordinary token-range KV. An SWA resource may be marked `recompute` only through
the explicit `replayed_cache_names` argument; the corresponding cache must have
been excluded from token-range groups. The adapter does not assert that replay
or compressor-state restoration is already correct.

Folded index K is included only if present in the actual registration. A
multi-plane index cache remains a single registered entry: its value/scale tuple
must not be flattened or renumbered. Equal shapes or shared backing pointers
are not evidence that two semantic owners contain the same data.

## Validation and Remaining Work

CPU tests cover ownership, registered indices, block-ID address spaces,
MessagePack roundtrips, conditional folded planes and rejection of incomplete
or incorrectly grouped registrations. They do not validate NPU copy kernels.

No configuration switch currently enables this prototype. Resource object
partitioning, selected-resource RPC, device events, forward-internal admission,
wait/save hooks, failure recovery and hardware output comparisons remain to be
implemented and tested before enabling layerwise serving.
