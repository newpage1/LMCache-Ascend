# SPDX-License-Identifier: Apache-2.0
"""Build a DSV4.1 resource manifest for LMCache-driven MP registration.

This adapter describes the backbone only. It does not alter registration, start
transfers, or enable layerwise execution. Scratch snapshots and replay remain
explicit so a token-range hit cannot silently omit required runtime state.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

# Third Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

if TYPE_CHECKING:
    # Third Party
    from vllm_ascend.models.deepseek_v41.topology import (
        DSV41LayerBinding,
        DeepseekV41Topology,
    )


def build_dsv41_layerwise_plan(
    topology: DeepseekV41Topology,
    registered_cache_names: Sequence[str],
    engine_group_infos: Sequence[EngineGroupInfo],
    *,
    prefix: str = "model",
    replayed_cache_names: Collection[str] = (),
) -> LayerwiseTransferPlan:
    """Bind DSV4.1 cache owners to the original MP tensor registration.

    Args:
        topology: Parsed backbone topology from vLLM-Ascend.
        registered_cache_names: Keys of the unmodified registered KV dictionary,
            in the order used by ``EngineGroupInfo.layer_indices``. A multi-plane
            cache tuple remains one registered entry; it is not flattened here.
        engine_group_infos: Existing LMCache groups describing engine block-ID
            spaces. Scratch/replayed entries must be excluded from these groups.
        prefix: Model root preceding ``layers.<index>.self_attn``.
        replayed_cache_names: Explicitly selected SWA resources reconstructed by
            bounded replay instead of loaded from token-range objects.

    Returns:
        Validated resource manifest. Compression state is ``boundary_state``;
        selected replay SWA is ``recompute``. Neither is a token-range transfer.
        Folded index K is included only if present in the actual registration.

    Raises:
        ValueError: If names or group coverage are inconsistent, a required
            resource is missing, or the registration contains non-backbone KV.
    """
    names = tuple(registered_cache_names)
    if len(set(names)) != len(names):
        raise ValueError("Registered cache names must be unique")
    name_to_index = {name: index for index, name in enumerate(names)}
    replayed = set(replayed_cache_names)
    if not replayed.issubset(name_to_index):
        raise ValueError("Replayed caches must belong to the registration")
    groups_by_tensor: dict[int, int] = {}
    for info in engine_group_infos:
        if info.engine_group_id < 0:
            raise ValueError("Engine group IDs must be nonnegative")
        for index in info.layer_indices:
            if index < 0 or index >= len(names):
                raise ValueError(f"Engine group refers to unknown tensor index {index}")
            if index in groups_by_tensor:
                raise ValueError(
                    f"Registered tensor {index} belongs to multiple groups"
                )
            groups_by_tensor[index] = info.engine_group_id

    root = f"{prefix}." if prefix else ""
    layer_names = tuple(
        f"{root}layers.{role.layer_idx}.self_attn.attn" for role in topology.layers
    )
    resources: list[CacheTransferResource] = []
    for cache in topology.cache_resources(prefix=prefix, include_folded_index=True):
        name = cache.cache_name
        if name not in name_to_index:
            if cache.cache_kind == "folded_index_k":
                continue
            raise ValueError(f"Missing required DSV4.1 cache: {name}")
        index = name_to_index[name]
        group_id = groups_by_tensor.get(index)
        storage_kind: Literal["token_range", "boundary_state", "recompute"]
        if name in replayed and cache.cache_kind != "swa":
            raise ValueError("Only SWA caches can be reconstructed by bounded replay")
        if cache.cache_kind == "compressor_state":
            storage_kind = "boundary_state"
        elif name in replayed:
            storage_kind = "recompute"
        else:
            storage_kind = "token_range"
        if storage_kind == "token_range" and group_id is None:
            raise ValueError(f"Token-range cache is missing its engine group: {name}")
        if storage_kind != "token_range" and group_id is not None:
            raise ValueError(
                f"Scratch/replay cache must be excluded from token-range groups: {name}"
            )
        resources.append(
            CacheTransferResource(
                resource_id=name,
                tensor_indices=(index,),
                engine_group_id=group_id,
                producer_layer=layer_names[cache.producer_layer],
                consumer_layers=tuple(
                    layer_names[layer] for layer in cache.consumer_layers
                ),
                storage_kind=storage_kind,
            )
        )
    covered = {resource.resource_id for resource in resources}
    unexpected = set(names) - covered
    if unexpected:
        raise ValueError(
            f"Unsupported caches outside the DSV4.1 backbone: {sorted(unexpected)}"
        )
    return LayerwiseTransferPlan(layer_order=layer_names, resources=tuple(resources))


def build_dsv41_layer_bindings(
    topology: DeepseekV41Topology,
    static_forward_context: Mapping[str, Any],
    plan: LayerwiseTransferPlan,
) -> tuple[DSV41LayerBinding, ...]:
    """Bind plan execution layers to the live attention objects.

    Identity is read from the real ``DeepseekV41Attention`` modules in the
    static forward context (``role.layer_idx``, ``prefix``,
    ``v41_layer_name``, ``layerwise_execution_layer``). Names are never
    derived from one another by rewriting ``v41_attn``/``attn`` strings:
    the declared attributes are only *validated* against their structural
    equalities, and any mismatch rejects the registration.

    Args:
        topology: Backbone topology from vLLM-Ascend.
        static_forward_context: The engine's static forward context mapping
            dispatch keys to live modules.
        plan: The validated layerwise plan this engine registered.

    Returns:
        Bindings ordered by backbone layer index.

    Raises:
        ValueError: On missing/extra/duplicate backbone bindings, a dispatch
            key not resolving to the module that declared it, attribute
            structure mismatches, or execution names disagreeing with
            ``plan.layer_order``.
    """
    # Local (deferred so plan building stays importable without vLLM-Ascend;
    # the binding step always runs inside an engine process).
    from vllm_ascend.models.deepseek_v41.topology import DSV41LayerBinding

    bindings: dict[int, DSV41LayerBinding] = {}
    for name, module in static_forward_context.items():
        execution_name = getattr(module, "layerwise_execution_layer", None)
        if execution_name is None:
            continue  # Not a DSV4.1 attention module; not part of the backbone.
        prefix = getattr(module, "prefix", None)
        dispatch_name = getattr(module, "v41_layer_name", None)
        role = getattr(module, "role", None)
        layer_idx = getattr(role, "layer_idx", None)
        if not isinstance(prefix, str) or not isinstance(dispatch_name, str):
            raise ValueError(
                f"Module '{name}' declares partial layerwise identity attributes"
            )
        if not isinstance(layer_idx, int):
            raise ValueError(
                f"Module '{name}' has no backbone layer index for layerwise binding"
            )
        if layer_idx in bindings:
            raise ValueError(
                f"Duplicate layerwise binding for backbone layer {layer_idx}: "
                f"{bindings[layer_idx].attention_prefix} and {prefix}"
            )
        # Structural validation of the declared attributes: the dispatch key
        # must point back at the very same object and both names must hang
        # off the module's own prefix.
        if static_forward_context.get(dispatch_name) is not module:
            raise ValueError(
                f"Dispatch key '{dispatch_name}' does not resolve to the "
                f"module that declared it ('{name}')"
            )
        if dispatch_name != f"{prefix}.v41_attn":
            raise ValueError(
                f"Dispatch name '{dispatch_name}' is not the v41_attn key of "
                f"prefix '{prefix}'"
            )
        if not isinstance(execution_name, str) or (
            execution_name != f"{prefix}.attn"
        ):
            raise ValueError(
                f"Execution name '{execution_name}' is not the attn key of "
                f"prefix '{prefix}'"
            )
        bindings[layer_idx] = DSV41LayerBinding(
            layer_idx=layer_idx,
            attention_prefix=prefix,
            dispatch_layer_name=dispatch_name,
            execution_layer_name=execution_name,
        )

    expected = {role.layer_idx for role in topology.layers}
    missing = expected - set(bindings)
    if missing:
        raise ValueError(
            f"Missing layerwise bindings for backbone layers {sorted(missing)}"
        )
    unexpected_layers = set(bindings) - expected
    if unexpected_layers:
        raise ValueError(
            f"Layerwise bindings outside the backbone: {sorted(unexpected_layers)}"
        )
    ordered = tuple(bindings[layer_idx] for layer_idx in sorted(bindings))
    if tuple(b.execution_layer_name for b in ordered) != tuple(plan.layer_order):
        raise ValueError(
            "Execution-layer names do not equal plan.layer_order in layer order"
        )
    roots = {b.attention_prefix.rsplit(".layers.", 1)[0] for b in ordered}
    if len(roots) != 1:
        raise ValueError(f"Bindings span multiple model roots: {sorted(roots)}")
    return ordered
