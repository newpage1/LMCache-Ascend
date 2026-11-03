# SPDX-License-Identifier: Apache-2.0
"""Build a DSV4.1 resource manifest for LMCache-driven MP registration.

This adapter describes the backbone only. It does not alter registration, start
transfers, or enable layerwise execution. Scratch snapshots and replay remain
explicit so a token-range hit cannot silently omit required runtime state.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Literal

# Third Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

if TYPE_CHECKING:
    # Third Party
    from vllm_ascend.models.deepseek_v41.topology import DeepseekV41Topology


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
