# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layerwise v2: group-level transfer planning.

The v2 design keeps the registered resource identity from
:class:`lmcache.v1.multiprocess.layerwise.LayerwiseTransferPlan` and adds a
scheduling-only view: four producer/readiness groups (layers 2, 8, 14 and 20).
SWA resources stay in the base plan as ``recompute`` but never enter a group.

The grouping is a *projection* of the registered manifest, never a rewrite of
it. Every descriptor stored in a :class:`TransferGroup` must be identical to
the base-plan descriptor with the same ``resource_id`` (same tensor indices,
engine group, producer, consumers and storage kind), so the scheduler can
route by group without inventing resource identities that the transport layer
does not know about.

Two validation layers run on every public construction path -- including a
direct ``GroupedTransferPlan(...)`` call:

* structural validation: groups are ordered by execution order, cover exactly
  the transferable resources, never own a recompute resource, agree with the
  base descriptors and derive their consumers from the base execution order;
* fixed-topology validation: when ``base`` is the registered DSV4.1 manifest
  (40 layers, 40 SWA recompute resources), the exact production shape is
  enforced: sources (2, 8, 14, 20), boundaries (2, 8, 14), exactly 8
  token-range resources, exactly 3 boundary-state resources and 11
  transferable resources in total. A folded index cache, a missing resource
  or an unexpected extra resource fails here, before any loader, RPC or
  forward pass starts.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Sequence
import re

# Third Party
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.")
_ATTENTION_SUFFIX = ".attn"

#: Production KV source / readiness layers, in execution order.
DSV41_LW_V2_SOURCE_LAYERS: tuple[int, ...] = (2, 8, 14, 20)
#: Production sources that own a boundary-state resource; layer 20 has none.
DSV41_LW_V2_BOUNDARY_LAYERS: tuple[int, ...] = (2, 8, 14)
#: Production backbone depth.
DSV41_LW_V2_LAYER_COUNT = 40
#: Production transferable-resource totals.
DSV41_LW_V2_GROUP_COUNT = 4
DSV41_LW_V2_TOKEN_RANGE_COUNT = 8
DSV41_LW_V2_BOUNDARY_RESOURCE_COUNT = 3
DSV41_LW_V2_TRANSFER_RESOURCE_COUNT = 11

_LONG_KV_SUFFIX = ".long_kv_cache"
_INDEX_K_SUFFIX = ".indexer.k_cache"
_FOLDED_INDEX_SUFFIX = ".indexer.k_cache_folded"
_BOUNDARY_SUFFIX = ".compressor.state_cache"
_SWA_SUFFIX = ".swa_cache"

_TRANSFERABLE_KINDS = ("token_range", "boundary_state")


@dataclass(frozen=True)
class TransferGroup:
    """One producer/readiness group of transferable DSV4.1 state.

    ``readiness_layer`` is both the group head and the producer layer: a single
    wait at that layer covers every consumer in ``consumer_layers`` (execution
    is sequential). ``transfer_resources`` contains token-range resources and
    at most one boundary-state resource; recompute/SWA resources are excluded.
    """

    group_id: str
    source_layer: int
    readiness_layer: str
    consumer_layers: tuple[str, ...]
    transfer_resources: tuple[CacheTransferResource, ...]
    boundary_resource: CacheTransferResource | None = None

    @property
    def token_range_resources(self) -> tuple[CacheTransferResource, ...]:
        return tuple(
            resource
            for resource in self.transfer_resources
            if resource.storage_kind == "token_range"
        )

    @property
    def has_boundary(self) -> bool:
        return self.boundary_resource is not None


@dataclass(frozen=True)
class GroupedTransferPlan:
    """Immutable group view over a validated base transfer plan.

    Args:
        base: The unmodified registered-cache manifest.
        groups: Transfer groups in producer-layer execution order.

    Raises:
        ValueError: If groups overlap, omit a transfer resource, contain a
            recompute resource, disagree with the base plan, or -- for the
            registered DSV4.1 manifest -- do not match the fixed topology.
    """

    base: LayerwiseTransferPlan
    groups: tuple[TransferGroup, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, LayerwiseTransferPlan):
            raise ValueError(
                "base must be a LayerwiseTransferPlan, got "
                f"{type(self.base).__name__}"
            )
        if not isinstance(self.groups, tuple):
            raise ValueError("groups must be a tuple of TransferGroup")
        _validate_group_structure(self.base, self.groups)
        if _is_default_dsv41_manifest(self.base):
            _validate_default_dsv41_manifest(self.base, self.groups)

    def group_for_layer(self, layer_name: str) -> TransferGroup | None:
        """Return the group whose readiness layer is ``layer_name``.

        Non-head layers deliberately return ``None``: only group heads wait.
        """
        for group in self.groups:
            if group.readiness_layer == layer_name:
                return group
        return None

    def group_by_id(self, group_id: str) -> TransferGroup:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        raise KeyError(group_id)

    def next_group(self, group_id: str) -> TransferGroup | None:
        for index, group in enumerate(self.groups):
            if group.group_id == group_id:
                return self.groups[index + 1] if index + 1 < len(self.groups) else None
        raise KeyError(group_id)

    @property
    def transfer_resource_count(self) -> int:
        return sum(len(group.transfer_resources) for group in self.groups)


def _non_negative_int(value: object, field_name: str) -> int:
    """Return ``value`` if it is a real non-negative int; never coerce.

    ``bool`` is an ``int`` subclass and numeric strings/floats can look like
    layer ids; accepting them would hide a scheduler/configuration bug instead
    of failing at construction time.
    """

    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field_name} must be a non-negative int, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def _layer_index(layer_name: str, field_name: str) -> int:
    match = _LAYER_RE.search(layer_name)
    if match is None:
        raise ValueError(
            f"cannot parse a layer index from {field_name}: {layer_name!r}"
        )
    return int(match.group(1))


def _attention_stem(layer_name: str) -> str:
    if not layer_name.endswith(_ATTENTION_SUFFIX):
        raise ValueError(
            f"layer name must end with {_ATTENTION_SUFFIX!r}: {layer_name!r}"
        )
    return layer_name[: -len(_ATTENTION_SUFFIX)]


def _resource_identity(resource: CacheTransferResource) -> tuple[object, ...]:
    """The full descriptor identity; comparing only ``resource_id`` is unsafe."""

    return (
        resource.resource_id,
        tuple(resource.tensor_indices),
        resource.engine_group_id,
        resource.producer_layer,
        tuple(resource.consumer_layers),
        resource.storage_kind,
    )


def _describe_identity_difference(
    expected: CacheTransferResource, actual: CacheTransferResource
) -> str:
    fields = (
        (
            "tensor_indices",
            tuple(expected.tensor_indices),
            tuple(actual.tensor_indices),
        ),
        ("engine_group_id", expected.engine_group_id, actual.engine_group_id),
        ("producer_layer", expected.producer_layer, actual.producer_layer),
        (
            "consumer_layers",
            tuple(expected.consumer_layers),
            tuple(actual.consumer_layers),
        ),
        ("storage_kind", expected.storage_kind, actual.storage_kind),
    )
    changed = [
        f"{name} {got!r} != {want!r}"
        for name, want, got in fields
        if want != got
    ]
    return "; ".join(changed)


def _base_resource_map(
    base: LayerwiseTransferPlan,
) -> dict[str, CacheTransferResource]:
    return {resource.resource_id: resource for resource in base.resources}


def _is_default_dsv41_manifest(base: LayerwiseTransferPlan) -> bool:
    """Detect a plan for the fixed DSV4.1 topology.

    Any plan whose layers are the contiguous DSV4.1 attention stack (at least
    half the production depth) is validated against the exact manifest, so a
    plan cannot escape the fixed-topology checks by dropping a resource or a
    layer. Small hand-built plans used by unit tests stay on the generic path.
    """

    minimum = DSV41_LW_V2_LAYER_COUNT // 2
    if len(base.layer_order) < minimum:
        return False
    indices: list[int] = []
    for layer_name in base.layer_order:
        if not layer_name.endswith(_ATTENTION_SUFFIX):
            return False
        try:
            indices.append(_layer_index(layer_name, "layer_order"))
        except ValueError:
            return False
    return sorted(indices) == list(range(len(indices)))


def _sorted_by_execution_order(
    base: LayerwiseTransferPlan,
    positions: dict[str, int],
    layers: Sequence[str],
    *,
    context: str,
) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for layer in layers:
        if not isinstance(layer, str) or layer not in positions:
            raise ValueError(f"{context} references an unknown layer: {layer!r}")
        if layer in seen:
            # Several resources of one group legitimately share consumers
            # (e.g. long KV covers the segment, index K a subset of it); the
            # group's consumer set is the union, so duplicates collapse.
            continue
        seen.add(layer)
        unique.append(layer)
    return tuple(sorted(unique, key=positions.__getitem__))


def _validate_group_structure(
    base: LayerwiseTransferPlan, groups: tuple[TransferGroup, ...]
) -> None:
    """Structural invariants shared by every construction path."""

    if not groups:
        raise ValueError("A grouped transfer plan needs at least one group")
    positions = {layer: index for index, layer in enumerate(base.layer_order)}
    base_resources = _base_resource_map(base)
    group_ids: set[str] = set()
    readiness_layers: set[str] = set()
    owned: dict[str, TransferGroup] = {}
    previous_position = -1
    previous_source = -1

    for group in groups:
        if not isinstance(group, TransferGroup):
            raise ValueError(
                f"groups must contain TransferGroup, got {type(group).__name__}"
            )
        if not isinstance(group.group_id, str) or not group.group_id:
            raise ValueError("Group IDs must be nonempty strings")
        if group.group_id in group_ids:
            raise ValueError(f"Duplicate group ID: {group.group_id}")
        group_ids.add(group.group_id)

        source_layer = _non_negative_int(
            group.source_layer, f"group {group.group_id} source_layer"
        )
        if source_layer <= previous_source:
            raise ValueError("Groups must be ordered by source layer")
        previous_source = source_layer

        if not isinstance(group.readiness_layer, str):
            raise ValueError(
                f"Group {group.group_id} readiness_layer must be a string"
            )
        if group.readiness_layer in readiness_layers:
            raise ValueError(
                "Readiness layer appears in multiple groups: "
                f"{group.readiness_layer}"
            )
        readiness_layers.add(group.readiness_layer)
        if group.readiness_layer not in positions:
            raise ValueError(f"Unknown readiness layer: {group.readiness_layer}")
        position = positions[group.readiness_layer]
        if position <= previous_position:
            raise ValueError("Groups must be ordered by readiness layer")
        previous_position = position
        if _layer_index(group.readiness_layer, "readiness_layer") != source_layer:
            raise ValueError(
                f"Group {group.group_id} readiness layer "
                f"{group.readiness_layer!r} is not layer {source_layer}"
            )
        if group.group_id != f"g{source_layer}":
            raise ValueError(
                f"Group {group.group_id!r} does not match its source layer "
                f"{source_layer} (expected 'g{source_layer}')"
            )

        if not isinstance(group.transfer_resources, tuple) or not (
            group.transfer_resources
        ):
            raise ValueError(f"Group {group.group_id} has no transfer resources")
        if not group.token_range_resources:
            raise ValueError(
                f"Group {group.group_id} has no token-range resources"
            )

        consumer_layers = group.consumer_layers
        if not isinstance(consumer_layers, tuple) or not consumer_layers:
            raise ValueError(f"Group {group.group_id} has no consumers")
        for layer in consumer_layers:
            if not isinstance(layer, str):
                raise ValueError(
                    f"Group {group.group_id} has a non-string consumer: {layer!r}"
                )
            if layer not in positions:
                raise ValueError(
                    f"Group {group.group_id} has unknown consumer layers"
                )
        if tuple(sorted(set(consumer_layers), key=positions.__getitem__)) != tuple(
            consumer_layers
        ):
            raise ValueError(
                f"Group {group.group_id} consumers are not in execution order"
            )
        if positions[consumer_layers[0]] < position:
            raise ValueError(
                f"Group {group.group_id} consumer precedes readiness layer"
            )

        boundary_members: list[CacheTransferResource] = []
        for resource in group.transfer_resources:
            if not isinstance(resource, CacheTransferResource):
                raise ValueError(
                    f"Group {group.group_id} contains a non-resource: {resource!r}"
                )
            if resource.storage_kind not in _TRANSFERABLE_KINDS:
                raise ValueError(
                    f"Group {group.group_id} contains a "
                    f"{resource.storage_kind!r} resource: {resource.resource_id}"
                )
            if _SWA_SUFFIX in resource.resource_id:
                raise ValueError(
                    "SWA must be recompute, not transferable: "
                    f"{resource.resource_id}"
                )
            if resource.resource_id in owned:
                raise ValueError(
                    f"Resource belongs to multiple groups: {resource.resource_id}"
                )
            base_resource = base_resources.get(resource.resource_id)
            if base_resource is None:
                raise ValueError(
                    f"Group {group.group_id} owns a resource outside the base "
                    f"plan: {resource.resource_id}"
                )
            if _resource_identity(resource) != _resource_identity(base_resource):
                difference = _describe_identity_difference(
                    base_resource, resource
                )
                raise ValueError(
                    f"Group {group.group_id} descriptor "
                    f"{resource.resource_id} differs from the base plan "
                    f"descriptor: {difference}"
                )
            if resource.storage_kind == "token_range":
                if resource.engine_group_id is None:
                    raise ValueError(
                        f"Token-range resource has no paged engine group: "
                        f"{resource.resource_id}"
                    )
                if resource.tensor_indices and resource.engine_group_id < 0:
                    raise ValueError(
                        f"Token-range resource has a negative engine group: "
                        f"{resource.resource_id}"
                    )
            else:
                boundary_members.append(resource)
                if resource.engine_group_id is not None:
                    raise ValueError(
                        f"Boundary resource must not be in a paged engine "
                        f"group: {resource.resource_id}"
                    )
            if _layer_index(resource.producer_layer, "producer_layer") != source_layer:
                raise ValueError(
                    f"Resource {resource.resource_id} producer "
                    f"{resource.producer_layer!r} is not group "
                    f"{group.group_id} source layer {source_layer}"
                )
            if resource.producer_layer != group.readiness_layer:
                raise ValueError(
                    f"Resource {resource.resource_id} producer differs from "
                    f"group {group.group_id} readiness layer"
                )
            owned[resource.resource_id] = group

        if len(boundary_members) > 1:
            raise ValueError(
                f"Group {group.group_id} owns multiple boundary resources: "
                f"{[resource.resource_id for resource in boundary_members]}"
            )
        if group.boundary_resource is not None and not isinstance(
            group.boundary_resource, CacheTransferResource
        ):
            raise ValueError(
                f"Group {group.group_id} boundary_resource must be a "
                "CacheTransferResource or None"
            )
        if group.boundary_resource is not None:
            if group.boundary_resource.storage_kind != "boundary_state":
                raise ValueError(
                    f"Group {group.group_id} boundary is not boundary_state"
                )
            if not boundary_members:
                raise ValueError(
                    f"Group {group.group_id} boundary resource is not grouped"
                )
            if _resource_identity(group.boundary_resource) != _resource_identity(
                boundary_members[0]
            ):
                raise ValueError(
                    f"Group {group.group_id} boundary_resource does not match "
                    f"its grouped boundary member: "
                    f"{group.boundary_resource.resource_id}"
                )
        elif boundary_members:
            raise ValueError(
                f"Group {group.group_id} has a boundary member but no "
                "boundary_resource"
            )

        expected_consumers = _sorted_by_execution_order(
            base,
            positions,
            [
                layer
                for resource in group.transfer_resources
                for layer in resource.consumer_layers
            ],
            context=f"group {group.group_id} resource consumers",
        )
        if tuple(expected_consumers) != tuple(consumer_layers):
            raise ValueError(
                f"Group {group.group_id} consumers differ from the union of "
                f"its resource consumers: {list(consumer_layers)!r} != "
                f"{list(expected_consumers)!r}"
            )

    transferable = {
        resource.resource_id: resource
        for resource in base.resources
        if resource.storage_kind in _TRANSFERABLE_KINDS
    }
    missing = set(transferable) - set(owned)
    if missing:
        raise ValueError(
            f"Transfer resources are not covered by groups: {sorted(missing)}"
        )
    extra = set(owned) - set(transferable)
    if extra:
        raise ValueError(
            f"Groups own resources outside the base transfer set: {sorted(extra)}"
        )


def _validate_default_dsv41_manifest(
    base: LayerwiseTransferPlan, groups: tuple[TransferGroup, ...]
) -> None:
    """Validate the exact fixed DSV4.1 registration manifest."""

    if len(base.layer_order) != DSV41_LW_V2_LAYER_COUNT:
        raise ValueError(
            f"DSV4.1 base plan must have {DSV41_LW_V2_LAYER_COUNT} layers, "
            f"got {len(base.layer_order)}"
        )
    indices = [
        _layer_index(layer, "layer_order") for layer in base.layer_order
    ]
    if sorted(indices) != list(range(DSV41_LW_V2_LAYER_COUNT)):
        raise ValueError(
            "DSV4.1 layer_order must cover layers 0.."
            f"{DSV41_LW_V2_LAYER_COUNT - 1} exactly once"
        )

    expected_ids: set[str] = set()
    for layer_name in base.layer_order:
        stem = _attention_stem(layer_name)
        expected_ids.add(f"{stem}{_SWA_SUFFIX}")
        index = _layer_index(layer_name, "layer_order")
        if index in DSV41_LW_V2_SOURCE_LAYERS:
            expected_ids.add(f"{stem}{_LONG_KV_SUFFIX}")
            expected_ids.add(f"{stem}{_INDEX_K_SUFFIX}")
        if index in DSV41_LW_V2_BOUNDARY_LAYERS:
            expected_ids.add(f"{stem}{_BOUNDARY_SUFFIX}")

    actual_ids = {resource.resource_id for resource in base.resources}
    folded = sorted(
        resource_id
        for resource_id in actual_ids
        if resource_id.endswith(_FOLDED_INDEX_SUFFIX)
    )
    if folded:
        raise ValueError(
            "Folded index caches are not part of the fixed DSV4.1 v2 "
            f"topology (expected {DSV41_LW_V2_TRANSFER_RESOURCE_COUNT} "
            f"transferable resources, not 15): {folded}"
        )
    unexpected = sorted(actual_ids - expected_ids)
    if unexpected:
        raise ValueError(
            f"Unexpected resources in the DSV4.1 base plan: {unexpected}"
        )
    missing = sorted(expected_ids - actual_ids)
    if missing:
        raise ValueError(f"Missing resources in the DSV4.1 base plan: {missing}")

    base_resources = _base_resource_map(base)
    for layer_name in base.layer_order:
        stem = _attention_stem(layer_name)
        resource = base_resources[f"{stem}{_SWA_SUFFIX}"]
        if resource.storage_kind != "recompute":
            raise ValueError(
                f"SWA resource must be recompute, not "
                f"{resource.storage_kind!r}: {resource.resource_id}"
            )
        if resource.engine_group_id is not None:
            raise ValueError(
                f"SWA resource must not be in a paged engine group: "
                f"{resource.resource_id}"
            )
        if tuple(resource.consumer_layers) != (layer_name,):
            raise ValueError(
                f"SWA resource must be replayed by its own layer only: "
                f"{resource.resource_id}"
            )
        if resource.producer_layer != layer_name:
            raise ValueError(
                f"SWA resource producer must be {layer_name!r}: "
                f"{resource.resource_id}"
            )

    token_range: list[CacheTransferResource] = []
    boundary: list[CacheTransferResource] = []
    recompute: list[CacheTransferResource] = []
    for resource in base.resources:
        if resource.storage_kind == "token_range":
            token_range.append(resource)
        elif resource.storage_kind == "boundary_state":
            boundary.append(resource)
        else:
            recompute.append(resource)
    if len(recompute) != DSV41_LW_V2_LAYER_COUNT:
        raise ValueError(
            f"DSV4.1 base plan must contain {DSV41_LW_V2_LAYER_COUNT} "
            f"recompute resources, got {len(recompute)}"
        )
    if len(token_range) != DSV41_LW_V2_TOKEN_RANGE_COUNT:
        raise ValueError(
            f"DSV4.1 base plan must contain {DSV41_LW_V2_TOKEN_RANGE_COUNT} "
            f"token-range resources, got {len(token_range)}"
        )
    if len(boundary) != DSV41_LW_V2_BOUNDARY_RESOURCE_COUNT:
        raise ValueError(
            f"DSV4.1 base plan must contain "
            f"{DSV41_LW_V2_BOUNDARY_RESOURCE_COUNT} boundary resources, "
            f"got {len(boundary)}"
        )

    by_source: dict[int, list[CacheTransferResource]] = {
        source: [] for source in DSV41_LW_V2_SOURCE_LAYERS
    }
    for resource in token_range + boundary:
        source = _layer_index(resource.producer_layer, "producer_layer")
        if source not in by_source:
            raise ValueError(
                f"Transfer resource is not owned by a source layer: "
                f"{resource.resource_id}"
            )
        by_source[source].append(resource)

    if len(groups) != DSV41_LW_V2_GROUP_COUNT:
        raise ValueError(
            f"DSV4.1 grouped plan must have {DSV41_LW_V2_GROUP_COUNT} groups, "
            f"got {len(groups)}"
        )
    expected_group_ids = tuple(
        f"g{source}" for source in DSV41_LW_V2_SOURCE_LAYERS
    )
    if tuple(group.group_id for group in groups) != expected_group_ids:
        raise ValueError(
            "DSV4.1 groups must be ordered as "
            f"{list(expected_group_ids)!r}, got "
            f"{[group.group_id for group in groups]!r}"
        )

    total = 0
    for group in groups:
        resources = by_source[group.source_layer]
        expected_group_ids_for_source = {
            f"{_attention_stem(group.readiness_layer)}{_LONG_KV_SUFFIX}",
            f"{_attention_stem(group.readiness_layer)}{_INDEX_K_SUFFIX}",
        }
        if group.source_layer in DSV41_LW_V2_BOUNDARY_LAYERS:
            expected_group_ids_for_source.add(
                f"{_attention_stem(group.readiness_layer)}{_BOUNDARY_SUFFIX}"
            )
        actual_group_ids = {
            resource.resource_id for resource in group.transfer_resources
        }
        if actual_group_ids != expected_group_ids_for_source:
            raise ValueError(
                f"Group {group.group_id} must own exactly "
                f"{sorted(expected_group_ids_for_source)}, got "
                f"{sorted(actual_group_ids)}"
            )
        if len(by_source[group.source_layer]) != len(actual_group_ids):
            raise ValueError(
                f"Group {group.group_id} does not cover every resource of "
                f"source layer {group.source_layer}"
            )
        for resource in group.token_range_resources:
            if not resource.resource_id.endswith(
                (_LONG_KV_SUFFIX, _INDEX_K_SUFFIX)
            ):
                raise ValueError(
                    f"Unexpected token-range resource in group "
                    f"{group.group_id}: {resource.resource_id}"
                )
            if tuple(resource.consumer_layers)[0] != group.readiness_layer:
                raise ValueError(
                    f"Token-range resource {resource.resource_id} must be "
                    "consumed from its group head first"
                )
        total += len(group.transfer_resources)

    if total != DSV41_LW_V2_TRANSFER_RESOURCE_COUNT:
        raise ValueError(
            f"DSV4.1 grouped plan must own "
            f"{DSV41_LW_V2_TRANSFER_RESOURCE_COUNT} transfer resources, "
            f"got {total}"
        )


def _source_layer(resource: CacheTransferResource) -> int:
    return _layer_index(resource.producer_layer, "producer_layer")


def _validate_source_ids(
    source_layer_ids: Sequence[object],
    boundary_source_layer_ids: Sequence[object],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(source_layer_ids, (str, bytes)) or not isinstance(
        source_layer_ids, Sequence
    ):
        raise ValueError("source_layer_ids must be a sequence of ints")
    sources = tuple(
        _non_negative_int(layer, "source_layer_ids entry")
        for layer in source_layer_ids
    )
    if not sources:
        raise ValueError("source_layer_ids must be nonempty")
    if len(set(sources)) != len(sources):
        raise ValueError("source_layer_ids must be unique")
    if tuple(sorted(sources)) != sources:
        raise ValueError("source_layer_ids must be strictly increasing")

    if isinstance(boundary_source_layer_ids, (str, bytes)) or not isinstance(
        boundary_source_layer_ids, Sequence
    ):
        raise ValueError("boundary_source_layer_ids must be a sequence of ints")
    boundaries = tuple(
        _non_negative_int(layer, "boundary_source_layer_ids entry")
        for layer in boundary_source_layer_ids
    )
    if not set(boundaries).issubset(set(sources)):
        raise ValueError("boundary layers must be KV source layers")
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("boundary_source_layer_ids must be unique")
    return sources, boundaries


def build_grouped_plan(
    base: LayerwiseTransferPlan,
    *,
    source_layer_ids: Sequence[int],
    boundary_source_layer_ids: Sequence[int] = DSV41_LW_V2_BOUNDARY_LAYERS,
) -> GroupedTransferPlan:
    """Group a validated layerwise plan by producer/readiness layer.

    Args:
        base: Plan built from the live registration. It may contain SWA
            resources as ``recompute``; those are excluded from groups.
        source_layer_ids: Ordered KV source layers (production: 2, 8, 14, 20).
        boundary_source_layer_ids: Layers allowed to own a boundary-state
            resource (production: 2, 8, 14). The final source (20) has none.

    Returns:
        A group view with one group per source layer.

    Raises:
        ValueError: If source IDs are not real ints, are not strictly
            increasing, a transfer resource is not owned by a source, a
            boundary appears on a disallowed source, a group is incomplete,
            or the grouped view disagrees with the base plan. When ``base`` is
            the registered DSV4.1 manifest, the requested sources and
            boundaries must be the production ones and the resulting groups
            must match the fixed 4/11/3 topology.
    """
    if not isinstance(base, LayerwiseTransferPlan):
        raise ValueError(
            f"base must be a LayerwiseTransferPlan, got {type(base).__name__}"
        )
    sources, boundaries = _validate_source_ids(
        source_layer_ids, boundary_source_layer_ids
    )
    if _is_default_dsv41_manifest(base):
        if sources != DSV41_LW_V2_SOURCE_LAYERS:
            raise ValueError(
                "DSV4.1 requires KV source layers "
                f"{list(DSV41_LW_V2_SOURCE_LAYERS)!r}, got {list(sources)!r}"
            )
        if boundaries != DSV41_LW_V2_BOUNDARY_LAYERS:
            raise ValueError(
                "DSV4.1 requires boundary source layers "
                f"{list(DSV41_LW_V2_BOUNDARY_LAYERS)!r}, got "
                f"{list(boundaries)!r}"
            )

    transferable = [
        resource
        for resource in base.resources
        if resource.storage_kind in _TRANSFERABLE_KINDS
    ]
    owned: dict[int, list[CacheTransferResource]] = {
        source: [] for source in sources
    }
    for resource in transferable:
        if _SWA_SUFFIX in resource.resource_id:
            raise ValueError(
                f"SWA must be recompute, not transferable: "
                f"{resource.resource_id}"
            )
        source = _source_layer(resource)
        if source not in owned:
            raise ValueError(
                "Transfer resource is not owned by a source layer: "
                f"{resource.resource_id}"
            )
        owned[source].append(resource)

    positions = {layer: index for index, layer in enumerate(base.layer_order)}
    groups: list[TransferGroup] = []
    for source in sources:
        resources = tuple(owned[source])
        if not resources:
            raise ValueError(f"Source layer {source} owns no transfer resources")
        token_range = tuple(
            resource
            for resource in resources
            if resource.storage_kind == "token_range"
        )
        if not token_range:
            raise ValueError(
                f"Source layer {source} has no token-range resources"
            )
        boundary = tuple(
            resource
            for resource in resources
            if resource.storage_kind == "boundary_state"
        )
        if len(boundary) > 1:
            raise ValueError(
                f"Source layer {source} owns multiple boundary resources"
            )
        if boundary and source not in boundaries:
            raise ValueError(
                f"Source layer {source} must not own a boundary resource"
            )
        if source in boundaries and not boundary:
            raise ValueError(
                f"Source layer {source} is missing its boundary resource"
            )
        readiness = resources[0].producer_layer
        if any(resource.producer_layer != readiness for resource in resources):
            raise ValueError(
                f"Source layer {source} has resources with mixed producer layers"
            )
        consumer_layers = _sorted_by_execution_order(
            base,
            positions,
            [
                layer
                for resource in resources
                for layer in resource.consumer_layers
            ],
            context=f"source layer {source} resource consumers",
        )
        groups.append(
            TransferGroup(
                group_id=f"g{source}",
                source_layer=source,
                readiness_layer=readiness,
                consumer_layers=tuple(consumer_layers),
                transfer_resources=resources,
                boundary_resource=boundary[0] if boundary else None,
            )
        )
    return GroupedTransferPlan(base=base, groups=tuple(groups))
