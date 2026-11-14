# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSV4.1 layerwise v2 grouped transfer plan.

The fixture mirrors the real registered topology: 40 per-layer SWA resources
(``recompute``), four KV source groups (2, 8, 14, 20) each exposing one
``long_kv_cache`` and one ``indexer.k_cache`` resource, and a
``compressor.state_cache`` boundary on 2, 8 and 14 only. Index-K consumers are
the *index-source subset* of their segment, never a rewrite to the whole
segment.
"""

# Standard
import ast
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

# Third Party
import pytest
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

ROOT = Path(__file__).resolve().parents[2]
DSV41_LW_V2_DIR = ROOT / "lmcache_ascend/integration/vllm/dsv41_lw_v2"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_plan_module():
    if "dsv41_lw_v2_plan_under_test" in sys.modules:
        return sys.modules["dsv41_lw_v2_plan_under_test"]
    return _load_module("dsv41_lw_v2_plan_under_test", DSV41_LW_V2_DIR / "plan.py")


plan_mod = _import_plan_module()
build_grouped_plan = plan_mod.build_grouped_plan
GroupedTransferPlan = plan_mod.GroupedTransferPlan


LAYERS = tuple(
    f"model.layers.{index}.self_attn.attn" for index in range(40)
)
SOURCES = (2, 8, 14, 20)
BOUNDARIES = (2, 8, 14)
SEGMENT_END = {2: 7, 8: 13, 14: 19, 20: 39}


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _stem(index: int) -> str:
    return f"model.layers.{index}.self_attn"


def _index_readers(source: int) -> tuple[int, ...]:
    """Index-source layers of one segment: a strict subset of the segment."""
    return tuple(
        layer
        for layer in range(source, SEGMENT_END[source] + 1)
        if (layer - source) % 2 == 0
    )


def _resource(
    resource_id: str,
    *,
    storage_kind: str,
    producer_layer: str,
    consumer_layers: Sequence[str],
    tensor_index: int,
    engine_group_id: int | None,
) -> CacheTransferResource:
    return CacheTransferResource(
        resource_id=resource_id,
        tensor_indices=(tensor_index,),
        engine_group_id=engine_group_id,
        producer_layer=producer_layer,
        consumer_layers=tuple(consumer_layers),
        storage_kind=storage_kind,  # type: ignore[arg-type]
    )


def _clone_resource(
    resource: CacheTransferResource, **changes
) -> CacheTransferResource:
    fields = {
        "resource_id": resource.resource_id,
        "tensor_indices": tuple(resource.tensor_indices),
        "engine_group_id": resource.engine_group_id,
        "producer_layer": resource.producer_layer,
        "consumer_layers": tuple(resource.consumer_layers),
        "storage_kind": resource.storage_kind,
    }
    fields.update(changes)
    return CacheTransferResource(**fields)


def _swa_resources() -> list[CacheTransferResource]:
    return [
        _resource(
            f"{_stem(index)}.swa_cache",
            storage_kind="recompute",
            producer_layer=_layer(index),
            consumer_layers=(_layer(index),),
            tensor_index=index,
            engine_group_id=None,
        )
        for index in range(40)
    ]


def _transfer_resources(
    *,
    index_first: bool = False,
    index_readers=None,
) -> list[CacheTransferResource]:
    readers = index_readers or _index_readers
    resources: list[CacheTransferResource] = []
    for source in SOURCES:
        segment = tuple(
            _layer(index)
            for index in range(source, SEGMENT_END[source] + 1)
        )
        long_kv = _resource(
            f"{_stem(source)}.long_kv_cache",
            storage_kind="token_range",
            producer_layer=_layer(source),
            consumer_layers=segment,
            tensor_index=40 + source,
            engine_group_id=0,
        )
        index_k = _resource(
            f"{_stem(source)}.indexer.k_cache",
            storage_kind="token_range",
            producer_layer=_layer(source),
            consumer_layers=tuple(
                _layer(index) for index in readers(source)
            ),
            tensor_index=80 + source,
            engine_group_id=0,
        )
        if index_first:
            resources.extend((index_k, long_kv))
        else:
            resources.extend((long_kv, index_k))
        if source in BOUNDARIES:
            resources.append(
                _resource(
                    f"{_stem(source)}.compressor.state_cache",
                    storage_kind="boundary_state",
                    producer_layer=_layer(source),
                    consumer_layers=(_layer(source),),
                    tensor_index=120 + source,
                    engine_group_id=None,
                )
            )
    return resources


def _base_plan(
    *,
    index_first: bool = False,
    index_readers=None,
    resources: Sequence[CacheTransferResource] | None = None,
) -> LayerwiseTransferPlan:
    if resources is None:
        resources = (
            _swa_resources()
            + _transfer_resources(
                index_first=index_first, index_readers=index_readers
            )
        )
    return LayerwiseTransferPlan(layer_order=LAYERS, resources=tuple(resources))


def _default_plan(**kwargs) -> "plan_mod.GroupedTransferPlan":
    return build_grouped_plan(
        _base_plan(**kwargs),
        source_layer_ids=SOURCES,
        boundary_source_layer_ids=BOUNDARIES,
    )


def _group(plan, group_id: str):
    return plan.group_by_id(group_id)


def _replace_resource(
    base: LayerwiseTransferPlan,
    resource_id: str,
    **changes,
) -> LayerwiseTransferPlan:
    resources = tuple(
        _clone_resource(resource, **changes)
        if resource.resource_id == resource_id
        else resource
        for resource in base.resources
    )
    return LayerwiseTransferPlan(base.layer_order, resources)


def _replace_group_resource(plan, group_id: str, index: int, **changes):
    """Return a plan-like tuple with one group resource descriptor mutated."""
    group = plan.group_by_id(group_id)
    resources = list(group.transfer_resources)
    resources[index] = _clone_resource(resources[index], **changes)
    groups = tuple(
        replace(candidate, transfer_resources=tuple(resources))
        if candidate.group_id == group_id
        else candidate
        for candidate in plan.groups
    )
    return groups


# ---------------------------------------------------------------------------
# Normal fixed topology
# ---------------------------------------------------------------------------


def test_default_manifest_builds_exact_four_groups():
    plan = _default_plan()

    assert tuple(group.group_id for group in plan.groups) == (
        "g2",
        "g8",
        "g14",
        "g20",
    )
    assert tuple(group.source_layer for group in plan.groups) == SOURCES
    assert tuple(group.readiness_layer for group in plan.groups) == tuple(
        _layer(source) for source in SOURCES
    )
    assert plan.transfer_resource_count == 11
    assert len(plan.base.resources) == 51  # 40 SWA + 8 token + 3 boundary

    token_range = [
        resource
        for group in plan.groups
        for resource in group.token_range_resources
    ]
    boundary = [
        group.boundary_resource
        for group in plan.groups
        if group.boundary_resource is not None
    ]
    assert len(token_range) == 8
    assert len(boundary) == 3
    assert all(
        ".swa_cache" not in resource.resource_id
        for group in plan.groups
        for resource in group.transfer_resources
    )


def test_each_group_owns_one_long_kv_one_index_k_and_segment_consumers():
    plan = _default_plan()
    for group in plan.groups:
        suffix = sorted(
            resource.resource_id.rsplit(".", 1)[-1] for resource in
            group.transfer_resources
        )
        assert suffix.count("long_kv_cache") == 1
        assert suffix.count("k_cache") == 1
        expected = tuple(
            _layer(index)
            for index in range(
                group.source_layer, SEGMENT_END[group.source_layer] + 1
            )
        )
        assert group.consumer_layers == expected


def test_index_consumers_are_the_index_subset_not_the_whole_segment():
    plan = _default_plan()
    index_k = next(
        resource
        for resource in _group(plan, "g2").transfer_resources
        if resource.resource_id.endswith(".indexer.k_cache")
    )
    expected = tuple(_layer(index) for index in _index_readers(2))
    whole_segment = tuple(
        _layer(index) for index in range(2, SEGMENT_END[2] + 1)
    )
    assert index_k.consumer_layers == expected
    assert index_k.consumer_layers != whole_segment
    # The group's consumer union is still the whole segment (long KV) sorted by
    # the base execution order, not by resource order.
    assert _group(plan, "g2").consumer_layers == whole_segment


def test_g20_has_no_boundary_and_other_groups_exactly_one():
    plan = _default_plan()
    assert _group(plan, "g20").boundary_resource is None
    assert not _group(plan, "g20").has_boundary
    for group_id in ("g2", "g8", "g14"):
        group = _group(plan, group_id)
        assert group.boundary_resource is not None
        assert group.boundary_resource.resource_id.endswith(
            ".compressor.state_cache"
        )
        assert group.boundary_resource.storage_kind == "boundary_state"


def test_group_for_layer_only_returns_heads():
    plan = _default_plan()
    for source in SOURCES:
        assert plan.group_for_layer(_layer(source)) is not None
        assert plan.group_for_layer(_layer(source)).group_id == f"g{source}"
    for layer in (0, 1, 3, 7, 9, 13, 15, 19, 21, 39):
        assert plan.group_for_layer(_layer(layer)) is None
    assert plan.group_for_layer("model.layers.40.self_attn.attn") is None


def test_group_lookup_and_ordering_helpers():
    plan = _default_plan()
    with pytest.raises(KeyError):
        plan.group_by_id("g99")
    assert _group(plan, "g2") is plan.groups[0]
    assert plan.next_group("g2").group_id == "g8"
    assert plan.next_group("g8").group_id == "g14"
    assert plan.next_group("g14").group_id == "g20"
    assert plan.next_group("g20") is None
    with pytest.raises(KeyError):
        plan.next_group("g99")


def test_swa_stays_recompute_with_zero_transfers():
    base = _base_plan()
    recompute = [
        resource
        for resource in base.resources
        if resource.storage_kind == "recompute"
    ]
    assert len(recompute) == 40
    assert all(
        resource.resource_id.endswith(".swa_cache") for resource in recompute
    )
    plan = build_grouped_plan(
        base, source_layer_ids=SOURCES, boundary_source_layer_ids=BOUNDARIES
    )
    assert plan.transfer_resource_count == 11


def test_all_swa_must_be_registered_even_if_recompute():
    base = _base_plan()
    resources = tuple(
        resource
        for resource in base.resources
        if resource.resource_id != f"{_stem(7)}.swa_cache"
    )
    with pytest.raises(ValueError, match="Missing resources"):
        build_grouped_plan(
            LayerwiseTransferPlan(LAYERS, resources),
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


# ---------------------------------------------------------------------------
# P1 - descriptor identity must match the base plan, not just the ID
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"tensor_indices": (999,)},
        {"engine_group_id": 99},
        {"producer_layer": _layer(3)},
        {"consumer_layers": (_layer(2),)},
        {"storage_kind": "boundary_state"},
    ],
)
def test_p1_descriptor_mismatch_rejected(changes):
    plan = _default_plan()
    groups = _replace_group_resource(plan, "g2", 0, **changes)
    with pytest.raises(ValueError, match="differs from the base plan"):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_p1_identical_descriptors_accept():
    plan = _default_plan()
    GroupedTransferPlan(base=plan.base, groups=plan.groups)


# ---------------------------------------------------------------------------
# P2 - group source layer / ID consistency
# ---------------------------------------------------------------------------


def test_p2_changed_source_layer_rejected():
    plan = _default_plan()
    groups = tuple(
        replace(group, source_layer=99) if group.group_id == "g2" else group
        for group in plan.groups
    )
    with pytest.raises(ValueError):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_p2_source_ids_must_be_the_production_sources():
    with pytest.raises(ValueError, match="KV source layers"):
        build_grouped_plan(
            _base_plan(),
            source_layer_ids=(2, 8, 14, 99),
            boundary_source_layer_ids=BOUNDARIES,
        )


# ---------------------------------------------------------------------------
# P3 - consumers must be the full execution-order union
# ---------------------------------------------------------------------------


def test_p3_consumer_shrink_rejected():
    plan = _default_plan()
    groups = tuple(
        replace(group, consumer_layers=(_layer(2),))
        if group.group_id == "g2"
        else group
        for group in plan.groups
    )
    with pytest.raises(ValueError, match="consumers"):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_p3_content_free_consumer_reorder_rejected():
    plan = _default_plan()
    original = _group(plan, "g2").consumer_layers
    groups = tuple(
        replace(
            group,
            consumer_layers=(original[1], original[0]) + original[2:],
        )
        if group.group_id == "g2"
        else group
        for group in plan.groups
    )
    with pytest.raises(ValueError, match="consumers"):
        GroupedTransferPlan(base=plan.base, groups=groups)


# ---------------------------------------------------------------------------
# P4 - boundary_resource must be the exact boundary member descriptor
# ---------------------------------------------------------------------------


def test_p4_boundary_alias_with_same_id_different_descriptor_rejected():
    plan = _default_plan()
    group = _group(plan, "g2")
    alias = _clone_resource(group.boundary_resource, tensor_indices=(999,))
    groups = tuple(
        replace(candidate, boundary_resource=alias)
        if candidate.group_id == "g2"
        else candidate
        for candidate in plan.groups
    )
    with pytest.raises(ValueError, match="boundary"):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_p4_boundary_resource_must_be_a_member():
    plan = _default_plan()
    foreign = _group(plan, "g8").boundary_resource
    groups = tuple(
        replace(candidate, boundary_resource=foreign)
        if candidate.group_id == "g2"
        else candidate
        for candidate in plan.groups
    )
    with pytest.raises(ValueError):
        GroupedTransferPlan(base=plan.base, groups=groups)


# ---------------------------------------------------------------------------
# P5 - resource order must not decide consumer order
# ---------------------------------------------------------------------------


def test_p5_index_first_resource_order_still_builds():
    base = _base_plan(index_first=True)
    assert base.resources[40].resource_id.endswith(".indexer.k_cache")
    plan = build_grouped_plan(
        base, source_layer_ids=SOURCES, boundary_source_layer_ids=BOUNDARIES
    )
    assert _group(plan, "g2").consumer_layers == tuple(
        _layer(index) for index in range(2, 8)
    )
    assert plan.transfer_resource_count == 11


def test_p5_boundary_first_resource_order_still_builds():
    # Rebuild the manifest with each group's boundary resource first.
    base = _base_plan()
    reordered = []
    for resource in base.resources:
        if resource.storage_kind == "boundary_state":
            reordered.insert(len(reordered) - 2, resource)
        else:
            reordered.append(resource)
    plan = build_grouped_plan(
        LayerwiseTransferPlan(LAYERS, tuple(reordered)),
        source_layer_ids=SOURCES,
        boundary_source_layer_ids=BOUNDARIES,
    )
    assert _group(plan, "g14").consumer_layers == tuple(
        _layer(index) for index in range(14, 20)
    )


# ---------------------------------------------------------------------------
# P6 - fixed default topology, folded index, missing/extra resources
# ---------------------------------------------------------------------------


def test_p6_folded_index_cache_rejected_at_registration():
    base = _base_plan()
    folded = _resource(
        f"{_stem(2)}.indexer.k_cache_folded",
        storage_kind="recompute",
        producer_layer=_layer(2),
        consumer_layers=(_layer(2),),
        tensor_index=200,
        engine_group_id=None,
    )
    with pytest.raises(ValueError, match="Folded index"):
        build_grouped_plan(
            LayerwiseTransferPlan(LAYERS, base.resources + (folded,)),
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_p6_missing_index_resource_rejected():
    base = _base_plan()
    resources = tuple(
        resource
        for resource in base.resources
        if resource.resource_id
        != f"{_stem(8)}.indexer.k_cache"
    )
    with pytest.raises(ValueError, match="Missing resources"):
        build_grouped_plan(
            LayerwiseTransferPlan(LAYERS, resources),
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_p6_extra_transfer_resource_rejected():
    base = _base_plan()
    extra = _resource(
        f"{_stem(3)}.long_kv_cache",
        storage_kind="token_range",
        producer_layer=_layer(3),
        consumer_layers=(_layer(3),),
        tensor_index=201,
        engine_group_id=0,
    )
    with pytest.raises(ValueError, match="not owned by a source|Unexpected resources"):
        build_grouped_plan(
            LayerwiseTransferPlan(LAYERS, base.resources + (extra,)),
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_p6_wrong_layer_count_rejected():
    base = _base_plan()
    resources = tuple(
        resource
        for resource in base.resources
        if "layers.39." not in resource.producer_layer
        and not any(
            "layers.39." in layer for layer in resource.consumer_layers
        )
    )
    with pytest.raises(ValueError, match="40 layers"):
        build_grouped_plan(
            LayerwiseTransferPlan(LAYERS[:39], resources),
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_p6_swa_engine_group_rejected():
    # The stock plan type already forbids an engine group on scratch/replay
    # resources, so the malformed base cannot even be constructed.
    base = _base_plan()
    with pytest.raises(ValueError, match="engine group"):
        _replace_resource(base, f"{_stem(5)}.swa_cache", engine_group_id=0)


def test_p6_group_count_must_be_four():
    plan = _default_plan()
    with pytest.raises(ValueError, match="not covered|4 groups"):
        GroupedTransferPlan(base=plan.base, groups=plan.groups[:3])


def test_p6_group_order_must_match_sources():
    plan = _default_plan()
    reordered = (plan.groups[1], plan.groups[0]) + plan.groups[2:]
    with pytest.raises(ValueError, match="ordered"):
        GroupedTransferPlan(base=plan.base, groups=reordered)


# ---------------------------------------------------------------------------
# P7 - source ids are real ints (no bool/float/str coercion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [True, False, 2.0, "2", -1])
def test_p7_source_ids_require_real_ints(bad):
    with pytest.raises(ValueError, match="non-negative int"):
        build_grouped_plan(
            _base_plan(),
            source_layer_ids=(bad, 8, 14, 20),
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_p7_boundary_ids_require_real_ints():
    with pytest.raises(ValueError, match="non-negative int"):
        build_grouped_plan(
            _base_plan(),
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=(2.0, 8, 14),
        )


def test_p7_numeric_string_sequence_rejected():
    with pytest.raises(ValueError, match="sequence"):
        build_grouped_plan(
            _base_plan(),
            source_layer_ids="281420",  # type: ignore[arg-type]
            boundary_source_layer_ids=BOUNDARIES,
        )


# ---------------------------------------------------------------------------
# Cross-group and ownership violations
# ---------------------------------------------------------------------------


def test_cross_group_consumer_of_other_segment_is_fine_only_if_in_union():
    # A consumer from another segment would widen the group union, so the
    # structural check must reject it.
    plan = _default_plan()
    group = _group(plan, "g2")
    widened = group.consumer_layers + (_layer(8),)
    groups = tuple(
        replace(candidate, consumer_layers=widened)
        if candidate.group_id == "g2"
        else candidate
        for candidate in plan.groups
    )
    with pytest.raises(ValueError, match="consumers"):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_duplicate_group_id_rejected():
    plan = _default_plan()
    groups = (plan.groups[0], replace(plan.groups[1], group_id="g2"))
    with pytest.raises(ValueError, match="Duplicate group ID"):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_resource_owned_by_two_groups_rejected():
    plan = _default_plan()
    groups = (
        plan.groups[0],
        replace(
            plan.groups[1],
            transfer_resources=plan.groups[1].transfer_resources
            + (plan.groups[0].transfer_resources[0],),
        ),
    ) + plan.groups[2:]
    with pytest.raises(ValueError):
        GroupedTransferPlan(base=plan.base, groups=groups)


def test_empty_groups_rejected():
    plan = _default_plan()
    with pytest.raises(ValueError, match="at least one group"):
        GroupedTransferPlan(base=plan.base, groups=())


def test_groups_must_be_a_tuple():
    plan = _default_plan()
    with pytest.raises(ValueError, match="tuple"):
        GroupedTransferPlan(base=plan.base, groups=list(plan.groups))


# ---------------------------------------------------------------------------
# Generic (non-default) plans: boundary and source ownership rules
# ---------------------------------------------------------------------------


def _small_resources() -> list[CacheTransferResource]:
    layers = tuple(_layer(index) for index in range(6))
    resources = [
        _resource(
            f"{_stem(index)}.swa_cache",
            storage_kind="recompute",
            producer_layer=_layer(index),
            consumer_layers=(_layer(index),),
            tensor_index=index,
            engine_group_id=None,
        )
        for index in range(6)
    ]
    resources.extend(
        (
            _resource(
                f"{_stem(1)}.long_kv_cache",
                storage_kind="token_range",
                producer_layer=_layer(1),
                consumer_layers=(layers[1], layers[2], layers[3]),
                tensor_index=10,
                engine_group_id=0,
            ),
            _resource(
                f"{_stem(1)}.indexer.k_cache",
                storage_kind="token_range",
                producer_layer=_layer(1),
                consumer_layers=(layers[1], layers[3]),
                tensor_index=11,
                engine_group_id=0,
            ),
            _resource(
                f"{_stem(1)}.compressor.state_cache",
                storage_kind="boundary_state",
                producer_layer=_layer(1),
                consumer_layers=(layers[1],),
                tensor_index=12,
                engine_group_id=None,
            ),
            _resource(
                f"{_stem(4)}.long_kv_cache",
                storage_kind="token_range",
                producer_layer=_layer(4),
                consumer_layers=(layers[4], layers[5]),
                tensor_index=13,
                engine_group_id=0,
            ),
            _resource(
                f"{_stem(4)}.indexer.k_cache",
                storage_kind="token_range",
                producer_layer=_layer(4),
                consumer_layers=(layers[4],),
                tensor_index=14,
                engine_group_id=0,
            ),
        )
    )
    return resources


def _small_plan(resources=None) -> LayerwiseTransferPlan:
    layers = tuple(_layer(index) for index in range(6))
    return LayerwiseTransferPlan(
        layer_order=layers,
        resources=tuple(resources if resources is not None else _small_resources()),
    )


def test_generic_plan_groups_two_sources():
    plan = build_grouped_plan(
        _small_plan(),
        source_layer_ids=(1, 4),
        boundary_source_layer_ids=(1,),
    )
    assert tuple(group.group_id for group in plan.groups) == ("g1", "g4")
    assert plan.transfer_resource_count == 5
    assert _group(plan, "g1").consumer_layers == (
        _layer(1),
        _layer(2),
        _layer(3),
    )
    assert plan.next_group("g1").group_id == "g4"


def test_generic_source_without_token_range_rejected():
    with pytest.raises(ValueError, match="owns no transfer resources|no token-range"):
        build_grouped_plan(
            _small_plan(),
            source_layer_ids=(1, 2, 4),
            boundary_source_layer_ids=(1,),
        )


def test_generic_transfer_resource_without_source_rejected():
    resources = _small_resources()
    orphan = _resource(
        f"{_stem(2)}.long_kv_cache",
        storage_kind="token_range",
        producer_layer=_layer(2),
        consumer_layers=(_layer(2),),
        tensor_index=30,
        engine_group_id=0,
    )
    with pytest.raises(ValueError, match="not owned by a source"):
        build_grouped_plan(
            _small_plan(resources + [orphan]),
            source_layer_ids=(1, 4),
            boundary_source_layer_ids=(1,),
        )


def test_generic_boundary_on_disallowed_source_rejected():
    with pytest.raises(ValueError, match="must not own a boundary"):
        build_grouped_plan(
            _small_plan(),
            source_layer_ids=(1, 4),
            boundary_source_layer_ids=(4,),
        )


def test_generic_missing_boundary_rejected():
    with pytest.raises(ValueError, match="missing its boundary"):
        build_grouped_plan(
            _small_plan(),
            source_layer_ids=(1, 4),
            boundary_source_layer_ids=(1, 4),
        )


def test_generic_swa_as_transferable_rejected():
    resources = _small_resources()
    mutated = [
        _clone_resource(resource, storage_kind="token_range", engine_group_id=0)
        if resource.resource_id == f"{_stem(2)}.swa_cache"
        else resource
        for resource in resources
    ]
    with pytest.raises(ValueError, match="SWA must be recompute"):
        build_grouped_plan(
            _small_plan(mutated),
            source_layer_ids=(1, 4),
            boundary_source_layer_ids=(1,),
        )


# ---------------------------------------------------------------------------
# F4: exact consumer segments are part of the fixed topology
# ---------------------------------------------------------------------------


def test_f4_consumer_segment_constant_matches_fixed_table():
    expected = tuple(
        (source, tuple(range(source, SEGMENT_END[source] + 1)))
        for source in SOURCES
    )
    assert plan_mod.DSV41_LW_V2_CONSUMER_SEGMENTS == expected


def test_f4_consumer_segments_match_engine_topology_constant():
    topology = (
        ROOT.parent
        / "vllm-ascend/vllm_ascend/models/deepseek_v41/topology.py"
    )
    if not topology.is_file():
        pytest.skip("vllm-ascend checkout is not next to this repository")
    tree = ast.parse(topology.read_text(encoding="utf-8"))
    engine_segments = None
    for node in tree.body:
        name = None
        if isinstance(node, ast.AnnAssign):
            name = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name == "DSV41_LW_V2_CONSUMER_SEGMENTS":
            engine_segments = ast.literal_eval(node.value)
    assert engine_segments is not None, (
        "vllm-ascend topology must define DSV41_LW_V2_CONSUMER_SEGMENTS"
    )
    assert plan_mod.DSV41_LW_V2_CONSUMER_SEGMENTS == engine_segments


def test_f4_wrong_but_self_consistent_segment_rejected():
    # Source 2's long KV and index K both shrink to 2-6: the group union is
    # consistent with its own resources, but it is not the fixed 2-7 segment.
    base = _replace_resource(
        _base_plan(),
        f"{_stem(2)}.long_kv_cache",
        consumer_layers=tuple(_layer(index) for index in range(2, 7)),
    )
    with pytest.raises(ValueError, match="fixed DSV4.1 segment"):
        build_grouped_plan(
            base,
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_f4_long_kv_must_cover_the_whole_segment():
    # The group union is exactly 2-7, but layer 7 is only a consumer of the
    # index K resource: the long KV descriptor itself no longer matches the
    # fixed layout and must be rejected.
    base = _replace_resource(
        _base_plan(),
        f"{_stem(2)}.indexer.k_cache",
        consumer_layers=(_layer(2), _layer(4), _layer(6), _layer(7)),
    )
    base = _replace_resource(
        base,
        f"{_stem(2)}.long_kv_cache",
        consumer_layers=tuple(_layer(index) for index in range(2, 7)),
    )
    with pytest.raises(ValueError, match="Long KV resource of group g2"):
        build_grouped_plan(
            base,
            source_layer_ids=SOURCES,
            boundary_source_layer_ids=BOUNDARIES,
        )


def test_f4_normal_plan_keeps_segment_while_index_stays_a_subset():
    plan = _default_plan()
    for source in SOURCES:
        segment = tuple(
            _layer(index)
            for index in range(source, SEGMENT_END[source] + 1)
        )
        group = _group(plan, f"g{source}")
        assert group.consumer_layers == segment
        long_kv = next(
            resource
            for resource in group.transfer_resources
            if resource.resource_id.endswith(".long_kv_cache")
        )
        assert long_kv.consumer_layers == segment
        index_k = next(
            resource
            for resource in group.transfer_resources
            if resource.resource_id.endswith(".indexer.k_cache")
        )
        assert set(index_k.consumer_layers) <= set(segment)

