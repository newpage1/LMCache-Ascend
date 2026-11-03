# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import importlib.util

# Third Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
import msgspec
import pytest

SPEC = importlib.util.spec_from_file_location(
    "dsv41_mp_layerwise_adapter",
    Path(__file__).resolve().parents[2]
    / "lmcache_ascend/integration/vllm/layerwise.py",
)
assert SPEC is not None and SPEC.loader is not None
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)
build_dsv41_layerwise_plan = ADAPTER.build_dsv41_layerwise_plan


@dataclass(frozen=True)
class CacheResource:
    cache_name: str
    cache_kind: str
    producer_layer: int
    consumer_layers: tuple[int, ...]


def _resource(name, kind, producer, consumers):
    return CacheResource(
        f"model.layers.{producer}.self_attn.{name}", kind, producer, consumers
    )


@pytest.fixture
def registration():
    descriptions = (
        _resource("swa_cache", "swa", 0, (0,)),
        _resource("swa_cache", "swa", 1, (1,)),
        _resource("swa_cache", "swa", 2, (2,)),
        _resource("long_kv_cache", "long_kv", 0, (0, 1, 2)),
        _resource("indexer.k_cache", "index_k", 0, (0, 2)),
        _resource("compressor.state_cache", "compressor_state", 0, (0,)),
        _resource("indexer.k_cache_folded", "folded_index_k", 0, (0, 2)),
    )
    topology = SimpleNamespace(
        layers=tuple(SimpleNamespace(layer_idx=index) for index in range(3)),
        cache_resources=lambda **kwargs: descriptions,
    )
    names = [resource.cache_name for resource in descriptions[:-1]]
    groups = [EngineGroupInfo(0, (3, 4)), EngineGroupInfo(1, (0, 1, 2))]
    return topology, names, groups


def test_adapter_preserves_registration_indices_and_distinct_consumers(registration):
    topology, names, groups = registration
    plan = build_dsv41_layerwise_plan(topology, names, groups)
    dependencies = plan.required_by("model.layers.1.self_attn.attn")
    assert [resource.resource_id for resource in dependencies] == [names[1], names[3]]
    assert dependencies[1].tensor_indices == (3,)
    assert dependencies[1].engine_group_id == 0
    assert plan.resources[5].storage_kind == "boundary_state"
    assert plan.resources[5].engine_group_id is None
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(plan), type=type(plan)) == plan


def test_folded_index_is_conditional_on_actual_registration(registration):
    topology, names, groups = registration
    names.append("model.layers.0.self_attn.indexer.k_cache_folded")
    groups.append(EngineGroupInfo(0, (6,)))
    plan = build_dsv41_layerwise_plan(topology, names, groups)
    assert plan.resources[-1].tensor_indices == (6,)
    assert plan.resources[-1].consumer_layers == (
        "model.layers.0.self_attn.attn",
        "model.layers.2.self_attn.attn",
    )


def test_bounded_replay_swa_is_explicitly_excluded_from_transfers(registration):
    topology, names, groups = registration
    groups[1] = EngineGroupInfo(1, (0, 1))
    with pytest.raises(ValueError, match="missing its engine group"):
        build_dsv41_layerwise_plan(topology, names, groups)
    plan = build_dsv41_layerwise_plan(
        topology, names, groups, replayed_cache_names=(names[2],)
    )
    assert plan.resources[2].storage_kind == "recompute"
    assert plan.resources[2].engine_group_id is None


def test_missing_source_cache_is_rejected(registration):
    topology, names, _ = registration
    names.remove("model.layers.0.self_attn.long_kv_cache")
    groups = [EngineGroupInfo(0, (3,)), EngineGroupInfo(1, (0, 1, 2))]
    with pytest.raises(ValueError, match="Missing required.*long_kv_cache"):
        build_dsv41_layerwise_plan(topology, names, groups)


def test_scratch_ring_cannot_be_reported_as_token_range_kv(registration):
    topology, names, groups = registration
    groups.append(EngineGroupInfo(2, (5,)))
    with pytest.raises(ValueError, match="Scratch/replay cache must be excluded"):
        build_dsv41_layerwise_plan(topology, names, groups)


def test_unrecognized_draft_cache_is_not_silently_omitted(registration):
    topology, names, groups = registration
    names.append("mtp.layers.0.self_attn.swa_cache")
    with pytest.raises(ValueError, match="outside the DSV4.1 backbone"):
        build_dsv41_layerwise_plan(topology, names, groups)


def test_non_swa_cache_cannot_be_marked_as_replay(registration):
    topology, names, groups = registration
    with pytest.raises(ValueError, match="Only SWA"):
        build_dsv41_layerwise_plan(
            topology, names, groups, replayed_cache_names=(names[5],)
        )


def test_registration_cannot_assign_one_tensor_to_two_block_spaces(registration):
    topology, names, groups = registration
    groups.append(EngineGroupInfo(4, (3,)))
    with pytest.raises(ValueError, match="multiple groups"):
        build_dsv41_layerwise_plan(topology, names, groups)
