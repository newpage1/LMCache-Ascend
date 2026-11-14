# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSV4.1 layerwise v2 connector skeleton.

The CPU test environment has no vLLM, so the stock connector module is
replaced by a stub before the v2 module is loaded. Everything below exercises
only the v2 routing/contract layer: strict configuration, group queueing
without inline waits, head-only wait/store, fail-closed seams, idempotency
and stale-generation handling. No real transport is involved.
"""

# Future
from __future__ import annotations

# Standard
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

# Third Party
import pytest
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

ROOT = Path(__file__).resolve().parents[2]
DSV41_LW_V2_DIR = ROOT / "lmcache_ascend/integration/vllm/dsv41_lw_v2"
PACKAGE = "lmcache_ascend.integration.vllm.dsv41_lw_v2"


# ---------------------------------------------------------------------------
# Import the v2 modules without importing the real stock connector (vLLM is
# not installed on CPU test hosts).
# ---------------------------------------------------------------------------


class StubLMCacheMPConnector:
    """Minimal stand-in for lmcache.integration.vllm.lmcache_mp_connector."""

    instances: list["StubLMCacheMPConnector"] = []

    def __init__(self, vllm_config, role, kv_cache_config=None) -> None:
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config
        self._connector_metadata = None
        StubLMCacheMPConnector.instances.append(self)


def _install_stock_stub() -> None:
    name = "lmcache.integration.vllm.lmcache_mp_connector"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)
    stub.LMCacheMPConnector = StubLMCacheMPConnector
    sys.modules[name] = stub


def _install_package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_v2_packages() -> None:
    _install_stock_stub()
    package_dir = ROOT / "lmcache_ascend"
    _install_package("lmcache_ascend", package_dir)
    _install_package("lmcache_ascend.integration", package_dir / "integration")
    _install_package(
        "lmcache_ascend.integration.vllm", package_dir / "integration/vllm"
    )
    _install_package(PACKAGE, DSV41_LW_V2_DIR)


def _import_modules():
    if PACKAGE + ".connector" in sys.modules:
        return (
            sys.modules[PACKAGE + ".plan"],
            sys.modules[PACKAGE + ".connector"],
        )
    _install_v2_packages()
    plan_mod = _load_module(PACKAGE + ".plan", DSV41_LW_V2_DIR / "plan.py")
    connector_mod = _load_module(
        PACKAGE + ".connector", DSV41_LW_V2_DIR / "connector.py"
    )
    return plan_mod, connector_mod


plan_mod, connector_mod = _import_modules()


# ---------------------------------------------------------------------------
# Real four-group plan (same fixed topology as the plan tests)
# ---------------------------------------------------------------------------


LAYERS = tuple(
    f"model.layers.{index}.self_attn.attn" for index in range(40)
)
SOURCES = (2, 8, 14, 20)
BOUNDARIES = (2, 8, 14)
SEGMENT_END = {2: 7, 8: 13, 14: 19, 20: 39}
HEAD_LAYERS = {source: LAYERS[source] for source in SOURCES}


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _stem(index: int) -> str:
    return f"model.layers.{index}.self_attn"


def _res(
    resource_id: str,
    *,
    storage_kind: str,
    producer_layer: str,
    consumer_layers,
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


def _build_grouped_plan():
    resources: list[CacheTransferResource] = [
        _res(
            f"{_stem(index)}.swa_cache",
            storage_kind="recompute",
            producer_layer=LAYERS[index],
            consumer_layers=(LAYERS[index],),
            tensor_index=index,
            engine_group_id=None,
        )
        for index in range(40)
    ]
    for source in SOURCES:
        segment = tuple(
            LAYERS[index]
            for index in range(source, SEGMENT_END[source] + 1)
        )
        index_readers = tuple(
            LAYERS[index]
            for index in range(source, SEGMENT_END[source] + 1)
            if (index - source) % 2 == 0
        )
        resources.extend(
            (
                _res(
                    f"{_stem(source)}.long_kv_cache",
                    storage_kind="token_range",
                    producer_layer=LAYERS[source],
                    consumer_layers=segment,
                    tensor_index=40 + source,
                    engine_group_id=0,
                ),
                _res(
                    f"{_stem(source)}.indexer.k_cache",
                    storage_kind="token_range",
                    producer_layer=LAYERS[source],
                    consumer_layers=index_readers,
                    tensor_index=80 + source,
                    engine_group_id=0,
                ),
            )
        )
        if source in BOUNDARIES:
            resources.append(
                _res(
                    f"{_stem(source)}.compressor.state_cache",
                    storage_kind="boundary_state",
                    producer_layer=LAYERS[source],
                    consumer_layers=(LAYERS[source],),
                    tensor_index=120 + source,
                    engine_group_id=None,
                )
            )
    base = LayerwiseTransferPlan(
        layer_order=LAYERS, resources=tuple(resources)
    )
    return plan_mod.build_grouped_plan(
        base,
        source_layer_ids=SOURCES,
        boundary_source_layer_ids=BOUNDARIES,
    )


_PLAN_CACHE = {}


def grouped_plan():
    if "plan" not in _PLAN_CACHE:
        _PLAN_CACHE["plan"] = _build_grouped_plan()
    return _PLAN_CACHE["plan"]


# ---------------------------------------------------------------------------
# Fakes: vLLM config, stock metadata, adapter, boundary hook
# ---------------------------------------------------------------------------


class _KvTransferConfig:
    def __init__(self, extra_config) -> None:
        self.kv_connector_extra_config = extra_config


class _SpeculativeConfig:
    def __init__(self, tokens: int) -> None:
        self.num_speculative_tokens = tokens


class FakeVllmConfig:
    def __init__(self, extra_config=None, speculative_tokens=0) -> None:
        self.kv_transfer_config = _KvTransferConfig(extra_config or {})
        self.speculative_config = (
            _SpeculativeConfig(speculative_tokens)
            if speculative_tokens
            else None
        )


@dataclass
class FakeOp:
    end: int
    start: int = 0
    skip_first_n_tokens: int = 0
    token_bytes: bytes = b"\x00\x00\x00\x01\x00\x00\x00\x02"
    block_ids: object = ((1, 2),)


@dataclass
class FakeEntry:
    request_id: str
    direction: str
    op: FakeOp
    cache_salt: object = None


@dataclass
class FakeMetadata:
    requests: list
    synthetic: bool = False


def _metadata(*entries, synthetic: bool = False) -> FakeMetadata:
    return FakeMetadata(requests=list(entries), synthetic=synthetic)


def _retrieve(request_id: str = "r1", end: int = 2048, **op_kwargs):
    return FakeEntry(
        request_id=request_id,
        direction="RETRIEVE",
        op=FakeOp(end=end, **op_kwargs),
    )


def _store(request_id: str = "r1", end: int = 2048, **op_kwargs):
    return FakeEntry(
        request_id=request_id,
        direction="STORE",
        op=FakeOp(end=end, **op_kwargs),
    )


class RecordingAdapter:
    def __init__(self) -> None:
        self.scheduled: dict[str, list[tuple[str, ...]]] = {}
        self.waited: list[tuple[str, tuple[str, ...]]] = []
        self.stores: list[tuple[str, tuple[str, ...]]] = []
        self.dropped: list[tuple[str, str]] = []
        self.drain_count = 0
        self.calls: list[tuple] = []

    def schedule_group_retrieves(self, group_id, request_ids, step):
        self.scheduled.setdefault(group_id, []).append(tuple(request_ids))
        self.calls.append(("schedule", group_id, tuple(request_ids)))

    def wait_group(self, group_id, request_ids, step, *, stream=None):
        self.waited.append((group_id, tuple(request_ids)))
        self.calls.append(("wait", group_id, stream))

    def submit_group_store(self, group_id, request_ids, step):
        self.stores.append((group_id, tuple(request_ids)))
        self.calls.append(("store", group_id, tuple(request_ids)))

    def drop_request(self, request_id, *, reason):
        self.dropped.append((request_id, reason))

    def drain_stores(self):
        self.drain_count += 1

    def take_finished(self):
        return set(), set()

    def take_failed_receiving(self):
        return set()


class RecordingBoundaryHook:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def capture_group_boundary(self, group_id):
        self.calls.append(("capture", group_id))

    def restore_group_boundary(self, group_id):
        self.calls.append(("restore", group_id))


def _key_factory_calls(adapter: RecordingAdapter | None = None) -> list[dict]:
    return _KEY_CALLS


_KEY_CALLS: list[dict] = []


def _stock_key_factory(**kwargs):
    _KEY_CALLS.append(dict(kwargs))
    return ("stock-key", kwargs["request_id"], kwargs["end"])


@pytest.fixture(autouse=True)
def _reset_key_calls():
    _KEY_CALLS.clear()


def _connector(**extra) -> "connector_mod.DSV41LwV2Connector":
    config = {"dsv41.lw_v2": True}
    config.update(extra)
    return connector_mod.DSV41LwV2Connector(
        FakeVllmConfig(config), role="worker"
    )


def _bound(**extra):
    connector = _connector(**extra)
    connector.bind_plan(grouped_plan())
    adapter = RecordingAdapter()
    hook = RecordingBoundaryHook()
    connector.bind_adapter(adapter)
    connector.bind_boundary_hook(hook)
    connector.bind_stock_key_factory(_stock_key_factory)
    return connector, adapter, hook


# ---------------------------------------------------------------------------
# Strict ``dsv41.lw_v2`` configuration
# ---------------------------------------------------------------------------


def test_flag_accepts_only_json_true():
    assert connector_mod.parse_lw_v2_flag({"dsv41.lw_v2": True}) is True
    assert (
        connector_mod.parse_lw_v2_flag({"dsv41.lw_v2": True, "other": 1})
        is True
    )


@pytest.mark.parametrize(
    "value", [False, 0, 1, "true", "True", None, [], {}, 1.0]
)
def test_flag_rejects_non_boolean_true(value):
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        connector_mod.parse_lw_v2_flag({"dsv41.lw_v2": value})


def test_flag_missing_rejected():
    with pytest.raises(
        connector_mod.DSV41LwV2ConfigError, match="must be set"
    ):
        connector_mod.parse_lw_v2_flag({})


def test_flag_none_config_rejected():
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        connector_mod.parse_lw_v2_flag(None)


@pytest.mark.parametrize("old_value", [False, True])
def test_old_flag_presence_rejected_even_when_false(old_value):
    with pytest.raises(
        connector_mod.DSV41LwV2ConfigError, match="retired"
    ):
        connector_mod.parse_lw_v2_flag(
            {"dsv41.lw_v2": True, "dsv41.layerwise": old_value}
        )


def test_old_flag_alone_rejected():
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        connector_mod.parse_lw_v2_flag({"dsv41.layerwise": True})


def test_connector_extra_config_reads_stock_field():
    config = FakeVllmConfig({"dsv41.lw_v2": True})
    assert connector_mod.connector_extra_config(config) == {
        "dsv41.lw_v2": True
    }


def test_connector_extra_config_defaults_to_empty():
    config = FakeVllmConfig(None)
    assert connector_mod.connector_extra_config(config) == {}
    config.kv_transfer_config = None
    assert connector_mod.connector_extra_config(config) == {}


def test_connector_extra_config_must_be_mapping():
    config = FakeVllmConfig(["not", "a", "mapping"])
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        connector_mod.connector_extra_config(config)


def test_constructor_requires_the_v2_flag_before_base_init():
    before = len(StubLMCacheMPConnector.instances)
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        connector_mod.DSV41LwV2Connector(FakeVllmConfig(None), role="worker")
    assert len(StubLMCacheMPConnector.instances) == before


def test_constructor_rejects_old_flag():
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        _connector(**{"dsv41.layerwise": True})


def test_constructor_rejects_intermediate_tensor_transfer():
    with pytest.raises(
        connector_mod.DSV41LwV2ConfigError, match="incompatible"
    ):
        _connector(**{"lmcache.mp.transfer_intermediate_tensors": True})


def test_constructor_rejects_speculative_decoding():
    config = {"dsv41.lw_v2": True}
    with pytest.raises(
        connector_mod.DSV41LwV2ConfigError, match="speculative"
    ):
        connector_mod.DSV41LwV2Connector(
            FakeVllmConfig(config, speculative_tokens=2), role="worker"
        )


def test_constructor_accepts_enabled_config():
    connector = _connector()
    assert connector.CONNECTOR_NAME == "DSV41LwV2Connector"
    assert connector.grouped_plan is None
    assert connector.metrics == {}


def test_registration_identity_and_head_set_are_complete():
    assert (
        connector_mod.DSV41_LW_V2_MODULE_PATH
        == "lmcache_ascend.integration.vllm.dsv41_lw_v2.connector"
    )
    assert connector_mod.DSV41_LW_V2_CONNECTOR_NAME == (
        "DSV41LwV2Connector"
    )
    connector, _adapter, _hook = _bound()
    plan = connector.grouped_plan
    assert plan is not None
    heads = [
        index
        for index in range(40)
        if connector.group_for_layer(_layer(index)) is not None
    ]
    assert heads == [2, 8, 14, 20]
    assert plan.transfer_resource_count == 11
    assert sum(
        len(group.token_range_resources) for group in plan.groups
    ) == 8
    assert sum(1 for group in plan.groups if group.has_boundary) == 3


# ---------------------------------------------------------------------------
# Shared step/request/operation types
# ---------------------------------------------------------------------------


def test_transfer_request_normalizes_flattened_block_ids():
    request = connector_mod.TransferRequest(
        request_id="r1",
        step_id=0,
        stock_key=object(),
        load_prefix_end=8,
        block_ids_by_engine_group=(1, 2),
    )
    assert request.block_ids_by_engine_group == ((1, 2),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_id": ""},
        {"step_id": -1},
        {"stock_key": None},
        {"load_prefix_end": 0, "store_prefix_end": 0},
        {"load_prefix_end": -1},
        {"block_ids_by_engine_group": ((1, True),)},
    ],
)
def test_transfer_request_rejects_invalid_fields(kwargs):
    base = dict(
        request_id="r1",
        step_id=0,
        stock_key=object(),
        load_prefix_end=8,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        connector_mod.TransferRequest(**base)


def test_step_view_rejects_duplicate_requests():
    key = object()
    request = connector_mod.TransferRequest(
        request_id="r1", step_id=0, stock_key=key, load_prefix_end=8
    )
    with pytest.raises(ValueError, match="duplicate"):
        connector_mod.StepView(
            step_id=0,
            registration_epoch=0,
            server_generation=0,
            synthetic=False,
            requests=(request, request),
        )


def test_step_view_rejects_mismatched_request_step():
    request = connector_mod.TransferRequest(
        request_id="r1", step_id=3, stock_key=object(), load_prefix_end=8
    )
    with pytest.raises(ValueError, match="belongs to step"):
        connector_mod.StepView(
            step_id=4,
            registration_epoch=0,
            server_generation=0,
            synthetic=False,
            requests=(request,),
        )


def test_group_op_key_carries_direction():
    load = connector_mod.GroupOpKey(0, 0, "r1", 1, "g2", "load")
    store = connector_mod.GroupOpKey(0, 0, "r1", 1, "g2", "store")
    assert load != store
    with pytest.raises(ValueError, match="direction"):
        connector_mod.GroupOpKey(0, 0, "r1", 1, "g2", "copy")


# ---------------------------------------------------------------------------
# Group-level routing of the vLLM hooks
# ---------------------------------------------------------------------------


def test_start_load_kv_queues_four_groups_without_waiting():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)

    assert list(adapter.scheduled) == ["g2", "g8", "g14", "g20"]
    assert adapter.scheduled["g2"] == [("r1",)]
    assert adapter.waited == []
    assert adapter.stores == []
    assert hook.calls == []
    assert connector.metrics["group_load_scheduled"] == 4


def test_start_load_kv_skips_requests_without_a_hit():
    connector, adapter, _hook = _bound()
    connector._connector_metadata = _metadata()
    connector.start_load_kv(None)
    assert adapter.calls == []


def test_non_head_layers_are_pure_mapping_queries():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)
    before = connector.metrics.copy()

    for layer_index in (0, 1, 3, 7, 9, 13, 15, 19, 21, 39):
        connector.wait_for_layer_load(_layer(layer_index))
        connector.save_kv_layer(_layer(layer_index), None, None)

    assert adapter.waited == []
    assert adapter.stores == []
    assert hook.calls == []
    assert connector.metrics == before


def test_head_waits_group_then_restores_boundary_and_is_idempotent():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)

    connector.wait_for_layer_load(_layer(2))
    assert adapter.waited == [("g2", ("r1",))]
    assert hook.calls == [("restore", "g2")]
    assert connector.metrics["group_load_ready"] == 1
    assert connector.metrics["boundary_restored"] == 1

    # Repeating through the hook or through the explicit facade must not
    # re-issue the wait or the restore.
    connector.wait_for_layer_load(_layer(2))
    connector.wait_group_load("g2")
    connector.restore_group_boundary("g2")
    assert len(adapter.waited) == 1
    assert len(hook.calls) == 1


def test_group_facades_and_hooks_dedupe_in_contract_order():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_retrieve(), _store())
    connector.start_load_kv(None)

    # T3 may call the facades first; the vLLM hooks must then be no-ops.
    connector.wait_group_load("g2")
    connector.restore_group_boundary("g2")
    connector.wait_for_layer_load(_layer(2))
    assert len(adapter.waited) == 1
    assert len(hook.calls) == 1

    connector.capture_group_boundary("g2")
    connector.submit_group_store("g2")
    connector.save_kv_layer(_layer(2), None, None)
    assert hook.calls[-1] == ("capture", "g2")
    assert adapter.stores == [("g2", ("r1",))]


def test_g20_head_waits_but_has_no_boundary_op():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_retrieve(), _store())
    connector.start_load_kv(None)

    connector.wait_for_layer_load(_layer(20))
    assert adapter.waited == [("g20", ("r1",))]
    assert hook.calls == []
    connector.restore_group_boundary("g20")
    assert hook.calls == []
    assert connector.metrics["boundary_restored"] == 0

    connector.save_kv_layer(_layer(20), None, None)
    assert hook.calls == []
    assert adapter.stores == [("g20", ("r1",))]


def test_store_only_step_captures_and_submits_per_group():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_store(end=2048))
    connector.start_load_kv(None)
    assert adapter.scheduled == {}
    assert connector.metrics["group_load_scheduled"] == 0

    for source in SOURCES:
        connector.save_kv_layer(_layer(source), None, None)

    assert adapter.stores == [
        ("g2", ("r1",)),
        ("g8", ("r1",)),
        ("g14", ("r1",)),
        ("g20", ("r1",)),
    ]
    assert hook.calls == [
        ("capture", "g2"),
        ("capture", "g8"),
        ("capture", "g14"),
    ]

    connector.wait_for_save()
    assert adapter.drain_count == 1
    assert connector.metrics["group_store_drained"] == 1


def test_wait_for_save_submits_pending_stores_and_drains():
    connector, adapter, _hook = _bound()
    connector._connector_metadata = _metadata(_store(end=2048))
    connector.start_load_kv(None)
    connector.save_kv_layer(_layer(8), None, None)

    connector.wait_for_save()
    assert ("g8", ("r1",)) in adapter.stores
    assert ("g2", ("r1",)) in adapter.stores
    assert adapter.drain_count == 1

    # Nothing pending on the next save hook: no duplicate drain.
    connector._connector_metadata = _metadata()
    connector.start_load_kv(None)
    connector.wait_for_save()
    assert adapter.drain_count == 1


def test_no_hit_decode_step_is_a_complete_noop():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata()
    connector.start_load_kv(None)

    connector.wait_for_layer_load(_layer(2))
    connector.save_kv_layer(_layer(2), None, None)
    connector.wait_for_save()

    assert adapter.calls == []
    assert adapter.drain_count == 0
    assert hook.calls == []
    assert connector.metrics["requests_dropped"] == 0


def test_synthetic_step_schedules_nothing():
    connector, adapter, hook = _bound()
    connector._connector_metadata = _metadata(_retrieve(), synthetic=True)
    connector.start_load_kv(None)

    connector.wait_for_layer_load(_layer(2))
    connector.save_kv_layer(_layer(2), None, None)
    connector.wait_for_save()

    assert adapter.calls == []
    assert hook.calls == []
    assert connector.metrics["group_load_scheduled"] == 0


def test_repeated_step_view_is_noted_once():
    connector, adapter, _hook = _bound()
    metadata = _metadata(_retrieve())
    connector._connector_metadata = metadata
    connector.start_load_kv(None)
    step = connector._lw_v2_current_step
    connector.start_load_kv(None)
    # A second forward is a new step: exactly one retrieve per group per step.
    assert all(
        calls == [("r1",), ("r1",)]
        for calls in adapter.scheduled.values()
    )
    assert connector._lw_v2_current_step is not None
    assert step is not None


# ---------------------------------------------------------------------------
# Fail-closed seams
# ---------------------------------------------------------------------------


def test_real_hit_without_adapter_fails_closed():
    connector = _connector()
    connector.bind_plan(grouped_plan())
    connector.bind_stock_key_factory(_stock_key_factory)
    connector._connector_metadata = _metadata(_retrieve())
    with pytest.raises(
        connector_mod.DSV41LwV2NotReadyError, match="adapter"
    ):
        connector.start_load_kv(None)


def test_real_hit_without_stock_key_factory_fails_closed():
    connector = _connector()
    connector.bind_plan(grouped_plan())
    connector.bind_adapter(RecordingAdapter())
    connector._connector_metadata = _metadata(_retrieve())
    with pytest.raises(
        connector_mod.DSV41LwV2NotReadyError, match="stock key"
    ):
        connector.start_load_kv(None)


def test_hooks_without_plan_fail_closed():
    connector = _connector()
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError):
        connector.wait_for_layer_load(_layer(2))
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError):
        connector.group_for_layer(_layer(2))


def test_boundary_restore_before_wait_fails_closed():
    connector, _adapter, _hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)
    with pytest.raises(
        connector_mod.DSV41LwV2NotReadyError, match="before its load"
    ):
        connector.restore_group_boundary("g2")


def test_real_boundary_hit_without_codec_fails_closed():
    connector = _connector()
    connector.bind_plan(grouped_plan())
    adapter = RecordingAdapter()
    connector.bind_adapter(adapter)
    connector.bind_stock_key_factory(_stock_key_factory)
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)
    connector.wait_group_load("g2")
    with pytest.raises(
        connector_mod.DSV41LwV2NotReadyError, match="BOUNDARY_NOT_BOUND"
    ):
        connector.restore_group_boundary("g2")


def test_store_without_adapter_fails_closed():
    connector = _connector()
    connector.bind_plan(grouped_plan())
    connector.bind_boundary_hook(RecordingBoundaryHook())
    connector.bind_stock_key_factory(_stock_key_factory)
    connector._connector_metadata = _metadata(_store(end=2048))
    connector.start_load_kv(None)
    connector.capture_group_boundary("g2")
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError):
        connector.submit_group_store("g2")


def test_unknown_group_raises_keyerror():
    connector, _adapter, _hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)
    for call in (
        connector.wait_group_load,
        connector.restore_group_boundary,
        connector.capture_group_boundary,
        connector.submit_group_store,
    ):
        with pytest.raises(KeyError):
            call("g99")


def test_metadata_must_expose_request_list():
    connector, _adapter, _hook = _bound()
    connector._connector_metadata = object()
    with pytest.raises(ValueError, match="requests"):
        connector.start_load_kv(None)


def test_adapter_must_implement_protocol():
    connector, _adapter, _hook = _bound()
    with pytest.raises(ValueError):
        connector.bind_adapter(object())


def test_boundary_hook_must_implement_protocol():
    connector, _adapter, _hook = _bound()
    with pytest.raises(ValueError):
        connector.bind_boundary_hook(object())


# ---------------------------------------------------------------------------
# Plan binding, generations and stopped requests
# ---------------------------------------------------------------------------


def test_plan_binding_is_idempotent_and_conflicts_are_rejected():
    connector = _connector()
    plan = grouped_plan()
    connector.bind_plan(plan)
    connector.bind_plan(plan)
    with pytest.raises(connector_mod.DSV41LwV2ConfigError):
        connector.bind_plan(_build_grouped_plan())
    connector.bind_plan(_build_grouped_plan(), replace=True)
    assert connector.grouped_plan is not None


def test_generation_change_drops_stale_operations():
    connector, adapter, _hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)

    connector.set_generations(registration_epoch=1, server_generation=0)
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)

    assert ("r1", "generation_changed") in adapter.dropped
    assert adapter.scheduled["g2"] == [("r1",), ("r1",)]
    assert connector.metrics["stale_ops_cleared"] == 4


def test_drop_request_filters_pending_ops_and_delegates():
    connector, adapter, _hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)
    connector.drop_request("r1", reason="request_aborted")
    assert adapter.dropped == [("r1", "request_aborted")]
    connector.wait_for_layer_load(_layer(2))
    assert adapter.waited == []


def test_step_view_merges_retrieve_and_store_endpoints():
    connector, _adapter, _hook = _bound()
    connector._connector_metadata = _metadata(
        _retrieve(end=2048, skip_first_n_tokens=16),
        _store(end=1024, skip_first_n_tokens=8),
    )
    connector.start_load_kv(None)
    step = connector._lw_v2_current_step
    assert step is not None
    assert len(step.requests) == 1
    request = step.requests[0]
    assert request.load_prefix_end == 2048
    assert request.store_prefix_end == 1024
    assert request.skip_first_n_tokens == 16
    assert _KEY_CALLS == [
        {
            "request_id": "r1",
            "token_bytes": FakeOp(end=0).token_bytes,
            "start": 0,
            "end": 2048,
            "cache_salt": None,
        }
    ]


def test_operation_keys_carry_request_group_and_direction():
    connector, _adapter, _hook = _bound()
    connector._connector_metadata = _metadata(_retrieve())
    connector.start_load_kv(None)

    load_keys = connector._lw_v2_router.operation_keys("load")
    store_keys = connector._lw_v2_router.operation_keys("store")
    assert {key.group_id for key in load_keys} == {"g2", "g8", "g14", "g20"}
    assert {key.direction for key in load_keys} == {"load"}
    assert {key.request_id for key in load_keys} == {"r1"}
    assert store_keys == ()


def test_operation_keys_empty_without_a_step():
    connector, _adapter, _hook = _bound()
    assert connector._lw_v2_router.operation_keys("load") == ()


def test_group_for_layer_query_on_connector():
    connector, _adapter, _hook = _bound()
    assert connector.group_for_layer(_layer(2)).group_id == "g2"
    assert connector.group_for_layer(_layer(3)) is None
    assert connector.group_for_layer("unknown") is None


# ---------------------------------------------------------------------------
# Metrics surface required by the contract
# ---------------------------------------------------------------------------


def test_metrics_surface_contains_contract_counters():
    connector, _adapter, _hook = _bound()
    metrics = connector.metrics
    for key in (
        "group_load_scheduled",
        "group_load_ready",
        "group_store_submitted",
        "group_store_drained",
        "boundary_captured",
        "boundary_restored",
        "requests_dropped",
        "stale_ops_cleared",
    ):
        assert key in metrics
