# SPDX-License-Identifier: Apache-2.0
"""Tests for the production adapter factory / transport wiring.

The CPU host has no vLLM, so the stock connector is stubbed exactly like the
other v2 tests. The *stock worker fields the production factory consumes* are
real attribute names taken from ``LMCacheMPWorkerAdapter``
(``kv_caches``/``req_client``/``lmcache_tokens_per_chunk``/``blocks_in_chunk``/
``parallel_strategy``/``instance_id``) and the request client is the typed
``RequestClient`` shape (the generated one has no ``submit_request``). This
proves the wiring, not the server: end-to-end stock hits still need the T6
hardware run.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
from pathlib import Path
import importlib.util
import subprocess
import sys
import types

# Third Party
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = ROOT / "lmcache_ascend/integration/vllm/dsv41_lw_v2"
PACKAGE = "lmcache_ascend.integration.vllm.dsv41_lw_v2"


# ---------------------------------------------------------------------------
# Module loading (shared loader pattern with the other v2 tests)
# ---------------------------------------------------------------------------


class _StubStockConnector:
    def __init__(self, vllm_config, role, kv_cache_config=None) -> None:
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config
        self._connector_metadata = None
        self.kv_caches: dict[str, object] = {}

    def register_kv_caches(self, kv_caches) -> None:
        self.kv_caches = dict(kv_caches)


def _install_stock_stub() -> None:
    name = "lmcache.integration.vllm.lmcache_mp_connector"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.LMCacheMPConnector = _StubStockConnector
        sys.modules[name] = stub


def _install_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_modules():
    order = ("plan", "connector", "adapter", "boundary", "production")
    if PACKAGE + ".production" in sys.modules:
        return tuple(sys.modules[PACKAGE + "." + name] for name in order)
    _install_stock_stub()
    root = ROOT / "lmcache_ascend"
    _install_package("lmcache_ascend", root)
    _install_package("lmcache_ascend.integration", root / "integration")
    _install_package(
        "lmcache_ascend.integration.vllm", root / "integration/vllm"
    )
    _install_package(PACKAGE, V2_DIR)
    loaded = {}
    for name in order:
        key = PACKAGE + "." + name
        loaded[name] = sys.modules.get(key) or _load_module(
            key, V2_DIR / f"{name}.py"
        )
    return tuple(loaded[name] for name in order)


plan_mod, connector_mod, adapter_mod, boundary_mod, production_mod = (
    _import_modules()
)


# ---------------------------------------------------------------------------
# vLLM config stub (same shape as the other v2 tests)
# ---------------------------------------------------------------------------


class _KvTransferConfig:
    def __init__(self, extra_config) -> None:
        self.kv_connector_extra_config = extra_config


class FakeVllmConfig:
    def __init__(self, extra_config=None) -> None:
        self.kv_transfer_config = _KvTransferConfig(extra_config or {})
        self.speculative_config = None


def _connector(worker_adapter=...):
    connector = connector_mod.DSV41LwV2Connector(
        FakeVllmConfig({"dsv41.lw_v2": True}), role="worker"
    )
    if worker_adapter is not ...:
        connector.worker_adapter = worker_adapter
    return connector


# ---------------------------------------------------------------------------
# Live registration double
# ---------------------------------------------------------------------------

LAYERS = 40
SOURCES = (2, 8, 14, 20)
BOUNDARIES = (2, 8, 14)


def _cache_name(layer: int, kind: str) -> str:
    return f"model.layers.{layer}.self_attn.{kind}"


def _registration_caches() -> dict[str, torch.Tensor]:
    caches: dict[str, torch.Tensor] = {}
    for layer in range(LAYERS):
        tensor = torch.zeros((8, 2, 4), dtype=torch.float32)
        caches[_cache_name(layer, "swa_cache")] = tensor
        if layer in SOURCES:
            caches[_cache_name(layer, "long_kv_cache")] = tensor
            caches[_cache_name(layer, "indexer.k_cache")] = tensor
        if layer in BOUNDARIES:
            caches[_cache_name(layer, "compressor.state_cache")] = tensor
    return caches


class _EngineGroupInfo:
    def __init__(self, engine_group_id: int, layer_indices) -> None:
        self.engine_group_id = engine_group_id
        self.layer_indices = tuple(layer_indices)


def _engine_groups():
    return [_EngineGroupInfo(0, range(LAYERS))]


class FakeTypedRequestClient:
    """Typed ``RequestClient``: resource-block RPCs, no ``submit_request``."""

    def __init__(self) -> None:
        self.register_calls = []
        self.ack_generation = 7

    def register_resource_blocks(self, registration):
        self.register_calls.append(registration)
        future: Future = Future()
        future.set_result(
            adapter_mod.ResourceBlockAck(
                instance_id=registration.instance_id,
                registration_epoch=registration.registration_epoch,
                plan_hash=registration.plan_hash,
                layout_hash=registration.layout_hash,
                server_generation=self.ack_generation,
                capabilities=adapter_mod.REQUIRED_CAPABILITIES,
            )
        )
        return future

    def store_resource_block(self, spec, payload):
        raise AssertionError("store is not part of registration")

    def retrieve_resource_block(self, spec):
        raise AssertionError("retrieve is not part of registration")

    def lookup_resource_blocks(self, request):
        raise AssertionError("lookup is not part of registration")


class _ParallelStrategy:
    def __init__(self, worker_id=0) -> None:
        self.kv_worker_id = worker_id


class FakeWorkerAdapter:
    def __init__(
        self,
        *,
        kv_caches=None,
        client=None,
        chunk_size: int = 64,
        blocks_in_chunk: int = 4,
        with_client: bool = True,
    ) -> None:
        self.kv_caches = (
            _registration_caches() if kv_caches is None else kv_caches
        )
        self.engine_group_infos = _engine_groups()
        self.req_client = (
            FakeTypedRequestClient() if with_client else None
        )
        if client is not None:
            self.req_client = client
        self.model_name = "DeepSeek-V4.1-Flash"
        self.world_size = 1
        self.instance_id = 123456789
        self.lmcache_tokens_per_chunk = chunk_size
        self.blocks_in_chunk = blocks_in_chunk
        self.parallel_strategy = _ParallelStrategy()


# ---------------------------------------------------------------------------
# Default production factory
# ---------------------------------------------------------------------------


def test_default_factory_builds_real_adapter_and_codec():
    connector = _connector(FakeWorkerAdapter())
    assert (
        connector._lw_v2_adapter_factory
        is connector_mod._USE_DEFAULT_ADAPTER_FACTORY
    )
    connector.register_kv_caches(FakeWorkerAdapter().kv_caches)

    adapter = connector.adapter
    assert isinstance(adapter, adapter_mod.DSV41LwV2GroupAdapter)
    assert adapter.registration_epoch == 0
    assert adapter.server_generation == 7
    assert connector._lw_v2_server_generation == 7
    codec = connector.boundary_codec
    assert isinstance(codec, boundary_mod.DSV41LwV2BoundaryCodec)
    assert codec.adapter is adapter
    assert connector.current_step is None
    # The live registration itself produced the grouped plan.
    groups = connector.grouped_plan.groups
    assert [group.group_id for group in groups] == ["g2", "g8", "g14", "g20"]
    assert len(adapter.layouts) == 11
    # Typed client: registration went through the typed RPC, not
    # submit_request (which this fake does not even expose).
    client = connector.worker_adapter.req_client
    assert len(client.register_calls) == 1
    assert (
        client.register_calls[0].plan_hash
        == production_mod.compute_plan_hash(connector.grouped_plan)
    )


def test_default_factory_uses_worker_registration_not_the_argument():
    worker = FakeWorkerAdapter()
    connector = _connector(worker)
    # A caller-supplied dict that is *not* the worker registration must not
    # be used to build layouts: the factory always reads the stock cache map.
    connector.register_kv_caches({"unrelated": torch.zeros(1)})
    assert connector.adapter is not None
    assert len(connector.adapter.layouts) == 11


def test_repeated_registration_is_idempotent():
    calls = []

    def factory(plan, connector):
        calls.append(1)
        return production_mod.create_production_adapter(plan, connector)

    connector = _connector(FakeWorkerAdapter())
    connector.bind_adapter_factory(factory)
    connector.register_kv_caches({})
    adapter = connector.adapter
    codec = connector.boundary_codec
    connector.register_kv_caches({})
    connector.register_kv_caches({})
    assert len(calls) == 1
    assert connector.adapter is adapter
    assert connector.boundary_codec is codec


# ---------------------------------------------------------------------------
# Fail-closed dependencies
# ---------------------------------------------------------------------------


def test_missing_worker_adapter_fails_closed():
    connector = _connector()
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError) as info:
        connector.register_kv_caches({})
    assert "worker_adapter" in str(info.value)


def test_missing_kv_caches_fails_closed():
    worker = FakeWorkerAdapter(kv_caches={})
    connector = _connector(worker)
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError) as info:
        connector.register_kv_caches({})
    assert "worker_adapter.kv_caches" in str(info.value)


def test_missing_request_client_fails_closed():
    worker = FakeWorkerAdapter(with_client=False)
    connector = _connector(worker)
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError) as info:
        connector.register_kv_caches({})
    assert "worker_adapter.req_client" in str(info.value)


def test_missing_chunk_geometry_fails_closed():
    worker = FakeWorkerAdapter(chunk_size=0)
    connector = _connector(worker)
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError) as info:
        connector.register_kv_caches({})
    assert "lmcache_tokens_per_chunk" in str(info.value)


def test_no_factory_fails_closed_instead_of_silent_inert():
    connector = _connector(FakeWorkerAdapter())
    connector.bind_adapter_factory(None)
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError) as info:
        connector.register_kv_caches({})
    assert "adapter factory" in str(info.value)
    assert connector.adapter is None


def test_unbuildable_registration_fails_closed():
    caches = _registration_caches()
    # Drop one layer: the fixed 40-layer DSV4.1 registration must be rejected.
    for kind in (
        "swa_cache",
        "long_kv_cache",
        "indexer.k_cache",
        "compressor.state_cache",
    ):
        caches.pop(_cache_name(39, kind), None)
    connector = _connector(FakeWorkerAdapter(kv_caches=caches))
    with pytest.raises(connector_mod.DSV41LwV2NotReadyError) as info:
        connector.register_kv_caches({})
    assert "kv_caches" in str(info.value)


# ---------------------------------------------------------------------------
# Plan derivation
# ---------------------------------------------------------------------------


def test_plan_derivation_matches_fixed_topology():
    plan = production_mod.build_grouped_plan_from_registration(
        _registration_caches(), _engine_groups()
    )
    assert [group.group_id for group in plan.groups] == [
        "g2",
        "g8",
        "g14",
        "g20",
    ]
    assert plan.transfer_resource_count == 11
    boundaries = [group.group_id for group in plan.groups if group.has_boundary]
    assert boundaries == ["g2", "g8", "g14"]
    assert len(plan.base.resources) == 51
    assert production_mod.compute_plan_hash(plan).startswith("sha256:")
    assert (
        production_mod.compute_plan_hash(plan)
        == production_mod.compute_plan_hash(plan)
    )


def test_plan_derivation_rejects_folded_index_cache():
    caches = _registration_caches()
    name = _cache_name(2, "indexer.k_cache")
    caches[_cache_name(2, "indexer.k_cache_folded")] = caches.pop(name)
    with pytest.raises(production_mod.DSV41LwV2ProductionUnavailable) as info:
        production_mod.build_grouped_plan_from_registration(
            caches, _engine_groups()
        )
    assert "indexer.k_cache" in str(info.value)


def test_plan_hash_rejects_shuffled_registration_order():
    caches = _registration_caches()
    plan = production_mod.build_grouped_plan_from_registration(
        caches, _engine_groups()
    )
    shuffled = dict(reversed(list(caches.items())))
    with pytest.raises(production_mod.DSV41LwV2ProductionUnavailable) as info:
        production_mod.compute_layout_hash(plan, shuffled)
    assert "tensor-index order" in str(info.value)


def test_plan_derivation_requires_source_engine_group():
    groups = [_EngineGroupInfo(0, range(0, 20))]
    with pytest.raises(production_mod.DSV41LwV2ProductionUnavailable) as info:
        production_mod.build_grouped_plan_from_registration(
            _registration_caches(), groups
        )
    assert "layer 20" in str(info.value)


# ---------------------------------------------------------------------------
# Transport selection (typed stock client vs generic submit_request client)
# ---------------------------------------------------------------------------


def test_transport_prefers_typed_stock_client():
    client = FakeTypedRequestClient()
    transport = adapter_mod.MPResourceBlockTransport(client)
    assert transport._typed is True
    assert not hasattr(client, "submit_request")


def test_transport_accepts_generic_submit_request_client():
    class GenericClient:
        def __init__(self) -> None:
            self.calls = []

        def submit_request(self, operation, payloads):
            self.calls.append((operation, payloads))
            future: Future = Future()
            future.set_result(None)
            return future

    client = GenericClient()
    transport = adapter_mod.MPResourceBlockTransport(client)
    assert transport._typed is False
    transport.retrieve_resource_block(object())
    assert client.calls[0][0] == "retrieve_resource_block"


def test_transport_rejects_partial_typed_client():
    class PartialClient:
        def register_resource_blocks(self, registration):
            raise AssertionError("unused")

    with pytest.raises(TypeError) as info:
        adapter_mod.MPResourceBlockTransport(PartialClient())
    assert "partial" in str(info.value)


# ---------------------------------------------------------------------------
# Old v1 branch stays untouched
# ---------------------------------------------------------------------------


def test_old_layerwise_sources_are_not_modified():
    try:
        completed = subprocess.run(
            ["git", "diff", "--name-only", "7ec8501"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pytest.skip("git is unavailable")
    if completed.returncode != 0:  # pragma: no cover
        pytest.skip("base commit is unavailable")
    changed = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }
    legacy_prefixes = (
        "lmcache_ascend/integration/vllm/dsv41_mp_adapter.py",
        "lmcache_ascend/integration/vllm/dsv41_boundary_codec.py",
        "lmcache_ascend/integration/vllm/dsv41_boundary_state.py",
        "lmcache_ascend/integration/vllm/dsv41_mp_connector.py",
        "lmcache_ascend/integration/vllm/dsv41_layerwise.py",
        "lmcache_ascend/integration/vllm/dsv41_layerwise/",
    )
    assert not any(path.startswith(legacy_prefixes) for path in changed)
    # The new modules must not import the v1 runtime either.
    for name in ("production.py", "connector.py", "adapter.py", "boundary.py"):
        text = (V2_DIR / name).read_text()
        assert "dsv41_mp_adapter" not in text
        assert "dsv41_boundary_codec" not in text
        assert "dsv41_mp_connector" not in text
