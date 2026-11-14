# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSV4.1 layerwise v2 group transport adapter and codec.

The CPU test host has no vLLM and no LMCache server, so the transport is a
small in-process fake that implements the same ``resource_blocks_v1`` future
semantics. The tests exercise only the v2 data plane: group retrieve/store,
terminal events, per-group/request drop, stale generation, wait fallback,
drain and the schema-2 boundary codec (capture -> store -> retrieve ->
restore, relocation, tamper and missing-payload failures).

No hardware, no real RPC.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import importlib.util
import struct
import sys
import threading
import types

# Third Party
import pytest
import torch
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

ROOT = Path(__file__).resolve().parents[2]
V2_DIR = ROOT / "lmcache_ascend/integration/vllm/dsv41_lw_v2"
PACKAGE = "lmcache_ascend.integration.vllm.dsv41_lw_v2"


# ---------------------------------------------------------------------------
# Load the v2 modules without importing the stock connector (no vLLM).
# ---------------------------------------------------------------------------


class _StubConnector:
    instances: list["_StubConnector"] = []

    def __init__(self, vllm_config, role, kv_cache_config=None) -> None:
        self.vllm_config = vllm_config
        self.role = role
        self.kv_cache_config = kv_cache_config
        self._connector_metadata = None
        self.kv_caches = {}
        self.stock_finished_sending: set[str] = set()
        self.stock_finished_receiving: set[str] = set()
        self.stock_failed_recving: set[str] = set()

    def register_kv_caches(self, kv_caches) -> None:
        # The stock base registers layouts with the LMCache server; the
        # CPU stub only records the caches the v2 factory reads back.
        self.kv_caches = dict(kv_caches)
        _StubConnector.instances.append(self)

    def get_finished(self, finished_req_ids):
        return (
            set(self.stock_finished_sending),
            set(self.stock_finished_receiving),
        )

    def get_transfer_results(self, finished_req_ids):
        sending, receiving = self.get_finished(finished_req_ids)
        return types.SimpleNamespace(
            finished_sending=sending,
            finished_recving=receiving,
            failed_recving=set(self.stock_failed_recving),
        )


def _install_stock_stub() -> None:
    name = "lmcache.integration.vllm.lmcache_mp_connector"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.LMCacheMPConnector = _StubConnector
        sys.modules[name] = stub


def _install_package(name: str, path: Path) -> None:
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


def _import_v2():
    if PACKAGE + ".boundary" in sys.modules:
        return tuple(
            sys.modules[PACKAGE + "." + name]
            for name in ("plan", "connector", "adapter", "boundary")
        )
    _install_stock_stub()
    root = ROOT / "lmcache_ascend"
    _install_package("lmcache_ascend", root)
    _install_package("lmcache_ascend.integration", root / "integration")
    _install_package(
        "lmcache_ascend.integration.vllm", root / "integration/vllm"
    )
    _install_package(PACKAGE, V2_DIR)
    plan_mod = _load_module(PACKAGE + ".plan", V2_DIR / "plan.py")
    connector_mod = _load_module(
        PACKAGE + ".connector", V2_DIR / "connector.py"
    )
    adapter_mod = _load_module(PACKAGE + ".adapter", V2_DIR / "adapter.py")
    boundary_mod = _load_module(
        PACKAGE + ".boundary", V2_DIR / "boundary.py"
    )
    return plan_mod, connector_mod, adapter_mod, boundary_mod


plan_mod, connector_mod, adapter_mod, boundary_mod = _import_v2()

# The cross-repo F1 test loads the *real* vllm-ascend boundary DTO module. It
# only needs its package skeleton (the module itself imports torch only), not
# the vLLM runtime.
VLLM_ASCEND = ROOT.parent / "vllm-ascend"


def _load_engine_boundary() -> types.ModuleType:
    """Load vllm-ascend's ``BoundaryRequestView``/``BoundaryTensorView``."""
    packages = (
        ("vllm_ascend", VLLM_ASCEND / "vllm_ascend"),
        ("vllm_ascend.attention", VLLM_ASCEND / "vllm_ascend/attention"),
        ("vllm_ascend.models", VLLM_ASCEND / "vllm_ascend/models"),
        (
            "vllm_ascend.models.deepseek_v41",
            VLLM_ASCEND / "vllm_ascend/models/deepseek_v41",
        ),
    )
    for name, path in packages:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package
    topology = "vllm_ascend.models.deepseek_v41.topology"
    if topology not in sys.modules:
        _load_module(
            topology,
            VLLM_ASCEND / "vllm_ascend/models/deepseek_v41/topology.py",
        )
    group_module = "vllm_ascend.attention.dsv41_lw_v2"
    if group_module not in sys.modules:
        _load_module(
            group_module,
            VLLM_ASCEND / "vllm_ascend/attention/dsv41_lw_v2.py",
        )
    name = "vllm_ascend.attention.dsv41_lw_v2_boundary"
    return sys.modules.get(name) or _load_module(
        name, VLLM_ASCEND / "vllm_ascend/attention/dsv41_lw_v2_boundary.py"
    )


# ---------------------------------------------------------------------------
# Real four-group plan (fixed DSV4.1 topology)
# ---------------------------------------------------------------------------

SOURCES = (2, 8, 14, 20)
BOUNDARIES = (2, 8, 14)
SEGMENT_END = {2: 7, 8: 13, 14: 19, 20: 39}
CHUNK_SIZE = 4
BLOCK_SIZE = 2
TOKENS = 8


def _layer(index: int) -> str:
    return f"model.layers.{index}.self_attn.attn"


def _stem(index: int) -> str:
    return f"model.layers.{index}.self_attn"


LAYERS = tuple(_layer(index) for index in range(40))


def _res(resource_id, *, kind, producer, consumers, tensor_index, group):
    return CacheTransferResource(
        resource_id=resource_id,
        tensor_indices=(tensor_index,),
        engine_group_id=group,
        producer_layer=producer,
        consumer_layers=tuple(consumers),
        storage_kind=kind,  # type: ignore[arg-type]
    )


def _build_plan():
    resources = [
        _res(
            f"{_stem(index)}.swa_cache",
            kind="recompute",
            producer=_layer(index),
            consumers=(_layer(index),),
            tensor_index=index,
            group=None,
        )
        for index in range(40)
    ]
    for source in SOURCES:
        segment = tuple(
            _layer(index)
            for index in range(source, SEGMENT_END[source] + 1)
        )
        readers = tuple(
            _layer(index)
            for index in range(source, SEGMENT_END[source] + 1)
            if (index - source) % 2 == 0
        )
        resources.extend(
            (
                _res(
                    f"{_stem(source)}.long_kv_cache",
                    kind="token_range",
                    producer=_layer(source),
                    consumers=segment,
                    tensor_index=40 + source,
                    group=0,
                ),
                _res(
                    f"{_stem(source)}.indexer.k_cache",
                    kind="token_range",
                    producer=_layer(source),
                    consumers=readers,
                    tensor_index=80 + source,
                    group=0,
                ),
            )
        )
        if source in BOUNDARIES:
            resources.append(
                _res(
                    f"{_stem(source)}.compressor.state_cache",
                    kind="boundary_state",
                    producer=_layer(source),
                    consumers=(_layer(source),),
                    tensor_index=120 + source,
                    group=None,
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


PLAN = _build_plan()
PLAN_HASH = "plan-abc"
LAYOUT_HASH = "layout-abc"
MODEL = "model"
WORLD = 1
SALT = ""


def _token_bytes(count: int) -> bytes:
    return struct.pack(f">{count}I", *range(count))


def _chunk_hash(tokens: bytes, start: int, end: int) -> bytes:
    return hashlib.sha256(tokens[start * 4 : end * 4]).digest()


def _stock_key(
    request_id: str, tokens: bytes, end: int = TOKENS
) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name=MODEL,
        world_size=WORLD,
        worker_id=0,
        token_bytes=tokens,
        start=0,
        end=end,
        request_id=request_id,
        cache_salt=SALT,
    )


def _request(
    request_id="r1",
    *,
    step_id=0,
    load_prefix_end=TOKENS,
    store_prefix_end=0,
    store_start=0,
    blocks=((0, 1, 2, 3),),
    tokens=None,
):
    tokens = _token_bytes(TOKENS) if tokens is None else tokens
    return connector_mod.TransferRequest(
        request_id=request_id,
        step_id=step_id,
        stock_key=_stock_key(request_id, tokens, max(load_prefix_end, store_prefix_end)),
        load_prefix_end=load_prefix_end,
        store_prefix_end=store_prefix_end,
        store_start=store_start,
        block_ids_by_engine_group=blocks,
    )


def _step(request, *, epoch=0, generation=0, synthetic=False):
    return connector_mod.StepView(
        step_id=request.step_id,
        registration_epoch=epoch,
        server_generation=generation,
        synthetic=synthetic,
        requests=(request,),
    )


# ---------------------------------------------------------------------------
# Fake resource-block server / transport
# ---------------------------------------------------------------------------


@dataclass
class _FakeServer:
    capabilities: tuple[str, ...] = adapter_mod.REQUIRED_CAPABILITIES
    delay: float = 0.0
    store_calls: list = field(default_factory=list)
    store_specs: list = field(default_factory=list)
    retrieve_calls: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    boundaries: dict = field(default_factory=dict)
    registration: object = None
    registration_ack: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def store(self, spec, payload):
        with self.lock:
            self.store_calls.append((spec.resource_id, spec.start_token, spec.end_token))
            self.store_specs.append(spec)
            if spec.resource_id in _BOUNDARY_IDS:
                if not payload.startswith(adapter_mod.BOUNDARY_PAYLOAD_MAGIC):
                    return self._result(spec, "failed", "PAYLOAD_CORRUPT")
                envelope = boundary_mod.decode_boundary_schema2(payload)
                if envelope.header["boundary_end"] != spec.prefix_end:
                    return self._result(spec, "invalid", "BOUNDARY_PREFIX_MISMATCH")
                self.boundaries[self._boundary_key(spec)] = payload
                return self._result(spec, "ok")
            planes = adapter_mod.decode_token_payload(
                payload, resource_id=spec.resource_id
            )
            expected_rows = len(spec.block_ids)
            total = sum(plane.shape[0] for plane in planes)
            if not planes or total != expected_rows:
                return self._result(spec, "failed", "PAYLOAD_CORRUPT")
            self.blocks[self._token_key(spec, spec.start_token, spec.end_token)] = (
                payload
            )
            return self._result(spec, "ok")

    def retrieve(self, spec):
        with self.lock:
            self.retrieve_calls.append(
                (spec.resource_id, spec.start_token, spec.end_token)
            )
            if spec.resource_id in _BOUNDARY_IDS:
                payload = self.boundaries.get(self._boundary_key(spec))
            else:
                payload = self.blocks.get(
                    self._token_key(spec, spec.start_token, spec.end_token)
                )
        if payload is None:
            return self._result(spec, "miss", "RESOURCE_MISSING"), b""
        return self._result(spec, "ok"), payload

    def _result(self, spec, status, code=""):
        return adapter_mod.ResourceBlockResult(
            operation_id=spec.operation_id,
            direction=spec.direction,
            resource_id=spec.resource_id,
            registration_epoch=spec.registration_epoch,
            server_generation=spec.server_generation,
            status=status,
            error_code=code,
        )

    def _token_key(self, spec, start, end):
        return (
            spec.plan_hash,
            spec.layout_hash,
            spec.stock_key.model_name,
            spec.stock_key.world_size,
            spec.stock_key.cache_salt,
            spec.resource_id,
            start,
            end,
            _chunk_hash(spec.stock_key.token_bytes, start, end),
        )

    def _boundary_key(self, spec):
        return (
            spec.plan_hash,
            spec.layout_hash,
            spec.resource_id,
            spec.prefix_end,
            _chunk_hash(spec.stock_key.token_bytes, 0, spec.prefix_end),
        )


class _PlanMismatchServer(_FakeServer):
    """A server whose registration plan does not match the worker's plan."""

    def retrieve(self, spec):
        with self.lock:
            self.retrieve_calls.append(
                (spec.resource_id, spec.start_token, spec.end_token)
            )
        return self._result(spec, "failed", adapter_mod.PLAN_MISMATCH), b""


class FakeTransport:
    def __init__(self, server: _FakeServer) -> None:
        self.server = server
        self.registered: list = []
        self.cancelled: list = []

    def _future(self, value):
        future = Future()
        if self.server.delay:
            threading.Timer(
                self.server.delay, future.set_result, args=(value,)
            ).start()
        else:
            future.set_result(value)
        return future

    def register_resource_blocks(self, registration):
        self.registered.append(registration)
        ack = adapter_mod.ResourceBlockAck(
            instance_id=registration.instance_id,
            registration_epoch=registration.registration_epoch,
            plan_hash=registration.plan_hash,
            layout_hash=registration.layout_hash,
            server_generation=0,
            capabilities=self.server.capabilities,
        )
        future = Future()
        future.set_result(ack)
        return future

    def store_resource_block(self, spec, payload):
        return self._future(self.server.store(spec, payload))

    def retrieve_resource_block(self, spec):
        return self._future(self.server.retrieve(spec))

    def lookup_resource_blocks(self, epoch, stock_key, limit):
        return self._future(
            adapter_mod.ResourceBlockLookupResult(prefix_end=0)
        )


class _KeyRecordingTransport:
    """Forward to a fake transport and record every spec's stock key.

    The adapter must pin each resource operation to the worker identity, so
    the test observes the keys the transport actually receives rather than
    the adapter's private bookkeeping.
    """

    def __init__(self, inner: FakeTransport) -> None:
        self.inner = inner
        self.store_keys: list = []
        self.retrieve_keys: list = []

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def store_resource_block(self, spec, payload):
        self.store_keys.append(spec.stock_key)
        return self.inner.store_resource_block(spec, payload)

    def retrieve_resource_block(self, spec):
        self.retrieve_keys.append(spec.stock_key)
        return self.inner.retrieve_resource_block(spec)


_BOUNDARY_IDS = {
    group.boundary_resource.resource_id
    for group in PLAN.groups
    if group.boundary_resource is not None
}


def _cache_names() -> list[str]:
    names = [f"unused-{index}" for index in range(135)]
    for resource_id in _resource_ids():
        index = PLAN_RESOURCES[resource_id].tensor_indices[0]
        names[index] = resource_id
    return names


PLAN_RESOURCES = {
    resource.resource_id: resource
    for group in PLAN.groups
    for resource in group.transfer_resources
}


def _resource_ids():
    return list(PLAN_RESOURCES)


def _make_caches(fill: float = 1.0, rows: int = 32):
    caches = {}
    for resource_id, resource in PLAN_RESOURCES.items():
        if resource.storage_kind == "boundary_state":
            continue
        caches[resource_id] = (
            torch.full((rows, 2, 3), fill, dtype=torch.float32).contiguous()
        )
    return caches


def _make_adapter(
    transport,
    *,
    caches=None,
    load_timeout=1.0,
    store_timeout=1.0,
    worker_id=0,
    world_size=WORLD,
    chunk_size=CHUNK_SIZE,
    block_size=BLOCK_SIZE,
):
    return adapter_mod.DSV41LwV2GroupAdapter(
        PLAN,
        transport=transport,
        instance_id="instance-1",
        model_name=MODEL,
        world_size=world_size,
        worker_id=worker_id,
        plan_hash=PLAN_HASH,
        layout_hash=LAYOUT_HASH,
        chunk_size=chunk_size,
        block_size=block_size,
        kv_caches=_make_caches() if caches is None else caches,
        cache_names=_cache_names(),
        load_timeout=load_timeout,
        store_timeout=store_timeout,
    )


def _seed_token_blocks(server: _FakeServer, request) -> None:
    """Pretend a previous store populated every token resource of the group."""
    for resource_id, resource in PLAN_RESOURCES.items():
        if resource.storage_kind != "token_range":
            continue
        plane = torch.full((32, 2, 3), 7.0, dtype=torch.float32).contiguous()
        spec = adapter_mod.ResourceBlockSpec(
            operation_id="seed",
            direction="load",
            instance_id="instance-1",
            request_id=request.request_id,
            step_id=request.step_id,
            registration_epoch=0,
            server_generation=0,
            plan_hash=PLAN_HASH,
            layout_hash=LAYOUT_HASH,
            stock_key=request.stock_key,
            resource_id=resource_id,
            engine_group_id=resource.engine_group_id,
            block_ids=(0, 1, 2, 3),
            start_token=0,
            end_token=TOKENS,
            prefix_end=TOKENS,
            skip_first_n_tokens=0,
        )
        for start in range(0, TOKENS, CHUNK_SIZE):
            end = start + CHUNK_SIZE
            blocks = request.block_ids_by_engine_group[0][
                start // BLOCK_SIZE : end // BLOCK_SIZE
            ]
            payload = adapter_mod.encode_token_payload(
                resource_id=resource_id,
                block_ids=blocks,
                planes=(plane[list(blocks)].clone(),),
            )
            server.blocks[server._token_key(spec, start, end)] = payload

    # G2/G8/G14 also require the boundary endpoint to be visible.
    for group_id in ("g2", "g8", "g14"):
        group = PLAN.group_by_id(group_id)
        if request.load_prefix_end != TOKENS:
            continue
        boundary_spec = adapter_mod.ResourceBlockSpec(
            operation_id="seed-boundary",
            direction="load",
            instance_id="instance-1",
            request_id=request.request_id,
            step_id=request.step_id,
            registration_epoch=0,
            server_generation=0,
            plan_hash=PLAN_HASH,
            layout_hash=LAYOUT_HASH,
            stock_key=request.stock_key,
            resource_id=group.boundary_resource.resource_id,
            engine_group_id=None,
            block_ids=(),
            start_token=TOKENS,
            end_token=TOKENS,
            prefix_end=TOKENS,
            skip_first_n_tokens=0,
        )
        server.boundaries[server._boundary_key(boundary_spec)] = (
            adapter_mod.BOUNDARY_PAYLOAD_MAGIC + b"seed"
        )


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------


def test_register_echoes_identity_and_requires_capabilities():
    server = _FakeServer()
    transport = FakeTransport(server)
    adapter = _make_adapter(transport)
    ack = adapter.register()
    assert ack.instance_id == "instance-1"
    assert ack.plan_hash == PLAN_HASH
    assert set(ack.capabilities) >= set(adapter_mod.REQUIRED_CAPABILITIES)

    bad = _FakeServer(capabilities=("resource_blocks_v1",))
    bad_adapter = _make_adapter(FakeTransport(bad))
    with pytest.raises(RuntimeError):
        bad_adapter.register()


def test_group_load_ready_writes_all_resources():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request()
    _seed_token_blocks(server, request)
    step = _step(request)

    adapter.schedule_group_retrieves("g2", ("r1",), step)
    adapter.wait_group("g2", ("r1",), step, stream=None)

    caches = adapter._kv_caches  # noqa: SLF001 - asserted cache binding
    for resource_id in (
        "model.layers.2.self_attn.long_kv_cache",
        "model.layers.2.self_attn.indexer.k_cache",
    ):
        rows = TOKENS // BLOCK_SIZE
        assert torch.allclose(
            caches[resource_id][:rows], torch.full((rows, 2, 3), 7.0)
        )
    assert adapter.metrics["group_load_ready"] == 1
    _, ready = adapter.take_finished()
    assert ready == {"r1"}


def test_missing_group_resource_fails_and_reports_receiving():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server), load_timeout=0.5)
    adapter.register()
    request = _request()
    _seed_token_blocks(server, request)
    # Drop the index scope for chunk 1: the whole group must fail closed.
    victim = "model.layers.2.self_attn.indexer.k_cache"
    for key in list(server.blocks):
        if key[5] == victim and key[6] == CHUNK_SIZE:
            del server.blocks[key]
    step = _step(request)
    adapter.schedule_group_retrieves("g2", ("r1",), step)
    with pytest.raises(connector_mod.DSV41GroupLoadError) as info:
        adapter.wait_group("g2", ("r1",), step, stream=None)
    assert info.value.group_id == "g2"
    assert info.value.error_code == adapter_mod.RESOURCE_MISSING
    assert adapter.take_failed_receiving() == {"r1"}


def test_plan_mismatch_propagates_to_failed_recving():
    """F5: a server plan mismatch is not rewritten into a generic failure."""
    server = _PlanMismatchServer()
    adapter = _make_adapter(FakeTransport(server), load_timeout=1.0)
    adapter.register()
    request = _request()
    step = _step(request)

    adapter.schedule_group_retrieves("g2", ("r1",), step)
    with pytest.raises(connector_mod.DSV41GroupLoadError) as info:
        adapter.wait_group("g2", ("r1",), step, stream=None)

    assert info.value.error_code == adapter_mod.PLAN_MISMATCH
    assert adapter.take_failed_receiving() == {"r1"}


def test_load_timeout_is_terminal():
    server = _FakeServer(delay=0.2)
    adapter = _make_adapter(FakeTransport(server), load_timeout=0.02)
    adapter.register()
    request = _request()
    _seed_token_blocks(server, request)
    step = _step(request)
    adapter.schedule_group_retrieves("g2", ("r1",), step)
    with pytest.raises(connector_mod.DSV41GroupLoadError) as info:
        adapter.wait_group("g2", ("r1",), step, stream=None)
    assert info.value.error_code == adapter_mod.TIMEOUT
    assert adapter.metrics["group_load_timeout"] >= 1


def test_wait_counts_host_fallback_when_pending():
    server = _FakeServer(delay=0.05)
    adapter = _make_adapter(FakeTransport(server), load_timeout=2.0)
    adapter.register()
    request = _request()
    _seed_token_blocks(server, request)
    step = _step(request)
    adapter.schedule_group_retrieves("g2", ("r1",), step)
    adapter.wait_group("g2", ("r1",), step, stream=None)
    assert adapter.metrics["host_wait_fallback"] >= 1


def test_drop_request_after_schedule_never_becomes_ready():
    server = _FakeServer(delay=0.05)
    adapter = _make_adapter(FakeTransport(server), load_timeout=1.0)
    adapter.register()
    request = _request()
    _seed_token_blocks(server, request)
    step = _step(request)
    adapter.schedule_group_retrieves("g2", ("r1",), step)
    adapter.drop_request("r1", reason="cancelled")
    with pytest.raises(connector_mod.DSV41GroupLoadError):
        adapter.wait_group("g2", ("r1",), step, stream=None)
    assert adapter.metrics["group_load_dropped"] == 1
    # A late callback must not resurrect the request as ready.
    assert adapter.take_finished()[1] == set()


def test_stale_generation_drops_old_load():
    server = _FakeServer(delay=0.05)
    adapter = _make_adapter(FakeTransport(server), load_timeout=1.0)
    adapter.register()
    request = _request()
    _seed_token_blocks(server, request)
    step = _step(request)
    adapter.schedule_group_retrieves("g2", ("r1",), step)
    adapter.set_generations(registration_epoch=1, server_generation=0)
    assert adapter.metrics["stale_ops_cleared"] >= 1
    with pytest.raises(connector_mod.DSV41GroupLoadError):
        adapter.wait_group("g2", ("r1",), step, stream=None)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def test_store_then_new_request_hits_same_prefix():
    server = _FakeServer()
    caches = _make_caches(fill=3.0)
    adapter = _make_adapter(FakeTransport(server), caches=caches)
    adapter.register()
    tokens = _token_bytes(TOKENS)
    store_request = _request(
        "r1", load_prefix_end=0, store_prefix_end=TOKENS, tokens=tokens
    )
    step = _step(store_request)
    # g20 has no boundary endpoint, so a token-only store is complete there.
    adapter.submit_group_store("g20", ("r1",), step)
    adapter.drain_stores()
    assert adapter.metrics["group_store_committed"] == 1
    assert server.store_calls
    assert adapter.take_finished()[0] == {"r1"}

    # A new request with the same prefix but a different suffix and id hits.
    fresh = _make_caches(fill=0.0)
    adapter.bind_kv_caches(fresh)
    load_request = _request("r2", load_prefix_end=TOKENS, tokens=tokens)
    load_step = _step(load_request)
    adapter.schedule_group_retrieves("g20", ("r2",), load_step)
    adapter.wait_group("g20", ("r2",), load_step, stream=None)
    rows = TOKENS // BLOCK_SIZE
    assert torch.allclose(
        fresh["model.layers.20.self_attn.long_kv_cache"][:rows],
        torch.full((rows, 2, 3), 3.0),
    )


def test_group_specs_pin_the_worker_identity():
    """TP>1: every store/retrieve spec addresses only this worker's rank."""
    server = _FakeServer()
    transport = _KeyRecordingTransport(FakeTransport(server))
    adapter = _make_adapter(transport, worker_id=3, world_size=8)
    try:
        tokens = _token_bytes(TOKENS)
        store_request = _request(
            "r1", load_prefix_end=0, store_prefix_end=TOKENS, tokens=tokens
        )
        store_step = _step(store_request)
        adapter.submit_group_store("g20", ("r1",), store_step)
        adapter.drain_stores()
        assert transport.store_keys
        for key in transport.store_keys:
            assert (key.worker_id, key.world_size) == (3, 8)

        load_request = _request("r2", load_prefix_end=TOKENS, tokens=tokens)
        load_step = _step(load_request)
        adapter.schedule_group_retrieves("g20", ("r2",), load_step)
        adapter.wait_group("g20", ("r2",), load_step, stream=None)
        assert transport.retrieve_keys
        for key in transport.retrieve_keys:
            assert (key.worker_id, key.world_size) == (3, 8)
    finally:
        adapter.close()


def test_default_adapter_keeps_the_tp1_worker_key():
    """TP=1 regression: the worker key is already single rank and unchanged."""
    server = _FakeServer()
    transport = _KeyRecordingTransport(FakeTransport(server))
    adapter = _make_adapter(transport)
    try:
        store_request = _request(
            "r1",
            load_prefix_end=0,
            store_prefix_end=TOKENS,
            tokens=_token_bytes(TOKENS),
        )
        step = _step(store_request)
        adapter.submit_group_store("g20", ("r1",), step)
        adapter.drain_stores()
        assert transport.store_keys
        for key in transport.store_keys:
            assert (key.worker_id, key.world_size) == (0, WORLD)
    finally:
        adapter.close()


def test_store_failure_is_dropped_not_committed():
    server = _FakeServer()
    transport = FakeTransport(server)

    def failing_store(spec, payload):
        raise OSError("boom")

    transport.store_resource_block = failing_store  # type: ignore[assignment]
    adapter = _make_adapter(transport)
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g2", ("r1",), _step(request))
    adapter.drain_stores()
    assert adapter.metrics["group_store_committed"] == 0
    assert adapter.metrics["group_store_failed"] >= 1
    assert adapter.take_finished()[0] == {"r1"}


# ---------------------------------------------------------------------------
# Boundary codec
# ---------------------------------------------------------------------------


@dataclass
class _BoundaryView:
    request_id: str
    resource_id: str
    prefix_end: int
    boundary_end: int
    step_id: int = 0
    registration_epoch: int = 0
    server_generation: int = 0
    request_index: int = 0
    dispatch_layer: str = "model.layers.2.self_attn.attn"
    execution_layer: str = "model.layers.2.self_attn.attn"
    step_tokens: int = 1
    ring_length: int = 4
    ring_page: int = 0
    next_slot: int = 3
    compression_ratio: int = 2
    compression_phase: int = 1
    position_control: tuple = (1, 2, 3, 4, 5)
    written_rows: tuple = (3,)


@dataclass
class _TensorView:
    ring: torch.Tensor
    page_ids: tuple
    ring_rows: int
    ring_head_size: int


def _boundary_view(request_id="r1", *, prefix_end=TOKENS - 1):
    return _BoundaryView(
        request_id=request_id,
        resource_id=PLAN.group_by_id("g2").boundary_resource.resource_id,
        prefix_end=prefix_end,
        boundary_end=prefix_end + 1,
        next_slot=prefix_end % 4,
    )


def _ring(pages=2, rows=4, head=2, fill=0.0):
    return torch.full((pages, rows, 2 * head), fill, dtype=torch.float32)


def _codec(adapter):
    return boundary_mod.DSV41LwV2BoundaryCodec(adapter)


def test_boundary_capture_store_retrieve_restore_with_page_relocation():
    server = _FakeServer()
    caches = _make_caches(fill=4.0)
    adapter = _make_adapter(FakeTransport(server), caches=caches)
    adapter.register()
    codec = _codec(adapter)

    capture_view = _boundary_view("r1", prefix_end=TOKENS - 1)
    capture_ring = _ring(fill=1.0)
    capture_ring[0, :, :2] = torch.arange(8).reshape(4, 2)
    capture_ring[0, :, 2:] = 100.0
    codec.capture_layer_boundary(
        capture_view,
        _TensorView(capture_ring, (0, 1), 4, 2),
    )
    assert adapter.metrics["boundary_captured"] == 1

    # Store the captured boundary (the source page is on ring page 0).
    store_request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g2", ("r1",), _step(store_request))
    adapter.drain_stores()
    assert adapter.metrics["group_store_committed"] == 1
    assert server.boundaries

    # A second request retrieves the same prefix and lives on page 1.
    caches = _make_caches(fill=0.0)
    adapter.bind_kv_caches(caches)
    load_request = _request("r2", load_prefix_end=TOKENS)
    adapter.schedule_group_retrieves("g2", ("r2",), _step(load_request))
    adapter.wait_group("g2", ("r2",), _step(load_request), stream=None)

    restore_ring = _ring(fill=-1.0)
    restore_view = _boundary_view("r2", prefix_end=TOKENS)
    codec.restore_layer_boundary(
        restore_view,
        _TensorView(restore_ring, (1, 0), 4, 2),
    )
    assert adapter.metrics["boundary_restored"] == 1
    assert torch.allclose(restore_ring[1, :, :2], capture_ring[0, :, :2])
    assert torch.allclose(restore_ring[1, :, 2:], capture_ring[0, :, 2:])
    # The capture page itself was not touched.
    assert torch.allclose(restore_ring[0], torch.full((4, 4), -1.0))


@pytest.mark.skipif(
    not VLLM_ASCEND.is_dir(), reason="vllm-ascend checkout is not present"
)
def test_real_engine_view_absolute_rows_cross_page_roundtrip():
    """F1: a real engine view on page >0 must survive the codec chain.

    The engine DTO emits *absolute* ring rows; the codec used to demand
    page-relative rows and therefore rejected every real capture/restore on a
    page other than zero.  This test uses the real vllm-ascend DTOs and its
    ``_written_rows`` kernel formula, then exercises the real v2 codec through
    capture -> store -> retrieve -> restore with different live pages.
    """
    engine = _load_engine_boundary()
    layer = "model.layers.2.self_attn.attn"
    resource_id = PLAN.group_by_id("g2").boundary_resource.resource_id
    capture_page, restore_page = 3, 7

    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server), caches=_make_caches(fill=4.0))
    adapter.register()
    codec = _codec(adapter)

    capture_ring = _ring(pages=8, fill=1.0)
    capture_ring[capture_page, :, :2] = torch.arange(8).reshape(4, 2)
    capture_ring[capture_page, :, 2:] = 100.0
    rows = engine._written_rows(  # noqa: SLF001 - real engine row formula
        ring_rows=4,
        prefix_end=TOKENS - 1,
        step_tokens=1,
        ring_page=capture_page,
    )
    assert rows == (15,)
    capture_view = engine.BoundaryRequestView(
        request_id="r1",
        request_index=0,
        step_id=0,
        registration_epoch=0,
        server_generation=0,
        resource_id=resource_id,
        dispatch_layer=layer,
        execution_layer=layer,
        prefix_end=TOKENS - 1,
        step_tokens=1,
        ring_length=4,
        ring_page=capture_page,
        next_slot=(TOKENS - 1) % 4,
        compression_ratio=2,
        compression_phase=(TOKENS - 1) % 2,
        position_control=(TOKENS - 1, 1, 0, 0, capture_page),
        written_rows=rows,
    )
    capture_tensor = engine.BoundaryTensorView(
        resource_id=resource_id,
        dispatch_layer=layer,
        execution_layer=layer,
        ring=capture_ring,
        page_ids=torch.tensor([capture_page], dtype=torch.int64),
        ring_rows=4,
        ring_head_size=2,
    )
    codec.capture_layer_boundary(capture_view, capture_tensor)
    assert adapter.metrics["boundary_captured"] == 1

    store_request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g2", ("r1",), _step(store_request))
    adapter.drain_stores()
    assert adapter.metrics["group_store_committed"] == 1

    # A second request hits the same prefix, but its live page is page 7.
    adapter.bind_kv_caches(_make_caches(fill=0.0))
    load_request = _request("r2", load_prefix_end=TOKENS)
    adapter.schedule_group_retrieves("g2", ("r2",), _step(load_request))
    adapter.wait_group("g2", ("r2",), _step(load_request), stream=None)

    restore_ring = _ring(pages=8, fill=-1.0)
    restore_rows = engine._written_rows(  # noqa: SLF001
        ring_rows=4,
        prefix_end=TOKENS,
        step_tokens=1,
        ring_page=restore_page,
    )
    assert restore_rows == (28,)
    restore_view = engine.BoundaryRequestView(
        request_id="r2",
        request_index=0,
        step_id=0,
        registration_epoch=0,
        server_generation=0,
        resource_id=resource_id,
        dispatch_layer=layer,
        execution_layer=layer,
        prefix_end=TOKENS,
        step_tokens=1,
        ring_length=4,
        ring_page=restore_page,
        next_slot=TOKENS % 4,
        compression_ratio=2,
        compression_phase=TOKENS % 2,
        position_control=(TOKENS, 1, 0, 0, restore_page),
        written_rows=restore_rows,
    )
    codec.restore_layer_boundary(
        restore_view,
        engine.BoundaryTensorView(
            resource_id=resource_id,
            dispatch_layer=layer,
            execution_layer=layer,
            ring=restore_ring,
            page_ids=torch.tensor([restore_page], dtype=torch.int64),
            ring_rows=4,
            ring_head_size=2,
        ),
    )
    assert adapter.metrics["boundary_restored"] == 1
    assert adapter.metrics["boundary_restore_failed"] == 0
    assert torch.allclose(
        restore_ring[restore_page, :, :2],
        capture_ring[capture_page, :, :2],
    )
    assert torch.allclose(
        restore_ring[restore_page, :, 2:],
        capture_ring[capture_page, :, 2:],
    )
    assert torch.allclose(
        restore_ring[restore_page - 1], torch.full((4, 4), -1.0)
    )


def test_boundary_restore_missing_payload_is_explicit_failure():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    codec = _codec(adapter)
    ring = _ring(fill=-1.0)
    view = _boundary_view("r9", prefix_end=TOKENS)
    with pytest.raises(boundary_mod.DSV41BoundaryRestoreError) as info:
        codec.restore_layer_boundary(view, _TensorView(ring, (0, 1), 4, 2))
    assert info.value.error_code == boundary_mod.BOUNDARY_PAYLOAD_MISSING
    assert adapter.metrics["boundary_restore_failed"] == 1


def test_boundary_schema1_and_tamper_are_rejected():
    header = {"schema": 1, "resource_id": "x"}
    with pytest.raises(boundary_mod.DSV41BoundaryStateError):
        boundary_mod.decode_boundary_schema2(b"DSV2BC1\n" + b"\x00" * 64)
    # A schema-2 payload with a flipped byte fails the digest.
    payload = boundary_mod.encode_boundary_schema2(
        {"resource_id": "x", "prefix_end": 8},
        torch.zeros(4, 2),
        torch.zeros(4, 2),
    )
    tampered = bytearray(payload)
    tampered[len(tampered) // 2] ^= 0xFF
    with pytest.raises(boundary_mod.DSV41BoundaryStateError) as info:
        boundary_mod.decode_boundary_schema2(bytes(tampered))
    assert info.value.error_code in (
        boundary_mod.BOUNDARY_PAYLOAD_CORRUPT,
        boundary_mod.BOUNDARY_GEOMETRY_MISMATCH,
    )


def test_boundary_wrong_prefix_on_restore_is_rejected():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    codec = _codec(adapter)
    capture_view = _boundary_view("r1", prefix_end=TOKENS - 1)
    codec.capture_layer_boundary(
        capture_view, _TensorView(_ring(fill=1.0), (0, 1), 4, 2)
    )
    # Make the staged payload available under a *different* prefix and try to
    # restore it as if it were valid for that prefix.
    payload = b""
    adapter._boundary_payloads[  # noqa: SLF001 - injects a retrieved payload
        ("r1", capture_view.resource_id, 4)
    ] = boundary_mod.encode_boundary_schema2(
        {
            "resource_id": capture_view.resource_id,
            "request_id": "r1",
            "dispatch_layer": capture_view.dispatch_layer,
            "execution_layer": capture_view.execution_layer,
            "prefix_end": TOKENS - 1,
            "boundary_end": TOKENS,
            "step_tokens": 1,
            "ring_length": 4,
            "next_slot": 3,
            "compression_ratio": 2,
            "compression_phase": 1,
            "position_control": [1, 2, 3, 4, 5],
            "written_rows": [3],
        },
        torch.zeros(4, 2),
        torch.zeros(4, 2),
    )
    restore_view = _boundary_view("r1", prefix_end=4)
    with pytest.raises(boundary_mod.DSV41BoundaryStateError):
        codec.restore_layer_boundary(
            restore_view, _TensorView(_ring(), (0, 1), 4, 2)
        )
    assert adapter.metrics["boundary_restore_failed"] == 1


def test_wire_dtos_are_shared_server_types_and_round_trip():
    """F1: the adapter must not define its own copies of the wire DTOs.

    A typed ``msgspec`` decode produces the server dataclass; if the adapter
    declared a second ``ResourceBlockSpec`` the subsequent ``isinstance``
    check failed with INVALID_SPEC on every real server reply.
    """
    # First Party
    import msgspec
    from lmcache.v1.multiprocess import resource_block_protocol as wire

    assert adapter_mod.ResourceBlockSpec is wire.ResourceBlockSpec
    assert adapter_mod.ResourceBlockRegistration is wire.ResourceBlockRegistration
    assert adapter_mod.ResourceBlockAck is wire.ResourceBlockAck

    spec = adapter_mod.ResourceBlockSpec(
        operation_id="op-1",
        direction="load",
        instance_id="instance-1",
        request_id="r1",
        step_id=3,
        registration_epoch=1,
        server_generation=2,
        plan_hash=PLAN_HASH,
        layout_hash=LAYOUT_HASH,
        stock_key=_stock_key("r1", _token_bytes(TOKENS)),
        resource_id=next(iter(PLAN_RESOURCES)),
        engine_group_id=0,
        block_ids=(0, 1),
        start_token=0,
        end_token=4,
        prefix_end=4,
        skip_first_n_tokens=0,
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(spec), type=wire.ResourceBlockSpec
    )
    assert type(decoded) is wire.ResourceBlockSpec
    assert isinstance(decoded, adapter_mod.ResourceBlockSpec)
    assert decoded == spec


# ---------------------------------------------------------------------------
# Per-step store staging (t21 root cause 1)
# ---------------------------------------------------------------------------

STREAMING_CHUNK = 256
STREAMING_BLOCK = 128
STREAMING_WINDOWS = (
    (0, 4096, tuple(range(0, 32))),
    (4096, 8192, tuple(range(32, 64))),
    (8192, 9472, tuple(range(64, 74))),
)


def _streaming_adapter(monkeypatch):
    server = _FakeServer()
    adapter = _make_adapter(
        FakeTransport(server),
        caches=_make_caches(rows=80),
        chunk_size=STREAMING_CHUNK,
        block_size=STREAMING_BLOCK,
    )
    adapter.register()
    warnings: list[str] = []
    monkeypatch.setattr(
        adapter_mod,
        "logger_warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )
    return adapter, server, warnings


def _submit_store_window(adapter, *, step_id, start, end, blocks):
    request = _request(
        "r1",
        step_id=step_id,
        load_prefix_end=0,
        store_prefix_end=end,
        store_start=start,
        blocks=(blocks,),
        tokens=_token_bytes(9690),
    )
    adapter.submit_group_store("g2", ("r1",), _step(request))
    adapter.drain_stores()


def test_store_staging_uses_each_steps_relative_block_list(monkeypatch):
    """The scheduler sends only this step's blocks (32/32/10 for 9690 tokens).

    Staging must map those blocks relative to ``store_start``; using the
    cumulative ``store_prefix_end`` raised "too few blocks" on windows 2/3 and
    silently dropped the whole group.
    """
    adapter, server, warnings = _streaming_adapter(monkeypatch)
    for step_id, (start, end, blocks) in enumerate(STREAMING_WINDOWS):
        _submit_store_window(
            adapter, step_id=step_id, start=start, end=end, blocks=blocks
        )

    long_kv_id = next(
        resource.resource_id
        for resource in PLAN.group_by_id("g2").token_range_resources
        if resource.resource_id.endswith(".long_kv_cache")
    )
    windows = sorted(
        (start, end)
        for resource_id, start, end in server.store_calls
        if resource_id == long_kv_id
    )
    expected = [
        (start, min(start + STREAMING_CHUNK, end))
        for start, end in ((0, 4096), (4096, 8192), (8192, 9472))
        for start in range(start, end, STREAMING_CHUNK)
    ]
    assert windows == expected
    # The unaligned tail (9472..9690) is never submitted.
    assert max(end for _, end in windows) == 9472

    first_blocks = {
        spec.start_token: spec.block_ids
        for spec in server.store_specs
        if spec.resource_id == long_kv_id
    }
    assert first_blocks[0] == (0, 1)
    assert first_blocks[4096] == (32, 33)
    assert first_blocks[8192] == (64, 65)

    assert adapter.metrics["group_store_dropped"] == 0
    assert adapter.metrics["group_store_committed"] == 3
    assert warnings == []


def test_store_staging_is_a_noop_when_the_step_adds_no_blocks(monkeypatch):
    adapter, server, warnings = _streaming_adapter(monkeypatch)
    _submit_store_window(
        adapter, step_id=0, start=9472, end=9472, blocks=()
    )

    assert server.store_calls == []
    assert adapter.metrics["group_store_dropped"] == 0
    assert adapter.metrics["group_store_committed"] == 0
    assert warnings == []


def test_store_staging_reports_short_block_windows(monkeypatch):
    adapter, server, warnings = _streaming_adapter(monkeypatch)
    _submit_store_window(
        adapter, step_id=0, start=0, end=4096, blocks=tuple(range(31))
    )

    assert server.store_calls == []
    assert adapter.metrics["group_store_dropped"] == 1
    assert len(warnings) == 1
    assert "too few blocks" in warnings[0]


def test_store_staging_keeps_load_and_store_endpoints_independent(monkeypatch):
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request(
        load_prefix_end=TOKENS, store_prefix_end=TOKENS, store_start=0
    )
    _seed_token_blocks(server, request)
    step = _step(request)

    adapter.schedule_group_retrieves("g2", ("r1",), step)
    adapter.wait_group("g2", ("r1",), step)
    assert adapter.metrics["group_load_ready"] == 1

    adapter.submit_group_store("g2", ("r1",), step)
    adapter.drain_stores()
    assert adapter.metrics["group_store_dropped"] == 0
    assert adapter.metrics["group_store_committed"] == 1
