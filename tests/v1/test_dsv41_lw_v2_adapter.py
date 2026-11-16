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
import json
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
        self.stock_lookup_result: tuple[int | None, bool] = (0, False)
        self.stock_finished_sending: set[str] = set()
        self.stock_finished_receiving: set[str] = set()
        self.stock_failed_recving: set[str] = set()
        self.meta_calls: list[object] = []

    def register_kv_caches(self, kv_caches) -> None:
        # The stock base registers layouts with the LMCache server; the
        # CPU stub only records the caches the v2 factory reads back.
        self.kv_caches = dict(kv_caches)
        _StubConnector.instances.append(self)

    def build_connector_meta(self, scheduler_output):
        self.meta_calls.append(scheduler_output)
        return getattr(scheduler_output, "kv_connector_metadata", None)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # Stock answer shape; the v2 connector converts a hit to a sync load.
        return self.stock_lookup_result

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
    _load_module(PACKAGE + ".mask_trace", V2_DIR / "mask_trace.py")
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
    max_queue=1024,
    async_store=None,
    store_retries=None,
    store_backpressure_timeout=None,
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
        max_queue=max_queue,
        async_store=async_store,
        store_retries=store_retries,
        store_backpressure_timeout=store_backpressure_timeout,
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
    assert adapter.metrics["boundary_staged"] == 1
    assert adapter.metrics["boundary_captured"] == 0

    # Store the captured boundary (the source page is on ring page 0).
    store_request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g2", ("r1",), _step(store_request))
    adapter.drain_stores()
    assert adapter.metrics["group_store_committed"] == 1
    assert adapter.metrics["boundary_captured"] == 1
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
    assert adapter.metrics["boundary_staged"] == 1
    assert adapter.metrics["boundary_captured"] == 0

    store_request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g2", ("r1",), _step(store_request))
    adapter.drain_stores()
    assert adapter.metrics["group_store_committed"] == 1
    assert adapter.metrics["boundary_captured"] == 1

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


def test_unaligned_boundary_capture_is_discarded_and_never_leaks(monkeypatch):
    """T25-1: a step whose store endpoint is chunk-clamped must degrade safely.

    The final B01 step captures the boundary at 9690 but only stores tokens up
    to the chunk-aligned endpoint 9472.  Attaching the 9690 page to 9472 would
    mislabel its compressor state, so the capture is discarded explicitly and
    the previous 8192 endpoint remains the warm-hit boundary.
    """
    adapter, server, warnings = _streaming_adapter(monkeypatch)
    resource_id = PLAN.group_by_id("g2").boundary_resource.resource_id
    codec = _codec(adapter)

    # Step 0 ends exactly on 8192: this aligned capture is attached.
    aligned = _boundary_view("r1", prefix_end=8191)
    aligned.ring_page = 8191 // 4
    aligned.next_slot = 8191 % 4
    aligned.written_rows = (8191,)
    codec.capture_layer_boundary(
        aligned,
        _TensorView(_ring(pages=2049), (aligned.ring_page,), 4, 2),
    )
    _submit_store_window(
        adapter, step_id=0, start=0, end=8192, blocks=tuple(range(64))
    )
    assert adapter.metrics["boundary_captured"] == 1
    boundary_keys = list(server.boundaries)
    assert len(boundary_keys) == 1
    assert boundary_keys[0][:4] == (PLAN_HASH, LAYOUT_HASH, resource_id, 8192)

    # The final step's live prefix is 9690 while only 9472 is chunk aligned.
    unaligned = _boundary_view("r1", prefix_end=9689)
    unaligned.ring_page = 9689 // 4
    unaligned.next_slot = 9689 % 4
    unaligned.written_rows = (9689,)
    codec.capture_layer_boundary(
        unaligned,
        _TensorView(_ring(pages=2423), (unaligned.ring_page,), 4, 2),
    )
    slot = adapter._boundary_stage[("r1", resource_id)]  # noqa: SLF001
    capture = slot[9690]
    assert capture.page is not None
    assert adapter.metrics["boundary_staged"] == 2

    _submit_store_window(
        adapter, step_id=1, start=8192, end=9472, blocks=tuple(range(64, 74))
    )

    # The unaligned page is newer than this window's endpoint: it is kept
    # staged for the window it actually belongs to, never attached to 9472.
    assert adapter.metrics["boundary_capture_discarded"] == 0
    assert adapter.metrics["boundary_deferred"] == 1
    assert adapter.metrics["boundary_attached"] == 1
    assert adapter.metrics["boundary_captured"] == 1
    assert capture.released is False
    assert capture.encoded is not None
    # The server still exposes only the last complete, attached endpoint.
    boundary_keys = list(server.boundaries)
    assert len(boundary_keys) == 1
    assert boundary_keys[0][:4] == (PLAN_HASH, LAYOUT_HASH, resource_id, 8192)

    # When the request finishes, the deferred page is released, not leaked.
    adapter._sending_ready.add("r1")  # noqa: SLF001
    adapter.take_finished()
    assert adapter._boundary_stage == {}  # noqa: SLF001
    assert capture.page is None
    assert capture.encoded is None
    assert capture.released is True
    assert warnings == []


def test_drop_request_releases_staged_boundary_capture():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    resource_id = PLAN.group_by_id("g2").boundary_resource.resource_id
    codec = _codec(adapter)
    codec.capture_layer_boundary(
        _boundary_view("r1", prefix_end=TOKENS - 1),
        _TensorView(_ring(fill=1.0), (0, 1), 4, 2),
    )
    capture = next(
        iter(adapter._boundary_stage[("r1", resource_id)].values())  # noqa: SLF001
    )
    assert capture.page is not None

    adapter.drop_request("r1", reason="dropped")

    assert adapter._boundary_stage == {}  # noqa: SLF001
    assert capture.page is None
    assert adapter.metrics["boundary_stage_released"] == 1


def test_replacing_a_boundary_capture_releases_the_previous_page():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    resource_id = PLAN.group_by_id("g2").boundary_resource.resource_id
    codec = _codec(adapter)
    codec.capture_layer_boundary(
        _boundary_view("r1", prefix_end=TOKENS - 1),
        _TensorView(_ring(fill=1.0), (0, 1), 4, 2),
    )
    first = next(
        iter(adapter._boundary_stage[("r1", resource_id)].values())  # noqa: SLF001
    )

    # Same endpoint => the newer page replaces the older one.
    second = _boundary_view("r1", prefix_end=TOKENS + 3)
    second.boundary_end = TOKENS
    second.ring_page = 2
    second.written_rows = (11,)
    codec.capture_layer_boundary(
        second,
        _TensorView(_ring(pages=3, fill=2.0), (2,), 4, 2),
    )

    assert first.page is None
    assert adapter.metrics["boundary_stage_released"] == 1
    assert len(adapter._boundary_stage) == 1  # noqa: SLF001


def test_finished_requests_release_any_staged_boundary_capture():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    resource_id = PLAN.group_by_id("g2").boundary_resource.resource_id
    codec = _codec(adapter)
    codec.capture_layer_boundary(
        _boundary_view("r1", prefix_end=TOKENS - 1),
        _TensorView(_ring(fill=1.0), (0, 1), 4, 2),
    )
    capture = next(
        iter(adapter._boundary_stage[("r1", resource_id)].values())  # noqa: SLF001
    )
    adapter._sending_ready.add("r1")  # noqa: SLF001

    finished_sending, finished_receiving = adapter.take_finished()

    assert finished_sending == {"r1"}
    assert finished_receiving == set()
    assert capture.page is None
    assert adapter._boundary_stage == {}  # noqa: SLF001
    assert adapter.metrics["boundary_stage_released"] == 1


def test_finished_requests_release_load_group_bookkeeping():
    adapter = _make_adapter(FakeTransport(_FakeServer()))
    adapter._expected_load_groups["r1"] = {"g2", "g8"}  # noqa: SLF001
    adapter._completed_load_groups["r1"] = {"g2", "g8"}  # noqa: SLF001
    adapter._receiving_ready.add("r1")  # noqa: SLF001

    sending, receiving = adapter.take_finished(finished_req_ids={"r1"})

    assert sending == set()
    assert receiving == {"r1"}
    # Load bookkeeping must not grow with every request the connector
    # finishes: the v2 connector consumes load completions in the forward and
    # only reports stores.
    assert "r1" not in adapter._expected_load_groups  # noqa: SLF001
    assert "r1" not in adapter._completed_load_groups  # noqa: SLF001


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


class _LiveBoundaryHook:
    """A stand-in for the T3 engine hook that captures the live ring page."""

    def __init__(self, codec, ring: torch.Tensor) -> None:
        self.codec = codec
        self.ring = ring
        self.prefix = 0
        self.calls: list[str] = []

    def capture_group_boundary(self, group_id: str) -> None:
        self.calls.append(group_id)
        if group_id != "g2":
            return
        prefix = self.prefix
        if prefix <= 0 or prefix % CHUNK_SIZE:
            return
        view = _BoundaryView(
            request_id="r1",
            resource_id=PLAN.group_by_id("g2").boundary_resource.resource_id,
            prefix_end=prefix - 1,
            boundary_end=prefix,
            next_slot=(prefix - 1) % 4,
            ring_page=(prefix - 1) // 4,
            written_rows=((prefix - 1),),
            step_id=prefix,
        )
        self.codec.capture_layer_boundary(
            view,
            _TensorView(self.ring, (view.ring_page,), 4, 2),
        )

    def restore_group_boundary(self, group_id: str) -> None:
        raise AssertionError(f"unexpected boundary restore for {group_id}")


def test_post_forward_ordering_stores_each_window_and_defers_boundary():
    """vLLM 0.30 calls start_load_kv *after* the forward for cold requests.

    The routing step is therefore created post-forward.  Stores must be
    submitted from ``wait_for_save`` (same step, KV rows still live) instead of
    waiting for a later forward, and a boundary captured one step ahead must be
    deferred to its own window rather than dropped or mislabelled.
    """
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    codec = _codec(adapter)
    ring = _ring(pages=8, fill=1.0)
    hook = _LiveBoundaryHook(codec, ring)
    router = connector_mod.GroupTransferRouter(
        PLAN, adapter=adapter, boundary_hook=hook
    )

    def forward(live_prefix: int) -> None:
        hook.prefix = live_prefix
        router.wait_group_load("g2")
        router.restore_group_boundary("g2")
        router.capture_group_boundary("g2")
        router.submit_group_store("g2")

    def post_forward(request) -> None:
        # ``start_load_kv`` (flush + note) runs after the forward; the
        # connector's ``wait_for_save`` then submits the new step's stores.
        router.flush_step()
        router.note_step(_step(request))
        router.wait_for_save()

    # r1 step 0: 0..4 computed, live prefix 4.
    forward(4)
    post_forward(
        _request(
            "r1", step_id=0, load_prefix_end=0, store_prefix_end=4,
            store_start=0,
        )
    )
    assert len(server.blocks) == 8
    # The prefill forward captures at its own endpoint (4) before vLLM 0.30
    # builds the store step; the post-forward submit attaches it to window
    # [0, 4) instead of dropping it.
    assert len(server.boundaries) == 1
    assert next(iter(server.boundaries))[3] == 4
    assert adapter.metrics["boundary_captured"] == 1
    assert adapter.metrics["boundary_attached"] == 1

    # r1 step 1: 4..8 computed, live prefix 8 => boundary endpoint 8.
    forward(8)
    post_forward(
        _request(
            "r1", step_id=1, load_prefix_end=0, store_prefix_end=8,
            store_start=4,
        )
    )
    assert len(server.blocks) == 16
    boundary_keys = list(server.boundaries)
    assert len(boundary_keys) == 2
    assert sorted(key[3] for key in boundary_keys) == [4, 8]
    assert adapter.metrics["boundary_captured"] == 2
    assert adapter.metrics["boundary_attached"] == 2

    # r2 is a single-step request: its stores must be submitted without any
    # further forward (a decode step served from a CUDA graph has no hooks).
    forward(8)
    post_forward(
        _request(
            "r2", step_id=2, load_prefix_end=0, store_prefix_end=8,
            store_start=0,
        )
    )
    r2_specs = [spec for spec in server.store_specs if spec.request_id == "r2"]
    assert len(r2_specs) == 16
    # r2 shares r1's prefix, so the same stock block keys are rewritten.
    assert len(server.blocks) == 16
    assert router.metrics["steps_with_store"] == 3
    # r1's second forward already captured endpoint 8, so r2 rewrites tokens
    # only; no further aligned prefill forward exists to capture a new page.
    assert adapter.metrics["boundary_captured"] == 2
    assert adapter.metrics["boundary_attached"] == 2
    assert adapter.metrics["group_store_dropped"] == 0
    assert adapter.metrics["worker_failures"] == 0


def test_two_cold_requests_store_every_window_at_production_sizes():
    """t37: a 9690-token request then a 543-token request both submit stores.

    Mirrors the B01 shape: chunk 256, block 128, twelve block groups, three
    prefill windows for the long request and one 512-token window for the
    short one.  Both requests must produce concrete store specs; nothing may
    be silently dropped at the submit/stage boundary.
    """
    server = _FakeServer()
    transport = FakeTransport(server)
    caches = _make_caches(rows=80)
    adapter = _make_adapter(
        transport,
        caches=caches,
        chunk_size=256,
        block_size=128,
        load_timeout=2.0,
        store_timeout=2.0,
    )
    router = connector_mod.GroupTransferRouter(PLAN)
    router.bind_adapter(adapter)

    long_blocks = tuple(tuple(range(76)) for _ in range(12))
    short_blocks = tuple(
        tuple(range(5)) if index == 0 else (0,) for index in range(12)
    )
    long_tokens = _token_bytes(9690)
    short_tokens = _token_bytes(543)
    windows = (
        ("r1", 0, 0, 4096),
        ("r1", 1, 4096, 8192),
        ("r1", 2, 8192, 9472),
        ("r2", 3, 0, 512),
    )
    for request_id, step_id, start, end in windows:
        request = _request(
            request_id,
            step_id=step_id,
            load_prefix_end=0,
            store_prefix_end=end,
            store_start=start,
            blocks=long_blocks if request_id == "r1" else short_blocks,
            tokens=long_tokens if request_id == "r1" else short_tokens,
        )
        router.note_step(_step(request))
        router.wait_for_save()

    adapter.drain_stores()

    r1_specs = [spec for spec in server.store_specs if spec.request_id == "r1"]
    r2_specs = [spec for spec in server.store_specs if spec.request_id == "r2"]
    assert len(r1_specs) == 8 * 37
    assert len(r2_specs) == 8 * 2
    assert {spec.start_token for spec in r2_specs} == {0, 256}
    assert adapter.metrics["group_store_dropped"] == 0
    assert adapter.metrics["store_stage_empty"] == 0
    assert adapter.metrics["worker_failures"] == 0
    assert adapter.metrics["group_store_committed"] == 16


class _RecordingIndexTarget:
    """Target stand-in that records the device of the copy arguments."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.index: torch.Tensor | None = None
        self.value: torch.Tensor | None = None

    def index_copy_(self, dim, index, value):
        assert dim == 0
        self.index = index
        self.value = value
        return self


def test_non_cuda_index_copy_moves_the_cpu_index_to_the_target_device():
    target = _RecordingIndexTarget(torch.device("meta"))
    index = torch.tensor([1, 3], dtype=torch.long)
    value = torch.zeros(2, 4)

    assert adapter_mod._copy_rows(target, index, value) is None
    assert target.index is not None
    assert target.index.device.type == "meta"
    assert target.value is not None
    assert target.value.device.type == "meta"


def test_retrieve_applies_rows_through_the_device_normalising_helper():
    import inspect

    source = inspect.getsource(adapter_mod.DSV41LwV2GroupAdapter._apply_retrieve)
    assert "_copy_rows(" in source
    assert "index_copy_" not in source


def test_token_payload_round_trips_int8_planes():
    plane = torch.arange(-8, 8, dtype=torch.int8).reshape(4, 4)
    payload = adapter_mod.encode_token_payload(
        resource_id="language_model.model.layers.2.self_attn.indexer.k_cache",
        block_ids=(0, 1),
        planes=(plane,),
    )
    (decoded,) = adapter_mod.decode_token_payload(payload, resource_id=None)
    assert decoded.dtype == torch.int8
    assert torch.equal(decoded, plane)


def test_mask_trace_is_default_off_and_joins_load_wait_store_events(
    tmp_path, monkeypatch
):
    """t66: the mask timeline is inert unless opted in, then covers both halves.

    Off: the same group load produces no file at all.  On: one JSONL stream
    carries ``load_start``/``load_ready``/``wait_enter``/``wait_exit`` and the
    store half (``store_submit``/``store_done``) for the long and the short
    request, with monotonic timestamps and joinable fields.
    """
    trace = sys.modules[PACKAGE + ".mask_trace"]
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    monkeypatch.delenv(trace.ENV_ENABLED, raising=False)
    monkeypatch.setenv(trace.ENV_DIR, str(off_dir))
    trace.reset_state()

    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request()
    # Scheduling already passes every emit point; the wait would need seeded
    # payloads and is covered by the enabled half below.
    adapter.schedule_group_retrieves("g20", ("r1",), _step(request))

    assert list(off_dir.iterdir()) == []
    assert trace.stats()["lines"] == 0

    on_dir = tmp_path / "on"
    on_dir.mkdir()
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(on_dir))
    trace.reset_state()

    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    router = connector_mod.GroupTransferRouter(PLAN)
    router.bind_adapter(adapter)

    loaded = _request("r1", step_id=0, load_prefix_end=TOKENS)
    _seed_token_blocks(server, loaded)
    router.note_step(_step(loaded))
    router.wait_group_load("g20")

    stored = _request(
        "r1", step_id=1, load_prefix_end=0, store_prefix_end=TOKENS
    )
    router.note_step(_step(stored))
    router.wait_for_save()
    short = _request("r2", step_id=2, load_prefix_end=0, store_prefix_end=TOKENS)
    router.note_step(_step(short))
    router.wait_for_save()

    files = list(on_dir.glob("mask-trace-*.jsonl"))
    assert len(files) == 1
    records = [
        __import__("json").loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]
    events = [record["event"] for record in records]
    assert "load_start" in events
    assert "load_ready" in events
    assert "wait_enter" in events
    assert "wait_exit" in events
    assert "store_submit" in events
    assert "store_done" in events
    assert [record["ts_ns"] for record in records] == sorted(
        record["ts_ns"] for record in records
    )
    assert [record["seq"] for record in records] == list(range(len(records)))
    submit_requests = {
        record["request_id"]
        for record in records
        if record["event"] == "store_submit"
    }
    assert submit_requests == {"r1", "r2"}
    for record in records:
        assert record["pid"]
        if record["event"] in ("load_start", "load_ready"):
            assert record["step_id"] == 0
            assert record["group_id"].startswith("g")
    assert "g20" in {
        record["group_id"]
        for record in records
        if record["event"] in ("load_start", "load_ready")
    }
    assert trace.stats()["truncated_requests"] == 0
    assert trace.stats()["lines"] == len(records)


def test_mask_trace_covers_the_production_9690_and_543_store_windows(
    tmp_path, monkeypatch
):
    """t66: the timeline covers both B01 request shapes end to end.

    The long request stores 37 chunks (9690 tokens -> windows ending at 4096,
    8192 and 9472) and the short request stores 2 chunks (543 tokens -> 512).
    Every resource of every window must produce a ``store_submit`` and a
    terminal ``store_done`` in the same stream.
    """
    trace = sys.modules[PACKAGE + ".mask_trace"]
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()

    server = _FakeServer()
    adapter = _make_adapter(
        FakeTransport(server),
        caches=_make_caches(rows=80),
        chunk_size=256,
        block_size=128,
        load_timeout=2.0,
        store_timeout=2.0,
    )
    router = connector_mod.GroupTransferRouter(PLAN)
    router.bind_adapter(adapter)

    long_blocks = tuple(tuple(range(76)) for _ in range(12))
    short_blocks = tuple(
        tuple(range(5)) if index == 0 else (0,) for index in range(12)
    )
    long_tokens = _token_bytes(9690)
    short_tokens = _token_bytes(543)
    windows = (
        ("r1", 0, 0, 4096),
        ("r1", 1, 4096, 8192),
        ("r1", 2, 8192, 9472),
        ("r2", 3, 0, 512),
    )
    for request_id, step_id, start, end in windows:
        request = _request(
            request_id,
            step_id=step_id,
            load_prefix_end=0,
            store_prefix_end=end,
            store_start=start,
            blocks=long_blocks if request_id == "r1" else short_blocks,
            tokens=long_tokens if request_id == "r1" else short_tokens,
        )
        router.note_step(_step(request))
        router.wait_for_save()

    adapter.drain_stores()
    files = list(tmp_path.glob("mask-trace-*.jsonl"))
    assert len(files) == 1
    records = [
        __import__("json").loads(line)
        for line in files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]
    submits = [
        record for record in records if record["event"] == "store_submit"
    ]
    r1_ends = {
        record["store_prefix_end"]
        for record in submits
        if record["request_id"] == "r1"
    }
    r2_ends = {
        record["store_prefix_end"]
        for record in submits
        if record["request_id"] == "r2"
    }
    assert r1_ends == {4096, 8192, 9472}
    assert r2_ends == {512}
    assert all(
        record["specs"] == 4
        for record in submits
        if record["request_id"] == "r2"
    )
    done = [record for record in records if record["event"] == "store_done"]
    assert len(done) == 8 * 37 + 8 * 2
    assert trace.stats()["truncated_requests"] == 0


# ---------------------------------------------------------------------------
# t71: async store (off-thread extraction, backpressure, retry, read ownership)
# ---------------------------------------------------------------------------


class _RetryOnceTransport(FakeTransport):
    """First store attempt fails with a retryable IO error."""

    def __init__(self, server: _FakeServer) -> None:
        super().__init__(server)
        self.failed_once = False

    def store_resource_block(self, spec, payload):
        if not self.failed_once:
            self.failed_once = True
            future = Future()
            future.set_result(
                adapter_mod.ResourceBlockResult(
                    operation_id=spec.operation_id,
                    direction=spec.direction,
                    resource_id=spec.resource_id,
                    registration_epoch=spec.registration_epoch,
                    server_generation=spec.server_generation,
                    status="failed",
                    error_code="IO_ERROR",
                )
            )
            return future
        return super().store_resource_block(spec, payload)


def test_async_store_extracts_on_the_worker_thread():
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    step = _step(request)

    adapter.submit_group_store("g2", ("r1",), step)
    adapter.drain_stores()

    metrics = adapter.metrics
    assert adapter.async_store is True
    assert metrics["store_extract_worker_thread"] >= 1
    assert metrics["store_extract_engine_thread"] == 0
    assert metrics["group_store_committed"] >= 1
    assert server.store_specs


def test_async_store_can_be_disabled_for_the_sync_fallback(monkeypatch):
    monkeypatch.setenv("DSV41_LW_V3_ASYNC_STORE", "0")
    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    step = _step(request)

    adapter.submit_group_store("g2", ("r1",), step)
    adapter.drain_stores()

    metrics = adapter.metrics
    assert adapter.async_store is False
    assert metrics["store_extract_engine_thread"] >= 1
    assert metrics["store_extract_worker_thread"] == 0
    assert server.store_specs


def test_store_backpressure_fails_closed_when_the_queue_stays_full():
    adapter = _make_adapter(
        FakeTransport(_FakeServer()),
        max_queue=1,
        store_backpressure_timeout=0.01,
    )
    # Swap in a queue with no consumer so the bounded put is deterministic.
    adapter._jobs = adapter_mod.queue.Queue(maxsize=1)  # noqa: SLF001
    adapter._jobs.put(("noop", None))  # noqa: SLF001

    queued = adapter._enqueue(("store", None), blocking=True)  # noqa: SLF001

    assert queued is False
    assert adapter.metrics["queue_full"] == 1


def test_store_retries_once_after_a_retryable_io_error():
    server = _FakeServer()
    transport = _RetryOnceTransport(server)
    adapter = _make_adapter(transport, store_retries=1)
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    step = _step(request)

    adapter.submit_group_store("g2", ("r1",), step)
    adapter.drain_stores()

    metrics = adapter.metrics
    assert metrics["store_retries"] >= 1
    assert metrics["group_store_committed"] >= 1
    assert metrics["group_store_failed"] == 0


def test_terminal_store_batch_keeps_source_read_ownership_until_worker_stops():
    adapter = _make_adapter(FakeTransport(_FakeServer()))
    event = adapter_mod.GroupEvent("g2", "r1", "store")
    batch = adapter_mod._StoreBatch(event=event)
    adapter._store_batches.append(batch)  # noqa: SLF001
    event.drop(reason="unit")

    adapter.drain_stores(wait=False)

    # Terminal status does not prove the worker stopped reading: the batch
    # must stay discoverable and flush_source_reads must report it unsafe.
    assert batch in adapter._store_batches  # noqa: SLF001
    assert adapter.flush_source_reads({"r1"}, timeout=0.01) == {"r1"}

    batch.read_done.set()
    adapter.drain_stores(wait=False)
    assert batch not in adapter._store_batches  # noqa: SLF001
    assert adapter.flush_source_reads({"r1"}, timeout=0.01) == set()

# ---------------------------------------------------------------------------
# t78: async-store C-component instrumentation
# ---------------------------------------------------------------------------


def _trace_records(directory: Path) -> list[dict]:
    records: list[dict] = []
    for trace_path in sorted(directory.glob("mask-trace-*.jsonl")):
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def test_async_store_trace_emits_describe_extract_encode_rpc_and_queue(
    tmp_path, monkeypatch
):
    """t78: every async store component lands in the JSONL stream."""
    trace = sys.modules[PACKAGE + ".mask_trace"]
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()

    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g20", ("r1",), _step(request))
    adapter.drain_stores()

    records = _trace_records(tmp_path)
    by_event: dict[str, list[dict]] = {}
    for record in records:
        by_event.setdefault(record["event"], []).append(record)
    for name in (
        "store_describe",
        "store_extract",
        "store_encode",
        "store_rpc",
        "queue_wait",
    ):
        assert by_event.get(name), f"missing {name} in {sorted(by_event)}"

    describe = by_event["store_describe"][0]
    assert describe["thread"] == "engine"
    assert describe["request_id"] == "r1"
    assert describe["group_id"] == "g20"
    assert describe["rank"] == 0
    assert describe["tid"] > 0
    assert describe["bytes"] > 0
    assert describe["status"] == "ok"

    extract = by_event["store_extract"][0]
    assert extract["thread"] == "worker"
    assert extract["bytes"] > 0
    assert extract["sync_ns"] >= 0
    assert extract["index_ns"] >= 0
    assert extract["status"] == "ok"

    encode = by_event["store_encode"][0]
    assert encode["thread"] == "worker"
    assert encode["encoding"] == "token_range_v1"
    assert encode["bytes"] > 0
    assert encode["status"] == "ok"

    rpc = by_event["store_rpc"][0]
    assert rpc["rpc_phase"] in {"submit", "done"}
    assert rpc["server_hash"] == "unavailable"
    assert rpc["server_hash_status"] == "planned"
    assert rpc["bytes"] > 0
    assert rpc["status"] == "ok"

    queue = [item for item in by_event["queue_wait"] if item.get("kind") == "store"]
    assert queue
    assert queue[0]["request_id"] == "r1"
    assert queue[0]["qsize"] >= 0
    assert queue[0]["status"] == "ok"
    adapter.close()


def test_async_store_trace_is_inert_when_disabled(tmp_path, monkeypatch):
    """t78: no opt-in means no file and no ordering side effects."""
    trace = sys.modules[PACKAGE + ".mask_trace"]
    monkeypatch.delenv(trace.ENV_ENABLED, raising=False)
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()

    server = _FakeServer()
    adapter = _make_adapter(FakeTransport(server))
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g20", ("r1",), _step(request))
    adapter.drain_stores()

    assert adapter.metrics["group_store_committed"] == 1
    assert list(tmp_path.iterdir()) == []
    assert trace.stats()["lines"] == 0
    adapter.close()


def test_async_store_trace_keeps_the_terminal_failure_path(
    tmp_path, monkeypatch
):
    """t78: tracing a failed store must not swallow or reorder the failure."""
    trace = sys.modules[PACKAGE + ".mask_trace"]
    monkeypatch.setenv(trace.ENV_ENABLED, "1")
    monkeypatch.setenv(trace.ENV_DIR, str(tmp_path))
    trace.reset_state()

    server = _FakeServer()
    transport = FakeTransport(server)

    def failing_store(spec, payload):
        raise OSError("boom")

    transport.store_resource_block = failing_store  # type: ignore[assignment]
    adapter = _make_adapter(transport, store_retries=0)
    adapter.register()
    request = _request("r1", load_prefix_end=0, store_prefix_end=TOKENS)
    adapter.submit_group_store("g20", ("r1",), _step(request))
    adapter.drain_stores()

    assert adapter.metrics["group_store_committed"] == 0
    assert adapter.metrics["group_store_failed"] >= 1
    assert adapter.take_finished()[0] == {"r1"}
    failed = [
        record
        for record in _trace_records(tmp_path)
        if record["event"] == "store_rpc" and record["status"] == "failed"
    ]
    assert failed
    assert failed[0]["error_code"] == "IO_ERROR"
    adapter.close()
