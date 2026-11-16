# SPDX-License-Identifier: Apache-2.0
"""v3 W2 batch retrieve: capability gate, real RPC and callback spans.

The batching container only groups RPCs.  These tests pin the observable
contract without a GPU or a live server:

* default off / capability-gated, single path unchanged;
* one real ``retrieve_resource_blocks`` RPC per bounded container, never a
  loop of the single RPC under a batch name;
* W0 item partition (64 item batches, long 299 / short 19 per rank);
* payload parity between the single and batch paths;
* partial/miss/corrupt batches fail the group closed and are not retried
  through the single path;
* the optional callback span answers the receive-loop HOL question.
"""

# Standard
import importlib.util
import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

# Third Party
import pytest
import torch

_HELPERS = Path(__file__).with_name("test_dsv41_lw_v2_adapter.py")
_spec = importlib.util.spec_from_file_location(
    "_lw_v2_adapter_helpers", _HELPERS
)
helpers = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helpers
_spec.loader.exec_module(helpers)  # type: ignore[union-attr]

adapter_mod = helpers.adapter_mod
connector_mod = helpers.connector_mod
plan_mod = helpers.plan_mod

BATCH_CAPABILITY = adapter_mod._BATCH_RETRIEVE_CAPABILITY


class _SpecView:
    """The fields the fake stock server reads from a spec."""

    def __init__(self, request, item):
        self.plan_hash = request.plan_hash
        self.layout_hash = request.layout_hash
        self.stock_key = request.stock_key
        self.resource_id = item.resource_id
        self.prefix_end = item.prefix_end
        self.start_token = item.start_token
        self.end_token = item.end_token


class BatchServer(helpers._FakeServer):
    """Fake server with one batched retrieve entry point."""

    def __init__(self, **kwargs):
        kwargs.setdefault(
            "capabilities",
            tuple(
                dict.fromkeys(
                    tuple(helpers.adapter_mod.REQUIRED_CAPABILITIES)
                    + (BATCH_CAPABILITY,)
                )
            ),
        )
        super().__init__(**kwargs)
        self.batch_slices: dict[tuple, bytes] = {}
        self.batch_item_counts: list[int] = []

    def retrieve_batch(self, request):
        payloads: list[bytes] = []
        item_results: list = []
        missing = False
        offset = 0
        with self.lock:
            for item in request.items:
                view = _SpecView(request, item)
                if item.resource_id in helpers._BOUNDARY_IDS:
                    payload = self.boundaries.get(self._boundary_key(view))
                else:
                    payload = self.blocks.get(
                        self._token_key(
                            view, item.start_token, item.end_token
                        )
                    )
                if payload is None:
                    missing = True
                    item_results.append(
                        adapter_mod.ResourceBlockBatchItemResult(
                            operation_id=item.operation_id,
                            resource_id=item.resource_id,
                            engine_group_id=item.engine_group_id,
                            start_token=item.start_token,
                            end_token=item.end_token,
                            prefix_end=item.prefix_end,
                            status="miss",
                            error_code=adapter_mod.RESOURCE_MISSING,
                        )
                    )
                    continue
                self.batch_slices[
                    (item.resource_id, item.start_token, item.end_token)
                ] = payload
                item_results.append(
                    adapter_mod.ResourceBlockBatchItemResult(
                        operation_id=item.operation_id,
                        resource_id=item.resource_id,
                        engine_group_id=item.engine_group_id,
                        start_token=item.start_token,
                        end_token=item.end_token,
                        prefix_end=item.prefix_end,
                        status="ok",
                        offset=offset,
                        length=len(payload),
                    )
                )
                payloads.append(payload)
                offset += len(payload)
        self.batch_item_counts.append(len(request.items))
        if missing:
            return (
                adapter_mod.ResourceBlockBatchResult(
                    batch_id=request.batch_id,
                    instance_id=request.instance_id,
                    registration_epoch=request.registration_epoch,
                    server_generation=request.server_generation,
                    plan_hash=request.plan_hash,
                    layout_hash=request.layout_hash,
                    status="miss",
                    error_code=adapter_mod.RESOURCE_MISSING,
                    items=tuple(item_results),
                ),
                b"",
            )
        payload = b"".join(payloads)
        return (
            adapter_mod.ResourceBlockBatchResult(
                batch_id=request.batch_id,
                instance_id=request.instance_id,
                registration_epoch=request.registration_epoch,
                server_generation=request.server_generation,
                plan_hash=request.plan_hash,
                layout_hash=request.layout_hash,
                status="ok",
                total_bytes=len(payload),
                items=tuple(item_results),
            ),
            payload,
        )


class BatchTransport(helpers.FakeTransport):
    """Fake transport that implements the batched RPC extension."""

    def __init__(self, server):
        super().__init__(server)
        self.batch_calls: list = []
        self.batch_override = None

    def retrieve_resource_blocks(self, request):
        self.batch_calls.append(request)
        if self.batch_override is not None:
            future = Future()
            future.set_result(self.batch_override(request))
            return future
        result, payload = self.server.retrieve_batch(request)
        return self._future((result, payload))


@pytest.fixture(autouse=True)
def _reset_trace(monkeypatch):
    monkeypatch.delenv(adapter_mod._BATCH_RETRIEVE_ENV, raising=False)
    monkeypatch.delenv("DSV41_LW_V2_MASK_TRACE", raising=False)
    monkeypatch.delenv("DSV41_LW_V2_MASK_TRACE_DIR", raising=False)
    adapter_mod.mask_trace.reset_state()
    yield
    adapter_mod.mask_trace.reset_state()


def _stock_key(request):
    return helpers._stock_key(
        request.request_id, request.stock_key.token_bytes
    )


def _seed(server, request):
    helpers._seed_token_blocks(server, request)


def _request(prefix_end=helpers.TOKENS):
    return helpers._request(
        "r1", step_id=1, load_prefix_end=prefix_end, blocks=((0, 1, 2, 3),)
    )


def _load(adapter, request, group_id="g2"):
    step = helpers._step(request)
    adapter.schedule_group_retrieves(group_id, [request.request_id], step)
    adapter.wait_group(group_id, [request.request_id], step)
    return step


def _group_resources(group_id):
    group = helpers.PLAN.group_by_id(group_id)
    return list(group.token_range_resources)


def _expected_rows(server, resource_id, chunk):
    payload = server.blocks[
        (
            helpers.PLAN_HASH,
            helpers.LAYOUT_HASH,
            helpers.MODEL,
            helpers.WORLD,
            "",
            resource_id,
            chunk.start_token,
            chunk.end_token,
            helpers._chunk_hash(
                _request().stock_key.token_bytes,
                chunk.start_token,
                chunk.end_token,
            ),
        )
    ]
    return adapter_mod.decode_token_payload(
        payload, resource_id=resource_id
    )


def test_batch_retrieve_defaults_off_and_requires_the_capability(monkeypatch):
    request = _request()
    server = BatchServer()
    _seed(server, request)
    transport = BatchTransport(server)

    # Default: env absent -> single path even though the server can batch.
    adapter = helpers._make_adapter(transport)
    assert adapter.batch_retrieve is False
    _load(adapter, request)
    assert transport.batch_calls == []
    assert server.retrieve_calls

    # Env on but the server did not negotiate the capability -> single path.
    server2 = helpers._FakeServer(
        capabilities=tuple(
            capability
            for capability in adapter_mod.REQUIRED_CAPABILITIES
            if capability != BATCH_CAPABILITY
        )
    )
    _seed(server2, request)
    transport2 = BatchTransport(server2)
    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    adapter2 = helpers._make_adapter(transport2)
    adapter2.register()
    assert adapter2.batch_retrieve is False
    _load(adapter2, request)
    assert transport2.batch_calls == []
    assert server2.retrieve_calls


def test_batch_retrieve_uses_one_real_rpc_and_payloads_match_single(
    monkeypatch,
):
    request = _request()
    server = BatchServer()
    _seed(server, request)
    transport = BatchTransport(server)
    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    caches = helpers._make_caches()
    adapter = helpers._make_adapter(transport, caches=caches)
    adapter.register()
    assert adapter.batch_retrieve is True

    _load(adapter, request)

    # One batch (2 token resources x 2 chunks + 1 boundary = 5 items) and no
    # single retrieve RPC hidden under the batch.
    assert len(transport.batch_calls) == 1
    assert len(transport.batch_calls[0].items) == 5
    assert server.retrieve_calls == []
    assert adapter.metrics["batch_load_rpcs"] == 1
    assert adapter.metrics["batch_load_items"] == 5
    assert adapter.metrics["single_load_rpcs"] == 0

    # Token rows were applied and the boundary payload was retained.
    group = helpers.PLAN.group_by_id("g2")
    for resource in group.token_range_resources:
        cache = caches[resource.resource_id]
        rows = []
        for chunk in (
            adapter_mod._ChunkRef(0, helpers.CHUNK_SIZE, (0, 1)),
            adapter_mod._ChunkRef(
                helpers.CHUNK_SIZE,
                helpers.TOKENS,
                (2, 3),
            ),
        ):
            plans = _expected_rows(server, resource.resource_id, chunk)
            rows.append(plans[0])
        expected = torch.cat(rows, dim=0)
        assert torch.equal(cache[: expected.shape[0]], expected)

    boundary = group.boundary_resource
    key = (request.request_id, boundary.resource_id, helpers.TOKENS)
    assert adapter._boundary_payloads[key] == server.boundaries[
        (
            helpers.PLAN_HASH,
            helpers.LAYOUT_HASH,
            boundary.resource_id,
            helpers.TOKENS,
            helpers._chunk_hash(
                request.stock_key.token_bytes, 0, helpers.TOKENS
            ),
        )
    ]


def test_batch_item_partition_matches_the_w0_manifest():
    """Long 37 chunks -> 75/75/75/74 items, short 2 -> 5/5/5/4."""
    server = BatchServer()
    transport = BatchTransport(server)
    adapter = helpers._make_adapter(transport)
    adapter.register()
    adapter._batch_retrieve_requested = True

    groups = [
        helpers.PLAN.group_by_id("g2"),
        helpers.PLAN.group_by_id("g8"),
        helpers.PLAN.group_by_id("g14"),
        helpers.PLAN.group_by_id("g20"),
    ]
    long_chunks = [
        adapter_mod._ChunkRef(i * 256, (i + 1) * 256, (i,)) for i in range(37)
    ]
    for group, expected_batches in zip(
        groups, ([64, 11], [64, 11], [64, 11], [64, 10])
    ):
        items = []
        for chunk in long_chunks:
            for resource in group.token_range_resources:
                items.append(
                    adapter_mod._LoadItem(
                        spec=SimpleNamespace(
                            resource_id=resource.resource_id
                        ),
                        chunk=chunk,
                        boundary=False,
                    )
                )
        if group.has_boundary:
            items.append(
                adapter_mod._LoadItem(
                    spec=SimpleNamespace(
                        resource_id=group.boundary_resource.resource_id
                    ),
                    chunk=None,
                    boundary=True,
                )
            )
        planned = adapter._plan_batches(items)
        assert len(items) == sum(expected_batches)
        assert [len(batch) for batch in planned] == expected_batches

    short_chunks = [
        adapter_mod._ChunkRef(0, 256, (0,)),
        adapter_mod._ChunkRef(256, 512, (1,)),
    ]
    for group, expected in zip(groups, ([5, 5, 5, 4])):
        items = []
        for chunk in short_chunks:
            for resource in group.token_range_resources:
                items.append(
                    adapter_mod._LoadItem(
                        spec=SimpleNamespace(
                            resource_id=resource.resource_id
                        ),
                        chunk=chunk,
                        boundary=False,
                    )
                )
        if group.has_boundary:
            items.append(
                adapter_mod._LoadItem(
                    spec=SimpleNamespace(
                        resource_id=group.boundary_resource.resource_id
                    ),
                    chunk=None,
                    boundary=True,
                )
            )
        assert len(items) == expected
        assert len(adapter._plan_batches(items)) == 1


def test_batch_miss_fails_the_group_and_is_not_retried_single(
    monkeypatch,
):
    request = _request()
    server = BatchServer()
    _seed(server, request)
    # Drop one token payload so the batch reports a miss.
    doomed = next(
        key
        for key in server.blocks
        if key[5] == _group_resources("g2")[0].resource_id
    )
    server.blocks.pop(doomed)
    transport = BatchTransport(server)
    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    adapter = helpers._make_adapter(transport)
    adapter.register()

    with pytest.raises(adapter_mod.DSV41GroupLoadError) as excinfo:
        _load(adapter, request)
    assert excinfo.value.error_code == adapter_mod.RESOURCE_MISSING
    assert transport.batch_calls
    assert server.retrieve_calls == []
    assert request.request_id in adapter.take_failed_receiving()


def test_batch_overlapping_slices_are_rejected(monkeypatch):
    request = _request()
    server = BatchServer()
    _seed(server, request)
    transport = BatchTransport(server)

    def corrupt(request):
        result, payload = server.retrieve_batch(request)
        items = list(result.items)
        first = items[0]
        second = items[1]
        items[1] = adapter_mod.ResourceBlockBatchItemResult(
            operation_id=second.operation_id,
            resource_id=second.resource_id,
            engine_group_id=second.engine_group_id,
            start_token=second.start_token,
            end_token=second.end_token,
            prefix_end=second.prefix_end,
            status="ok",
            offset=first.offset,
            length=second.length,
        )
        return (
            adapter_mod.ResourceBlockBatchResult(
                batch_id=result.batch_id,
                instance_id=result.instance_id,
                registration_epoch=result.registration_epoch,
                server_generation=result.server_generation,
                plan_hash=result.plan_hash,
                layout_hash=result.layout_hash,
                status="ok",
                total_bytes=result.total_bytes,
                items=tuple(items),
            ),
            payload,
        )

    transport.batch_override = corrupt
    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    adapter = helpers._make_adapter(transport)
    adapter.register()
    with pytest.raises(adapter_mod.DSV41GroupLoadError) as excinfo:
        _load(adapter, request)
    assert excinfo.value.error_code == adapter_mod.PAYLOAD_CORRUPT
    assert server.retrieve_calls == []


def test_batch_callback_span_reports_the_receive_loop_components(
    monkeypatch, tmp_path
):
    request = _request()
    server = BatchServer()
    _seed(server, request)
    transport = BatchTransport(server)
    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    monkeypatch.setenv("DSV41_LW_V2_MASK_TRACE", "1")
    monkeypatch.setenv("DSV41_LW_V2_MASK_TRACE_DIR", str(tmp_path))
    adapter_mod.mask_trace.reset_state()
    adapter = helpers._make_adapter(transport)
    adapter.register()
    _load(adapter, request)

    events = []
    for path in tmp_path.glob("mask-trace-*.jsonl"):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    callbacks = [event for event in events if event["event"] == "load_callback"]
    assert callbacks
    callback = next(event for event in callbacks if event["batch"])
    for field in (
        "submit_to_ready_ns",
        "payload_decode_ns",
        "host_apply_ns",
        "device_copy_ns",
        "total_ns",
    ):
        assert field in callback
        assert callback[field] >= 0
    assert callback["response_decode_status"] == "unavailable"
    assert adapter.metrics["load_callback_count"] >= 1
    assert adapter.metrics["load_callback_total_ns"] > 0


def test_single_and_batch_paths_apply_identical_bytes(monkeypatch):
    """Four-group fence parity: same seed, same applied cache/boundary bytes."""
    request = _request()
    server = BatchServer()
    _seed(server, request)
    transport_single = BatchTransport(server)
    caches_single = helpers._make_caches()
    adapter_single = helpers._make_adapter(
        transport_single, caches=caches_single
    )
    adapter_single.register()
    for group_id in ("g2", "g8", "g14", "g20"):
        _load(adapter_single, request, group_id)

    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    transport_batch = BatchTransport(server)
    caches_batch = helpers._make_caches()
    adapter_batch = helpers._make_adapter(transport_batch, caches=caches_batch)
    adapter_batch.register()
    for group_id in ("g2", "g8", "g14", "g20"):
        _load(adapter_batch, request, group_id)

    assert transport_batch.batch_calls
    assert transport_single.batch_calls == []
    assert server.retrieve_calls
    for name, tensor in caches_single.items():
        assert torch.equal(tensor, caches_batch[name]), name
    assert adapter_single._boundary_payloads == adapter_batch._boundary_payloads
    assert transport_batch.batch_calls


# ---------------------------------------------------------------------------
# t92: trace-off zero-clock and W0 37-chunk parity
# ---------------------------------------------------------------------------

LONG_CHUNKS = 37
LONG_BLOCKS = 74
LONG_TOKENS = LONG_CHUNKS * helpers.CHUNK_SIZE


def _long_request():
    return helpers._request(
        "r1",
        step_id=1,
        load_prefix_end=LONG_TOKENS,
        blocks=(tuple(range(LONG_BLOCKS)),),
    )


def _long_caches() -> dict:
    caches = {}
    for index, (resource_id, resource) in enumerate(
        sorted(helpers.PLAN_RESOURCES.items())
    ):
        if resource.storage_kind == "boundary_state":
            continue
        base = torch.arange(
            LONG_BLOCKS * 2 * 3, dtype=torch.float32
        ).reshape(LONG_BLOCKS, 2, 3)
        caches[resource_id] = (base + float(index)).contiguous()
    return caches


def _seed_long(server, request, caches) -> None:
    for resource_id, resource in helpers.PLAN_RESOURCES.items():
        if resource.storage_kind != "token_range":
            continue
        plane = caches[resource_id]
        for chunk_index in range(LONG_CHUNKS):
            start = chunk_index * helpers.CHUNK_SIZE
            end = start + helpers.CHUNK_SIZE
            blocks = request.block_ids_by_engine_group[0][
                start // helpers.BLOCK_SIZE : end // helpers.BLOCK_SIZE
            ]
            spec = adapter_mod.ResourceBlockSpec(
                operation_id="seed",
                direction="load",
                instance_id="instance-1",
                request_id=request.request_id,
                step_id=request.step_id,
                registration_epoch=0,
                server_generation=0,
                plan_hash=helpers.PLAN_HASH,
                layout_hash=helpers.LAYOUT_HASH,
                stock_key=request.stock_key,
                resource_id=resource_id,
                engine_group_id=resource.engine_group_id,
                block_ids=blocks,
                start_token=start,
                end_token=end,
                prefix_end=LONG_TOKENS,
                skip_first_n_tokens=0,
            )
            payload = adapter_mod.encode_token_payload(
                resource_id=resource_id,
                block_ids=blocks,
                planes=(plane[list(blocks)].clone(),),
            )
            server.blocks[server._token_key(spec, start, end)] = payload

    for group_id in ("g2", "g8", "g14"):
        group = helpers.PLAN.group_by_id(group_id)
        spec = adapter_mod.ResourceBlockSpec(
            operation_id="seed-boundary",
            direction="load",
            instance_id="instance-1",
            request_id=request.request_id,
            step_id=request.step_id,
            registration_epoch=0,
            server_generation=0,
            plan_hash=helpers.PLAN_HASH,
            layout_hash=helpers.LAYOUT_HASH,
            stock_key=request.stock_key,
            resource_id=group.boundary_resource.resource_id,
            engine_group_id=None,
            block_ids=(),
            start_token=LONG_TOKENS,
            end_token=LONG_TOKENS,
            prefix_end=LONG_TOKENS,
            skip_first_n_tokens=0,
        )
        server.boundaries[server._boundary_key(spec)] = (
            adapter_mod.BOUNDARY_PAYLOAD_MAGIC + b"seed-long-" + group_id.encode()
        )


def test_trace_off_reads_no_host_clock_and_passes_no_timings(monkeypatch):
    """t92 F1: the load path must not sample the host clock when trace is off."""
    request = _request()
    server = BatchServer()
    _seed(server, request)
    transport = BatchTransport(server)
    monkeypatch.delenv("DSV41_LW_V2_MASK_TRACE", raising=False)
    adapter_mod.mask_trace.reset_state()
    adapter = helpers._make_adapter(transport)
    adapter.register()

    calls = {"count": 0}
    real_monotonic_ns = adapter_mod.time.monotonic_ns

    def counting_monotonic_ns():
        calls["count"] += 1
        return real_monotonic_ns()

    monkeypatch.setattr(
        adapter_mod.time, "monotonic_ns", counting_monotonic_ns
    )
    seen_timings = []
    original_apply = adapter._apply_retrieve

    def recording_apply(**kwargs):
        seen_timings.append(kwargs.get("timings", "missing"))
        return original_apply(**kwargs)

    monkeypatch.setattr(adapter, "_apply_retrieve", recording_apply)
    for group_id in ("g2", "g8", "g14", "g20"):
        _load(adapter, request, group_id)

    assert seen_timings
    assert all(timings is None for timings in seen_timings)
    assert calls["count"] == 0


def test_long_37_chunk_single_and_batch_parity(monkeypatch):
    """t92/O-2: 299 items, 8 batch RPCs and byte parity at W0 long shape."""
    request = _long_request()

    server = BatchServer()
    caches_single = _long_caches()
    _seed_long(server, request, caches_single)
    transport_single = BatchTransport(server)
    adapter_single = helpers._make_adapter(
        transport_single, caches=caches_single
    )
    adapter_single.register()
    for group_id in ("g2", "g8", "g14", "g20"):
        _load(adapter_single, request, group_id)
    assert len(server.retrieve_calls) == 299

    monkeypatch.setenv(adapter_mod._BATCH_RETRIEVE_ENV, "1")
    caches_batch = _long_caches()
    transport_batch = BatchTransport(server)
    adapter_batch = helpers._make_adapter(transport_batch, caches=caches_batch)
    adapter_batch.register()
    for group_id in ("g2", "g8", "g14", "g20"):
        _load(adapter_batch, request, group_id)

    assert len(transport_batch.batch_calls) == 8
    assert len(server.retrieve_calls) == 299
    lengths = [len(call.items) for call in transport_batch.batch_calls]
    assert lengths == [64, 11, 64, 11, 64, 11, 64, 10]
    assert sum(lengths) == 299

    for name, tensor in caches_single.items():
        assert torch.equal(tensor, caches_batch[name]), name
    assert adapter_single._boundary_payloads == adapter_batch._boundary_payloads
