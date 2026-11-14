# SPDX-License-Identifier: Apache-2.0
"""Production wiring for the DSV4.1 layerwise v2 group runtime.

T4 shipped the data plane (``adapter.py``/``boundary.py``) and the server-side
``resource_blocks_v1`` module, but left the connector waiting for a caller to
bind an adapter factory. This module closes that gap:

* :func:`build_grouped_plan_from_registration` derives the grouped transfer
  plan directly from the caches the stock worker adapter registered. Cache
  names carry the layer index and the resource kind, so no engine import and
  no hand-written manifest is needed.
* :func:`create_production_adapter` builds
  :class:`~lmcache_ascend.integration.vllm.dsv41_lw_v2.adapter.DSV41LwV2GroupAdapter`
  on the *real* ``worker_adapter.kv_caches`` and the *real* stock request
  client owned by the worker adapter. Nothing is faked and no host or port is
  hard-coded: the multiprocess client already knows how to reach the server.
* :data:`DEFAULT_ADAPTER_FACTORY` is installed by
  :class:`~lmcache_ascend.integration.vllm.dsv41_lw_v2.connector.DSV41LwV2Connector`
  at construction time, so the normal vLLM startup path builds the adapter in
  ``register_kv_caches`` and installs the boundary codec before
  ``initialize_kv_cache``.

Every missing dependency raises :class:`DSV41LwV2ProductionUnavailable` with
the dependency name; the connector converts it to
``DSV41LwV2NotReadyError`` instead of silently falling back to the stock
full-tensor copy path.

Topology note: the consumer segments mirror
``vllm_ascend.models.deepseek_v41.topology.DSV41_LW_V2_CONSUMER_SEGMENTS``
(t3 branch ``dsv41_lw_v2``). They are duplicated here as literals because the
worker must not import the engine package while it is still initializing its
own caches; the plan validator rejects any registration that does not match
the fixed four-source DSV4.1 topology.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import Any
import hashlib
import json
import re

# Third Party
import torch
from lmcache.v1.multiprocess.layerwise import (
    CacheTransferResource,
    LayerwiseTransferPlan,
)

# First Party
from lmcache_ascend.integration.vllm.dsv41_lw_v2.adapter import (
    DSV41LwV2GroupAdapter,
    MPResourceBlockTransport,
)
from lmcache_ascend.integration.vllm.dsv41_lw_v2.plan import (
    DSV41_LW_V2_BOUNDARY_LAYERS,
    DSV41_LW_V2_LAYER_COUNT,
    DSV41_LW_V2_SOURCE_LAYERS,
    GroupedTransferPlan,
    build_grouped_plan,
)

#: Blocking timeout for the registration acknowledgement and group loads.
PRODUCTION_LOAD_TIMEOUT = 30.0
#: Blocking timeout used while ``drain_stores`` waits for committed stores.
PRODUCTION_STORE_TIMEOUT = 60.0
#: Bounded worker queue; a full queue fails loads / drops stores.
PRODUCTION_MAX_QUEUE = 1024

#: Consumer segments of the four KV sources, in execution order. This is the
#: fixed DSV4.1 backbone layout: source 2 feeds layers 2-7, source 8 feeds
#: 8-13, source 14 feeds 14-19 and source 20 feeds the remaining layers.
DSV41_LW_V2_CONSUMER_SEGMENTS: tuple[tuple[int, tuple[int, ...]], ...] = (
    (2, tuple(range(2, 8))),
    (8, tuple(range(8, 14))),
    (14, tuple(range(14, 20))),
    (20, tuple(range(20, DSV41_LW_V2_LAYER_COUNT))),
)

_CACHE_NAME_RE = re.compile(
    r"^(?P<prefix>.+)\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<kind>long_kv_cache|indexer\.k_cache|compressor\.state_cache|"
    r"swa_cache)$"
)
_FOLDED_INDEX_MARKER = ".indexer.k_cache_folded"
_ATTENTION_SUFFIX = ".attn"

_REQUIRED_CLIENT_METHODS = (
    "register_resource_blocks",
    "store_resource_block",
    "retrieve_resource_block",
    "lookup_resource_blocks",
)


class DSV41LwV2ProductionUnavailable(RuntimeError):
    """A dependency required by the production factory is missing.

    Attributes:
        dependency: Short name of the missing dependency, suitable for a
            ``NotReady`` message.
    """

    def __init__(self, dependency: str, detail: str) -> None:
        self.dependency = dependency
        super().__init__(f"{dependency}: {detail}")


def _layer_name(prefix: str, index: int) -> str:
    return f"{prefix}.layers.{index}.self_attn{_ATTENTION_SUFFIX}"


def _stem(prefix: str, index: int) -> str:
    return f"{prefix}.layers.{index}.self_attn"


def _parse_cache_names(
    cache_names: Sequence[str],
) -> tuple[str, dict[int, dict[str, int]]]:
    """Parse registered cache names into ``layer -> kind -> tensor index``.

    Raises:
        DSV41LwV2ProductionUnavailable: when the registration is not the fixed
            40-layer DSV4.1 backbone or contains an unknown cache kind.
    """
    parsed: dict[int, dict[str, int]] = {}
    prefix: str | None = None
    for tensor_index, name in enumerate(cache_names):
        if _FOLDED_INDEX_MARKER in name:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                "indexed KV caches must use the unfolded .indexer.k_cache "
                f"form (got {name!r})",
            )
        match = _CACHE_NAME_RE.match(name)
        if match is None:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                "unrecognized DSV4.1 cache name "
                f"{name!r}; expected .swa_cache/.long_kv_cache/"
                ".indexer.k_cache/.compressor.state_cache",
            )
        name_prefix = match.group("prefix")
        if prefix is None:
            prefix = name_prefix
        elif name_prefix != prefix:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                f"mixed cache-name prefixes {prefix!r} and {name_prefix!r}",
            )
        layer = int(match.group("layer"))
        kind = match.group("kind")
        kinds = parsed.setdefault(layer, {})
        if kind in kinds:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                f"duplicate {kind} registration for layer {layer}",
            )
        kinds[kind] = tensor_index

    if prefix is None:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches", "no registered caches"
        )
    if sorted(parsed) != list(range(DSV41_LW_V2_LAYER_COUNT)):
        missing = sorted(set(range(DSV41_LW_V2_LAYER_COUNT)) - set(parsed))
        extra = sorted(set(parsed) - set(range(DSV41_LW_V2_LAYER_COUNT)))
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches",
            "expected the fixed 40-layer DSV4.1 backbone; "
            f"missing layers={missing!r} unexpected layers={extra!r}",
        )
    for layer, kinds in parsed.items():
        if "swa_cache" not in kinds:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                f"layer {layer} has no .swa_cache recompute resource",
            )
        if layer in DSV41_LW_V2_SOURCE_LAYERS:
            for kind in ("long_kv_cache", "indexer.k_cache"):
                if kind not in kinds:
                    raise DSV41LwV2ProductionUnavailable(
                        "worker_adapter.kv_caches",
                        f"KV source layer {layer} has no .{kind} resource",
                    )
        else:
            for kind in ("long_kv_cache", "indexer.k_cache"):
                if kind in kinds:
                    raise DSV41LwV2ProductionUnavailable(
                        "worker_adapter.kv_caches",
                        f"non-source layer {layer} unexpectedly registered "
                        f".{kind}",
                    )
        has_boundary_kind = "compressor.state_cache" in kinds
        if layer in DSV41_LW_V2_BOUNDARY_LAYERS and not has_boundary_kind:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                f"boundary source layer {layer} has no .compressor.state_cache",
            )
        if layer not in DSV41_LW_V2_BOUNDARY_LAYERS and has_boundary_kind:
            raise DSV41LwV2ProductionUnavailable(
                "worker_adapter.kv_caches",
                f"layer {layer} must not own a boundary resource",
            )
    return prefix, parsed


def _engine_group_for_source(
    engine_group_infos: Sequence[Any], source: int
) -> int:
    """Return the stock engine group whose layers include ``source``."""
    matches: set[int] = set()
    for info in engine_group_infos:
        layer_indices = getattr(info, "layer_indices", None)
        if layer_indices is None:
            continue
        if source in tuple(int(index) for index in layer_indices):
            matches.add(int(info.engine_group_id))
    if not matches:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.engine_group_infos",
            f"no KV engine group covers KV source layer {source}",
        )
    if len(matches) > 1:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.engine_group_infos",
            f"KV source layer {source} belongs to several engine groups "
            f"{sorted(matches)!r}",
        )
    return matches.pop()


def build_grouped_plan_from_registration(
    kv_caches: Mapping[str, Any],
    engine_group_infos: Sequence[Any],
) -> GroupedTransferPlan:
    """Derive the fixed grouped plan from the live stock cache registration.

    Args:
        kv_caches: Caches registered by the stock worker adapter, keyed by
            their engine cache name.
        engine_group_infos: Stock engine-group descriptors from the same
            registration (used for the token resources' ``engine_group_id``).

    Returns:
        The grouped transfer plan matching the registered caches.

    Raises:
        DSV41LwV2ProductionUnavailable: when the registration is incomplete,
            uses the wrong topology, or cannot be mapped to engine groups.
    """
    if not isinstance(kv_caches, Mapping) or not kv_caches:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches", "no registered KV caches"
        )
    cache_names = tuple(kv_caches)
    prefix, parsed = _parse_cache_names(cache_names)
    source_groups = {
        source: _engine_group_for_source(engine_group_infos, source)
        for source in DSV41_LW_V2_SOURCE_LAYERS
    }

    resources: list[CacheTransferResource] = []
    for index in range(DSV41_LW_V2_LAYER_COUNT):
        kinds = parsed[index]
        layer = _layer_name(prefix, index)
        resources.append(
            CacheTransferResource(
                resource_id=f"{_stem(prefix, index)}.swa_cache",
                tensor_indices=(kinds["swa_cache"],),
                engine_group_id=None,
                producer_layer=layer,
                consumer_layers=(layer,),
                storage_kind="recompute",
            )
        )
    for source, consumers in DSV41_LW_V2_CONSUMER_SEGMENTS:
        kinds = parsed[source]
        layer = _layer_name(prefix, source)
        consumer_layers = tuple(
            _layer_name(prefix, index) for index in consumers
        )
        resources.append(
            CacheTransferResource(
                resource_id=f"{_stem(prefix, source)}.long_kv_cache",
                tensor_indices=(kinds["long_kv_cache"],),
                engine_group_id=source_groups[source],
                producer_layer=layer,
                consumer_layers=consumer_layers,
                storage_kind="token_range",
            )
        )
        resources.append(
            CacheTransferResource(
                resource_id=f"{_stem(prefix, source)}.indexer.k_cache",
                tensor_indices=(kinds["indexer.k_cache"],),
                engine_group_id=source_groups[source],
                producer_layer=layer,
                consumer_layers=(layer,),
                storage_kind="token_range",
            )
        )
        if source in DSV41_LW_V2_BOUNDARY_LAYERS:
            resources.append(
                CacheTransferResource(
                    resource_id=(
                        f"{_stem(prefix, source)}.compressor.state_cache"
                    ),
                    tensor_indices=(kinds["compressor.state_cache"],),
                    engine_group_id=None,
                    producer_layer=layer,
                    consumer_layers=(layer,),
                    storage_kind="boundary_state",
                )
            )

    base = LayerwiseTransferPlan(
        layer_order=tuple(
            _layer_name(prefix, index)
            for index in range(DSV41_LW_V2_LAYER_COUNT)
        ),
        resources=tuple(resources),
    )
    return build_grouped_plan(
        base,
        source_layer_ids=DSV41_LW_V2_SOURCE_LAYERS,
        boundary_source_layer_ids=DSV41_LW_V2_BOUNDARY_LAYERS,
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def compute_plan_hash(plan: GroupedTransferPlan) -> str:
    """Stable hash of the registered layer order and resource identities."""
    payload = {
        "version": 1,
        "layer_order": list(plan.base.layer_order),
        "resources": [
            {
                "resource_id": resource.resource_id,
                "tensor_indices": list(resource.tensor_indices),
                "engine_group_id": resource.engine_group_id,
                "producer_layer": resource.producer_layer,
                "consumer_layers": list(resource.consumer_layers),
                "storage_kind": resource.storage_kind,
            }
            for resource in plan.base.resources
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _resource_planes(
    resource: CacheTransferResource,
    cache_names: Sequence[str],
    kv_caches: Mapping[str, Any],
) -> tuple[Any, ...]:
    index = resource.tensor_indices[0]
    if index < 0 or index >= len(cache_names):
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches",
            f"tensor index {index} of {resource.resource_id} is outside the "
            f"{len(cache_names)} registered caches",
        )
    name = cache_names[index]
    if name != resource.resource_id:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches",
            f"plan resource {resource.resource_id!r} resolves to registered "
            f"cache {name!r}; the plan tensor-index order does not match the "
            "live cache registration",
        )
    try:
        value = kv_caches[name]
    except KeyError as exc:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches",
            f"registered cache {name!r} is missing",
        ) from exc
    planes = value if isinstance(value, tuple) else (value,)
    if not planes:
        raise DSV41LwV2ProductionUnavailable(
            "worker_adapter.kv_caches", f"cache {name!r} has no planes"
        )
    return tuple(planes)


def compute_layout_hash(
    plan: GroupedTransferPlan,
    kv_caches: Mapping[str, Any],
) -> str:
    """Stable hash of the transferable cache geometry."""
    cache_names = tuple(kv_caches)
    entries: list[dict[str, Any]] = []
    for resource in plan.base.resources:
        if resource.storage_kind == "recompute":
            continue
        planes = _resource_planes(resource, cache_names, kv_caches)
        entries.append(
            {
                "resource_id": resource.resource_id,
                "cache_name": cache_names[resource.tensor_indices[0]],
                "planes": [
                    {
                        "shape": list(plane.shape),
                        "stride": list(plane.stride()),
                        "dtype": _dtype_name(plane.dtype),
                    }
                    for plane in planes
                ],
            }
        )
    encoded = json.dumps(
        {"version": 1, "resources": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, dependency: str, detail: str) -> None:
    if not condition:
        raise DSV41LwV2ProductionUnavailable(dependency, detail)


def create_production_adapter(
    plan: GroupedTransferPlan, connector: Any
) -> DSV41LwV2GroupAdapter:
    """Build the group adapter on the real worker registration.

    Args:
        plan: Grouped plan derived from this worker's cache registration.
        connector: The owning ``DSV41LwV2Connector`` (worker role).

    Returns:
        A registered adapter whose layouts match ``worker_adapter.kv_caches``.

    Raises:
        DSV41LwV2ProductionUnavailable: for every missing/incompatible
            dependency, naming it in the message.
    """
    _require(
        isinstance(plan, GroupedTransferPlan),
        "grouped plan",
        "the connector has no GroupedTransferPlan bound",
    )
    worker_adapter = getattr(connector, "worker_adapter", None)
    _require(
        worker_adapter is not None,
        "worker_adapter",
        "only the KV worker role registers caches; the scheduler has no "
        "worker adapter",
    )
    kv_caches = getattr(worker_adapter, "kv_caches", None)
    _require(
        isinstance(kv_caches, Mapping) and bool(kv_caches),
        "worker_adapter.kv_caches",
        "the stock registration did not populate any KV cache",
    )
    request_client = getattr(worker_adapter, "req_client", None)
    _require(
        request_client is not None,
        "worker_adapter.req_client",
        "the stock request client is not initialized",
    )
    missing_methods = [
        method
        for method in _REQUIRED_CLIENT_METHODS
        if not callable(getattr(request_client, method, None))
    ]
    _require(
        not missing_methods,
        "worker_adapter.req_client",
        "the request client lacks resource-block RPCs "
        f"{missing_methods!r}; the LMCache server must expose "
        "resource_blocks_v1",
    )
    chunk_size = getattr(worker_adapter, "lmcache_tokens_per_chunk", 0)
    blocks_in_chunk = getattr(worker_adapter, "blocks_in_chunk", 0)
    _require(
        isinstance(chunk_size, int) and chunk_size > 0,
        "worker_adapter.lmcache_tokens_per_chunk",
        f"invalid chunk size {chunk_size!r}",
    )
    _require(
        isinstance(blocks_in_chunk, int)
        and blocks_in_chunk > 0
        and chunk_size % blocks_in_chunk == 0,
        "worker_adapter.blocks_in_chunk",
        f"chunk size {chunk_size} is not a multiple of blocks_in_chunk "
        f"{blocks_in_chunk!r}",
    )
    block_size = chunk_size // blocks_in_chunk
    model_name = getattr(worker_adapter, "model_name", "")
    _require(
        isinstance(model_name, str) and bool(model_name),
        "worker_adapter.model_name",
        "the stock worker adapter has no model name",
    )
    world_size = getattr(worker_adapter, "world_size", 0)
    _require(
        isinstance(world_size, int) and world_size > 0,
        "worker_adapter.world_size",
        f"invalid KV world size {world_size!r}",
    )
    instance_id = getattr(worker_adapter, "instance_id", None)
    _require(
        instance_id is not None,
        "worker_adapter.instance_id",
        "the stock worker adapter has no instance id",
    )
    parallel_strategy = getattr(worker_adapter, "parallel_strategy", None)
    worker_id = getattr(parallel_strategy, "kv_worker_id", None)
    _require(
        worker_id is not None,
        "worker_adapter.parallel_strategy.kv_worker_id",
        "the KV worker id is unknown",
    )

    transport = MPResourceBlockTransport(
        request_client, timeout=PRODUCTION_LOAD_TIMEOUT
    )
    adapter = DSV41LwV2GroupAdapter(
        plan,
        transport=transport,
        instance_id=str(instance_id),
        model_name=model_name,
        world_size=world_size,
        worker_id=worker_id,
        plan_hash=compute_plan_hash(plan),
        layout_hash=compute_layout_hash(plan, kv_caches),
        registration_epoch=int(
            getattr(connector, "_lw_v2_registration_epoch", 0)
        ),
        server_generation=int(
            getattr(connector, "_lw_v2_server_generation", 0)
        ),
        chunk_size=chunk_size,
        block_size=block_size,
        kv_caches=kv_caches,
        cache_names=tuple(kv_caches),
        load_timeout=PRODUCTION_LOAD_TIMEOUT,
        store_timeout=PRODUCTION_STORE_TIMEOUT,
        max_queue=PRODUCTION_MAX_QUEUE,
    )
    ack = adapter.register()
    setter = getattr(connector, "set_generations", None)
    if callable(setter):
        setter(
            registration_epoch=adapter.registration_epoch,
            server_generation=ack.server_generation,
        )
    return adapter


#: Factory installed by the connector when no explicit factory is bound.
DEFAULT_ADAPTER_FACTORY = create_production_adapter
