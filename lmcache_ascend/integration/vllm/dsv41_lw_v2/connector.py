# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layerwise v2 connector: strict config, fail-closed group routing.

Scope of this module (T2): the vLLM connector registration entry point, the
``dsv41.lw_v2`` strict boolean switch, the step/request view types shared with
the transport adapter, and the *group-level routing* of the four vLLM hooks
plus the explicit group facades that the layerwise plan (T3) calls directly.

The module deliberately does not implement transport: real retrieves, stores
and boundary capture/restore are injected through the
:class:`DSV41LwV2Adapter` and :class:`DSV41LwV2BoundaryHook` seams (T4/T3).
Whenever a hook would need such a seam and none is bound, the connector fails
closed with :class:`DSV41LwV2NotReadyError` instead of falling back to the
stock full-tensor copy path.

Routing rules (see ``docs/v2-api-contract.md`` §4):

* Only group heads (readiness layers 2, 8, 14, 20) wait or submit stores.
  Every other layer is a pure mapping query: no adapter call, no boundary
  hook, no counter, no future/ring access.
* ``start_load_kv`` only *queues* one retrieve per group for requests with a
  real external hit; it never waits inline.
* ``wait_for_layer_load`` waits the group and then restores its boundary;
  ``save_kv_layer`` captures the group boundary and then submits its store.
  Repeated calls (T3 direct group calls and hook-driven calls) are idempotent
  within a step.
* ``wait_for_save`` submits every pending store and drains the adapter; it is
  the only place that blocks on stores.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

# Third Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

# First Party
from lmcache_ascend.integration.vllm.dsv41_lw_v2.plan import (
    GroupedTransferPlan,
    TransferGroup,
)

if TYPE_CHECKING:
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey


DSV41_LW_V2_CONNECTOR_NAME = "DSV41LwV2Connector"
DSV41_LW_V2_MODULE_PATH = (
    "lmcache_ascend.integration.vllm.dsv41_lw_v2.connector"
)
#: New, independent switch. The v2 connector never reads the v1 flag.
DSV41_LW_V2_FLAG = "dsv41.lw_v2"
#: Retired v1 flag; its presence is a configuration error even when false.
DSV41_LAYERWISE_FLAG = "dsv41.layerwise"

# Incompatible stock features kept from the v1 prototype: both would put
# unregistered tensors (intermediate tensors) or speculative tokens into the
# paged KV path that v2 plans against.
DSV41_LW_V2_INCOMPATIBLE_FLAGS = (
    "lmcache.mp.transfer_intermediate_tensors",
)

#: Sentinel meaning "use the production adapter factory from
#: :mod:`lmcache_ascend.integration.vllm.dsv41_lw_v2.production`. It is
#: resolved lazily so importing the connector does not import torch/msgspec
#: data-plane modules on the scheduler.
_USE_DEFAULT_ADAPTER_FACTORY = object()

_LOAD = "load"
_STORE = "store"
_DIRECTIONS = (_LOAD, _STORE)


class DSV41LwV2ConfigError(ValueError):
    """Invalid ``dsv41.lw_v2`` configuration (fail closed at startup)."""


class DSV41LwV2NotReadyError(RuntimeError):
    """A v2 hook needs a seam (plan, adapter, boundary codec) that is unset.

    The v2 connector must never silently degrade to stock whole-tensor
    behavior, so an unbound seam is a hard error rather than a no-op.
    """


class DSV41GroupLoadError(RuntimeError):
    """A group load failed and the request's inference must be invalidated."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str,
        group_id: str,
        resource_id: str | None = None,
        error_code: str = "",
    ) -> None:
        super().__init__(
            f"{message} (request_id={request_id!r}, group_id={group_id!r}, "
            f"resource_id={resource_id!r}, error_code={error_code!r})"
        )
        self.request_id = request_id
        self.group_id = group_id
        self.resource_id = resource_id
        self.error_code = error_code


def _non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(
            f"{field_name} must be a non-negative int, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def _non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _merge_id_sets(
    base: object, extra: object
) -> set[str] | None:
    """Union a stock terminal set with the v2 adapter's set.

    ``None`` from the stock connector means "nothing new this step"; keep it
    untouched when the adapter has nothing to add so the stock scheduler sees
    exactly the same value as before.
    """
    extra_ids = set(extra) if extra else set()
    if not extra_ids:
        return base  # type: ignore[return-value]
    base_ids = set(base) if base else set()
    return base_ids | extra_ids


@dataclass(frozen=True)
class TransferRequest:
    """One request's transfer-relevant state for one scheduler step.

    ``stock_key`` is the stock cache identity of the request's full prompt
    prefix; ``load_prefix_end`` / ``store_prefix_end`` select the sub-range of
    that prefix this step loads or stores. Requests with both endpoints zero
    are not part of a step view.
    """

    request_id: str
    step_id: int
    stock_key: "IPCCacheServerKey"
    load_prefix_end: int = 0
    store_prefix_end: int = 0
    block_ids_by_engine_group: tuple[tuple[int, ...], ...] = ()
    skip_first_n_tokens: int = 0
    # First token of this step's STORE window. The scheduler passes only the
    # blocks that are new for this step, so staging must address them relative
    # to this start instead of the request's absolute (cumulative) store end.
    store_start: int = 0

    def __post_init__(self) -> None:
        _non_empty_str(self.request_id, "request_id")
        _non_negative_int(self.step_id, "step_id")
        if self.stock_key is None:
            raise ValueError(
                f"request {self.request_id} has no stock cache key; a v2 "
                "transfer must use the stock key rather than a side lookup"
            )
        _non_negative_int(self.load_prefix_end, "load_prefix_end")
        _non_negative_int(self.store_prefix_end, "store_prefix_end")
        _non_negative_int(self.skip_first_n_tokens, "skip_first_n_tokens")
        _non_negative_int(self.store_start, "store_start")
        if not self.load_prefix_end and not self.store_prefix_end:
            raise ValueError(
                f"request {self.request_id} has neither a load nor a store "
                "prefix endpoint"
            )
        if self.store_prefix_end == 0:
            if self.store_start:
                raise ValueError(
                    f"request {self.request_id} has a store_start "
                    f"{self.store_start} without a store prefix"
                )
        elif self.store_start > self.store_prefix_end:
            raise ValueError(
                f"request {self.request_id} store_start "
                f"{self.store_start} exceeds store_prefix_end "
                f"{self.store_prefix_end}"
            )
        # Accept the stock flattened single-engine-group form and normalize it
        # to one tuple per engine group.
        object.__setattr__(
            self,
            "block_ids_by_engine_group",
            _normalize_block_ids(self.block_ids_by_engine_group),
        )


@dataclass(frozen=True)
class StepView:
    """All requests of one scheduler step, as seen by the v2 router."""

    step_id: int
    registration_epoch: int
    server_generation: int
    synthetic: bool
    requests: tuple[TransferRequest, ...] = ()

    def __post_init__(self) -> None:
        _non_negative_int(self.step_id, "step_id")
        _non_negative_int(self.registration_epoch, "registration_epoch")
        _non_negative_int(self.server_generation, "server_generation")
        if type(self.synthetic) is not bool:
            raise ValueError("synthetic must be a bool")
        if not isinstance(self.requests, tuple):
            raise ValueError("requests must be a tuple of TransferRequest")
        seen: set[str] = set()
        for request in self.requests:
            if not isinstance(request, TransferRequest):
                raise ValueError(
                    f"requests must contain TransferRequest, got "
                    f"{type(request).__name__}"
                )
            if request.request_id in seen:
                raise ValueError(
                    f"duplicate request in step view: {request.request_id}"
                )
            seen.add(request.request_id)
            if request.step_id != self.step_id:
                raise ValueError(
                    f"request {request.request_id} belongs to step "
                    f"{request.step_id}, not {self.step_id}"
                )


@dataclass(frozen=True)
class GroupOpKey:
    """Stable identity of one group operation for one request.

    ``direction`` is explicit: a load and a store of the same request/group
    step are different operations and must never share terminal state.
    """

    registration_epoch: int
    server_generation: int
    request_id: str
    step_id: int
    group_id: str
    direction: Literal["load", "store"]

    def __post_init__(self) -> None:
        _non_negative_int(self.registration_epoch, "registration_epoch")
        _non_negative_int(self.server_generation, "server_generation")
        _non_empty_str(self.request_id, "request_id")
        _non_negative_int(self.step_id, "step_id")
        _non_empty_str(self.group_id, "group_id")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_DIRECTIONS!r}, got "
                f"{self.direction!r}"
            )


@runtime_checkable
class DSV41LwV2Adapter(Protocol):
    """Transport adapter seam implemented by T4.

    Every method receives the request ids of one group and the step view, so
    the adapter can derive resource sub-keys from the stock key carried by
    each request. The adapter owns terminal state and reports it through
    ``take_finished`` / ``take_failed_receiving``; the connector only routes.
    """

    def schedule_group_retrieves(
        self, group_id: str, request_ids: Sequence[str], step: StepView
    ) -> None: ...

    def wait_group(
        self,
        group_id: str,
        request_ids: Sequence[str],
        step: StepView,
        *,
        stream: Any = None,
    ) -> None: ...

    def submit_group_store(
        self, group_id: str, request_ids: Sequence[str], step: StepView
    ) -> None: ...

    def drop_request(self, request_id: str, *, reason: str) -> None: ...

    def drain_stores(self) -> None: ...

    def take_finished(
        self, *, finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]: ...

    def take_failed_receiving(self) -> set[str]: ...


@runtime_checkable
class DSV41LwV2BoundaryHook(Protocol):
    """Boundary capture/restore seam implemented by the T3 codec."""

    def capture_group_boundary(self, group_id: str) -> None: ...

    def restore_group_boundary(self, group_id: str) -> None: ...


@dataclass
class _GroupStepOps:
    """Per-group, per-step routing state (idempotency bookkeeping)."""

    load_request_ids: tuple[str, ...] = ()
    store_request_ids: tuple[str, ...] = ()
    scheduled: bool = False
    waited: bool = False
    boundary_restored: bool = False
    boundary_captured: bool = False
    submitted: bool = False


class GroupTransferRouter:
    """Route one step's requests to the four groups, in contract order.

    The router never performs I/O itself. It owns the idempotency of the
    group facades and the fail-closed checks for missing seams, so the same
    group cannot be waited/stored twice or skipped accidentally.
    """

    def __init__(
        self,
        plan: GroupedTransferPlan,
        *,
        adapter: "DSV41LwV2Adapter | None" = None,
        boundary_hook: "DSV41LwV2BoundaryHook | None" = None,
        registration_epoch: int = 0,
        server_generation: int = 0,
    ) -> None:
        if not isinstance(plan, GroupedTransferPlan):
            raise ValueError(
                "plan must be a GroupedTransferPlan, got "
                f"{type(plan).__name__}"
            )
        self._plan = plan
        self._adapter = adapter
        self._boundary_hook = boundary_hook
        self._registration_epoch = _non_negative_int(
            registration_epoch, "registration_epoch"
        )
        self._server_generation = _non_negative_int(
            server_generation, "server_generation"
        )
        self._step: StepView | None = None
        self._ops: dict[str, _GroupStepOps] = {}
        # Store batches submitted to the adapter and not drained yet. The
        # save hook must drain if any are outstanding, even when the current
        # step itself carries no store work.
        self._pending_stores = 0
        self._counters: dict[str, int] = {
            "group_load_scheduled": 0,
            "group_load_ready": 0,
            "group_store_submitted": 0,
            "group_store_drained": 0,
            "boundary_captured": 0,
            "boundary_restored": 0,
            "requests_dropped": 0,
            "stale_ops_cleared": 0,
        }

    @property
    def plan(self) -> GroupedTransferPlan:
        return self._plan

    @property
    def metrics(self) -> dict[str, int]:
        return dict(self._counters)

    @property
    def current_step(self) -> StepView | None:
        return self._step

    @property
    def adapter(self) -> "DSV41LwV2Adapter | None":
        return self._adapter

    def bind_adapter(self, adapter: DSV41LwV2Adapter) -> None:
        if not isinstance(adapter, DSV41LwV2Adapter):
            raise ValueError("adapter does not implement DSV41LwV2Adapter")
        self._adapter = adapter

    def bind_boundary_hook(self, hook: DSV41LwV2BoundaryHook) -> None:
        if not isinstance(hook, DSV41LwV2BoundaryHook):
            raise ValueError(
                "hook does not implement DSV41LwV2BoundaryHook"
            )
        self._boundary_hook = hook

    def note_step(self, step: StepView) -> None:
        """Queue one retrieve per group for this step's real load hits.

        No inline wait happens here. A synthetic/profiling step still updates
        the generation state but schedules nothing.
        """
        if not isinstance(step, StepView):
            raise ValueError(
                f"step must be a StepView, got {type(step).__name__}"
            )
        if self._step is not None:
            if step.step_id == self._step.step_id:
                return
            if step.step_id < self._step.step_id:
                raise ValueError(
                    f"step_id moved backwards: {step.step_id} < "
                    f"{self._step.step_id}"
                )
        if (
            step.registration_epoch != self._registration_epoch
            or step.server_generation != self._server_generation
        ):
            self.set_generations(
                registration_epoch=step.registration_epoch,
                server_generation=step.server_generation,
            )
        # A new step supersedes any unfinished routing state; terminal
        # operation state itself lives in the adapter.
        self.finish_step()
        self._step = step
        if step.synthetic:
            return

        load_ids = tuple(
            request.request_id
            for request in step.requests
            if request.load_prefix_end > 0
        )
        store_ids = tuple(
            request.request_id
            for request in step.requests
            if request.store_prefix_end > 0
        )
        if load_ids and self._adapter is None:
            raise DSV41LwV2NotReadyError(
                "step has a real external load hit but no v2 transport "
                "adapter is bound; refusing to fall back to stock "
                "whole-tensor retrieval"
            )
        for group in self._plan.groups:
            if not load_ids and not store_ids:
                continue
            ops = _GroupStepOps(
                load_request_ids=load_ids,
                store_request_ids=store_ids,
                scheduled=bool(load_ids),
            )
            self._ops[group.group_id] = ops
            if load_ids:
                assert self._adapter is not None  # checked above
                self._adapter.schedule_group_retrieves(
                    group.group_id, load_ids, step
                )
                self._counters["group_load_scheduled"] += 1

    def set_generations(
        self, *, registration_epoch: int, server_generation: int
    ) -> None:
        """Advance the generation and drop every pending op immediately.

        The drop must not wait for the next ``note_step``: ``flush_step`` (run
        by the next ``start_load_kv``) would otherwise close the previous step
        first and the stale request ids would never reach the adapter.
        """
        registration_epoch = _non_negative_int(
            registration_epoch, "registration_epoch"
        )
        server_generation = _non_negative_int(
            server_generation, "server_generation"
        )
        if (
            registration_epoch == self._registration_epoch
            and server_generation == self._server_generation
        ):
            return
        self._clear_stale_ops()
        self._registration_epoch = registration_epoch
        self._server_generation = server_generation

    def _clear_stale_ops(self) -> None:
        stale = [
            request_id
            for ops in self._ops.values()
            for request_id in (
                ops.load_request_ids + ops.store_request_ids
            )
        ]
        for request_id in dict.fromkeys(stale):
            if self._adapter is not None:
                self._adapter.drop_request(
                    request_id, reason="generation_changed"
                )
        self._counters["stale_ops_cleared"] += len(self._ops)
        self._ops.clear()
        self._pending_stores = 0
        self._step = None

    def operation_keys(self, direction: str) -> tuple[GroupOpKey, ...]:
        """Return the per-request operation identities of the current step."""
        if direction not in _DIRECTIONS:
            raise ValueError(f"unknown direction: {direction!r}")
        if self._step is None:
            return ()
        keys: list[GroupOpKey] = []
        for group in self._plan.groups:
            ops = self._ops.get(group.group_id)
            if ops is None:
                continue
            request_ids = (
                ops.load_request_ids
                if direction == _LOAD
                else ops.store_request_ids
            )
            for request_id in request_ids:
                keys.append(
                    GroupOpKey(
                        registration_epoch=self._registration_epoch,
                        server_generation=self._server_generation,
                        request_id=request_id,
                        step_id=self._step.step_id,
                        group_id=group.group_id,
                        direction=direction,  # type: ignore[arg-type]
                    )
                )
        return tuple(keys)

    def _require_group(self, group_id: str) -> TransferGroup:
        """Look up a group by id (KeyError for unknown ids)."""
        return self._plan.group_by_id(group_id)

    def wait_group_load(
        self, group_id: str, *, stream: Any = None
    ) -> None:
        group = self._require_group(group_id)
        ops = self._ops.get(group.group_id)
        if ops is None or not ops.load_request_ids or ops.waited:
            return
        step = self._step
        if step is None:
            return
        if self._adapter is None:
            raise DSV41LwV2NotReadyError(
                f"no v2 transport adapter bound for group {group_id}"
            )
        self._adapter.wait_group(
            group.group_id, ops.load_request_ids, step, stream=stream
        )
        ops.waited = True
        self._counters["group_load_ready"] += 1

    def restore_group_boundary(self, group_id: str) -> None:
        group = self._require_group(group_id)
        ops = self._ops.get(group.group_id)
        if ops is None or not ops.load_request_ids or not group.has_boundary:
            return
        if ops.boundary_restored:
            return
        if not ops.waited:
            raise DSV41LwV2NotReadyError(
                f"group {group_id} boundary cannot be restored before its "
                "load is ready"
            )
        if self._boundary_hook is None:
            raise DSV41LwV2NotReadyError(
                f"group {group_id} has a real boundary hit but no boundary "
                "codec is bound (BOUNDARY_NOT_BOUND)"
            )
        self._boundary_hook.restore_group_boundary(group.group_id)
        ops.boundary_restored = True
        self._counters["boundary_restored"] += 1

    def capture_group_boundary(self, group_id: str) -> None:
        group = self._require_group(group_id)
        ops = self._ops.get(group.group_id)
        if ops is None or not ops.store_request_ids or not group.has_boundary:
            return
        if ops.boundary_captured:
            return
        if self._boundary_hook is None:
            raise DSV41LwV2NotReadyError(
                f"group {group_id} must store a boundary but no boundary "
                "codec is bound (BOUNDARY_NOT_BOUND)"
            )
        self._boundary_hook.capture_group_boundary(group.group_id)
        ops.boundary_captured = True
        self._counters["boundary_captured"] += 1

    def submit_group_store(self, group_id: str) -> None:
        group = self._require_group(group_id)
        ops = self._ops.get(group.group_id)
        if ops is None or not ops.store_request_ids or ops.submitted:
            return
        step = self._step
        if step is None:
            return
        if self._adapter is None:
            raise DSV41LwV2NotReadyError(
                f"no v2 transport adapter bound for group {group_id}"
            )
        self._adapter.submit_group_store(
            group.group_id, ops.store_request_ids, step
        )
        ops.submitted = True
        self._pending_stores += 1
        self._counters["group_store_submitted"] += 1


    def wait_for_save(self) -> None:
        """Drain stores submitted by the previous forward, keep this step.

        vLLM calls ``wait_for_save`` *before* the forward that
        ``start_load_kv`` just prepared (it waits for the previous step's
        saves).  Finishing the routing step here would erase the live
        per-group ops before the post-forward boundary capture/group store
        hooks run, which is exactly the t13 F5 failure.  The step therefore
        stays alive until the next ``start_load_kv`` (or ``get_finished``).
        """
        self.drain_pending_stores()

    def flush_step(self) -> None:
        """Close the current step after its forward and store submission."""
        if self._step is None:
            return
        for group in self._plan.groups:
            self.submit_group_store(group.group_id)
        self.drain_pending_stores()
        self.finish_step()

    def drain_pending_stores(self) -> None:
        """Drain already-submitted store batches without touching step ops."""
        if self._adapter is not None and self._pending_stores > 0:
            self._adapter.drain_stores()
            self._pending_stores = 0
            self._counters["group_store_drained"] += 1

    def finish_step(self) -> None:
        self._ops.clear()
        self._step = None

    def drop_request(self, request_id: str, *, reason: str) -> None:
        _non_empty_str(request_id, "request_id")
        if self._adapter is not None:
            self._adapter.drop_request(
                request_id, reason=reason or "dropped"
            )
        self._counters["requests_dropped"] += 1
        for group_id in list(self._ops):
            ops = self._ops[group_id]
            load_remaining = tuple(
                rid for rid in ops.load_request_ids if rid != request_id
            )
            store_remaining = tuple(
                rid for rid in ops.store_request_ids if rid != request_id
            )
            if not load_remaining and not store_remaining:
                del self._ops[group_id]
                continue
            ops.load_request_ids = load_remaining
            ops.store_request_ids = store_remaining


def connector_extra_config(
    vllm_config: Any,
) -> Mapping[str, object]:
    """Return the connector extra config mapping, or an empty mapping."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    extra_config = getattr(
        kv_transfer_config, "kv_connector_extra_config", None
    )
    if extra_config is None:
        return {}
    if not isinstance(extra_config, Mapping):
        raise DSV41LwV2ConfigError(
            "kv_connector_extra_config must be a mapping, got "
            f"{type(extra_config).__name__}"
        )
    return extra_config


def parse_lw_v2_flag(
    extra_config: Mapping[str, object] | None,
) -> bool:
    """Validate the dedicated ``dsv41.lw_v2`` switch.

    The dedicated connector is only ever selected explicitly, so it requires
    the JSON boolean ``true``. Missing/false values are configuration errors
    (never a silent stock fallback), and the retired ``dsv41.layerwise`` flag
    must not be present at all -- the old connector owns it.
    """
    config: Mapping[str, object] = (
        {} if extra_config is None else extra_config
    )
    if not isinstance(config, Mapping):
        raise DSV41LwV2ConfigError(
            "extra_config must be a mapping, got "
            f"{type(config).__name__}"
        )
    if DSV41_LAYERWISE_FLAG in config:
        raise DSV41LwV2ConfigError(
            f"{DSV41_LAYERWISE_FLAG} is retired for the v2 connector; "
            "remove it (the v1 dsv41_layerwise connector keeps its own flag)"
        )
    if DSV41_LW_V2_FLAG not in config:
        raise DSV41LwV2ConfigError(
            f"{DSV41_LW_V2_FLAG} must be set to the JSON boolean true when "
            f"using {DSV41_LW_V2_CONNECTOR_NAME}"
        )
    value = config[DSV41_LW_V2_FLAG]
    if type(value) is not bool or value is not True:
        raise DSV41LwV2ConfigError(
            f"{DSV41_LW_V2_FLAG} must be the JSON boolean true, got "
            f"{type(value).__name__}: {value!r}"
        )
    return True


def require_lw_v2_enabled(extra_config: Mapping[str, object] | None) -> bool:
    """Alias with an explicit name for the connector constructor."""
    return parse_lw_v2_flag(extra_config)


def _reject_incompatible_features(
    extra_config: Mapping[str, object], vllm_config: Any
) -> None:
    for flag in DSV41_LW_V2_INCOMPATIBLE_FLAGS:
        if extra_config.get(flag, False):
            raise DSV41LwV2ConfigError(
                f"{flag} is incompatible with {DSV41_LW_V2_CONNECTOR_NAME}"
            )
    speculative_config = getattr(vllm_config, "speculative_config", None)
    speculative_tokens = getattr(
        speculative_config, "num_speculative_tokens", 0
    )
    if speculative_tokens:
        raise DSV41LwV2ConfigError(
            "speculative decoding is incompatible with the layerwise v2 "
            "group transfer (the replayed prefix must be a real prefix)"
        )


class DSV41LwV2Connector(LMCacheMPConnector):
    """Stock connector subclass that routes by KV-source group only.

    Stock construction, registration, scheduler lookup and request cleanup are
    inherited unchanged. The four layerwise hooks below are overridden so they
    never reach the stock whole-tensor retrieve/store paths; all data-plane
    work is delegated to a bound :class:`DSV41LwV2Adapter` (T4) and boundary
    codec (T3).
    """

    CONNECTOR_NAME = DSV41_LW_V2_CONNECTOR_NAME

    def __init__(
        self,
        vllm_config: Any,
        role: Any,
        kv_cache_config: Any = None,
    ) -> None:
        extra_config = connector_extra_config(vllm_config)
        # Fail fast, before the stock constructor starts any handshake.
        require_lw_v2_enabled(extra_config)
        _reject_incompatible_features(extra_config, vllm_config)
        super().__init__(vllm_config, role, kv_cache_config)
        self._lw_v2_plan: GroupedTransferPlan | None = None
        self._lw_v2_router: GroupTransferRouter | None = None
        self._lw_v2_stock_key_factory: Any = None
        self._lw_v2_step_serial = 0
        self._lw_v2_registration_epoch = 0
        self._lw_v2_server_generation = 0
        self._lw_v2_current_step: StepView | None = None
        self._lw_v2_step_noted = False
        self._lw_v2_boundary_codec: Any = None
        self._lw_v2_adapter_factory: Any = _USE_DEFAULT_ADAPTER_FACTORY

    # ------------------------------------------------------------------
    # Wiring seams (T4 builds the adapter on registration, T3 the boundary)
    # ------------------------------------------------------------------

    def bind_plan(
        self, plan: GroupedTransferPlan, *, replace: bool = False
    ) -> None:
        if not isinstance(plan, GroupedTransferPlan):
            raise DSV41LwV2ConfigError(
                "plan must be a GroupedTransferPlan, got "
                f"{type(plan).__name__}"
            )
        if self._lw_v2_plan is not None and not replace:
            if self._lw_v2_plan is plan:
                return
            raise DSV41LwV2ConfigError(
                "a grouped plan is already bound; pass replace=True after a "
                "new registration epoch"
            )
        self._lw_v2_plan = plan
        self._lw_v2_router = GroupTransferRouter(plan)

    def bind_adapter(self, adapter: DSV41LwV2Adapter) -> None:
        self._require_router().bind_adapter(adapter)

    def bind_boundary_hook(self, hook: DSV41LwV2BoundaryHook) -> None:
        self._require_router().bind_boundary_hook(hook)

    def bind_stock_key_factory(self, factory: Any) -> None:
        """Bind the stock cache-key factory used to build ``TransferRequest``.

        The factory signature is ``factory(*, request_id, token_bytes, start,
        end, cache_salt) -> IPCCacheServerKey`` and must come from stock key
        resolution (same key used for lookup and resource transport).
        """
        if factory is not None and not callable(factory):
            raise DSV41LwV2ConfigError("stock key factory must be callable")
        self._lw_v2_stock_key_factory = factory

    @property
    def grouped_plan(self) -> GroupedTransferPlan | None:
        return self._lw_v2_plan

    @property
    def current_step(self) -> StepView | None:
        """The step most recently queued by ``start_load_kv`` (read-only).

        The engine-side boundary hook uses this to obtain the real hit/store
        sets instead of guessing from a prefix. It is cleared after
        ``wait_for_save``.
        """
        return self._lw_v2_current_step

    @property
    def boundary_codec(self) -> Any:
        """The installed schema-2 boundary codec, or ``None`` when unbound."""
        return self._lw_v2_boundary_codec

    def set_boundary_codec(self, codec: Any) -> None:
        """Install the single production boundary codec.

        Called by ``install_dsv41_lw_v2_boundary_codec`` during initialization,
        before the first real forward. Installing a second, different codec
        instance is a configuration error: the engine binds exactly one.
        """
        if codec is None:
            raise DSV41LwV2ConfigError("boundary codec must not be None")
        capture = getattr(codec, "capture_layer_boundary", None)
        restore = getattr(codec, "restore_layer_boundary", None)
        if not callable(capture) or not callable(restore):
            raise DSV41LwV2ConfigError(
                "boundary codec must implement capture_layer_boundary and "
                "restore_layer_boundary"
            )
        existing = self._lw_v2_boundary_codec
        if existing is not None and existing is not codec:
            raise DSV41LwV2ConfigError(
                "a different boundary codec is already installed; the v2 "
                "engine binds exactly one codec instance"
            )
        self._lw_v2_boundary_codec = codec

    def bind_adapter_factory(self, factory: Any) -> None:
        """Override the factory that builds the group adapter on registration.

        Signature: ``factory(plan, connector) -> DSV41LwV2Adapter``. The
        production factory is installed by default; passing ``None`` disables
        it explicitly, which makes ``register_kv_caches`` fail closed instead
        of silently falling back to stock full-tensor behavior. Tests that
        bind a prebuilt adapter directly do not need this seam.
        """
        if factory is not None and not callable(factory):
            raise DSV41LwV2ConfigError("adapter factory must be callable")
        self._lw_v2_adapter_factory = factory

    @property
    def adapter(self) -> Any:
        """The currently bound group adapter, or ``None``."""
        router = self._lw_v2_router
        return None if router is None else router.adapter

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Run stock cache registration, then build the v2 group runtime.

        The stock registration derives the edited cache views and registers
        the layer layout with the LMCache server. Once that has happened the
        optional adapter factory builds the group adapter over the edited
        caches and the connector installs the schema-2 boundary codec, so a
        ready engine finds both before the first forward.

        Registration itself does not open a step: ``current_step`` is ``None``
        here and is filled by each step's ``start_load_kv`` and cleared by
        ``wait_for_save``. The engine seam therefore reads a step (and the
        schema-2 ``boundary_codec``, installed at the end of this method) only
        during the forward phase.

        The production factory is the default. A missing dependency (no
        worker adapter, no registered caches, no multiprocess request client,
        no chunk geometry, no grouped plan) raises ``NotReady`` naming the
        dependency: the connector never silently falls back to the stock
        full-tensor path.
        """
        super().register_kv_caches(kv_caches)
        if self.adapter is not None:
            # Repeated stock registration (heartbeats) is idempotent: the
            # adapter and its single codec are built exactly once.
            return
        # First Party
        from lmcache_ascend.integration.vllm.dsv41_lw_v2 import production

        factory = self._lw_v2_adapter_factory
        if factory is _USE_DEFAULT_ADAPTER_FACTORY:
            factory = production.DEFAULT_ADAPTER_FACTORY
        if factory is None:
            raise DSV41LwV2NotReadyError(
                "no layerwise v2 adapter factory is bound; refusing to fall "
                "back to the stock full-tensor transfer path"
            )
        plan = self._lw_v2_plan
        if plan is None:
            plan = self._build_plan_from_registration()
            self.bind_plan(plan)
        try:
            adapter = factory(plan, self)
        except production.DSV41LwV2ProductionUnavailable as exc:
            raise DSV41LwV2NotReadyError(
                "the layerwise v2 adapter cannot be built: "
                f"{exc.dependency} is unavailable ({exc}); bind the missing "
                "dependency before registering KV caches"
            ) from exc
        self.bind_adapter(adapter)
        if self._lw_v2_boundary_codec is not None:
            return
        if not callable(getattr(adapter, "stage_boundary_capture", None)):
            return
        # First Party
        from lmcache_ascend.integration.vllm.dsv41_lw_v2.boundary import (
            install_dsv41_lw_v2_boundary_codec,
        )

        install_dsv41_lw_v2_boundary_codec(self, adapter=adapter)

    def _build_plan_from_registration(self) -> GroupedTransferPlan:
        """Derive the grouped plan from this worker's stock registration."""
        # First Party
        from lmcache_ascend.integration.vllm.dsv41_lw_v2 import production

        worker_adapter = getattr(self, "worker_adapter", None)
        if worker_adapter is None:
            raise DSV41LwV2NotReadyError(
                "cannot derive the grouped plan: worker_adapter is not "
                "initialized (the scheduler role does not register caches)"
            )
        kv_caches = getattr(worker_adapter, "kv_caches", None)
        if not kv_caches:
            raise DSV41LwV2NotReadyError(
                "cannot derive the grouped plan: worker_adapter.kv_caches is "
                "empty after stock registration"
            )
        engine_group_infos = (
            getattr(worker_adapter, "engine_group_infos", None) or ()
        )
        try:
            return production.build_grouped_plan_from_registration(
                kv_caches, engine_group_infos
            )
        except production.DSV41LwV2ProductionUnavailable as exc:
            raise DSV41LwV2NotReadyError(
                f"cannot derive the grouped plan: {exc.dependency} is "
                f"unavailable ({exc})"
            ) from exc

    @property
    def metrics(self) -> dict[str, int]:
        router = self._lw_v2_router
        return router.metrics if router is not None else {}

    def group_for_layer(self, layer_name: str) -> TransferGroup | None:
        """Pure mapping query; non-head and unknown layers return ``None``."""
        return self._require_router().plan.group_for_layer(layer_name)

    # ------------------------------------------------------------------
    # vLLM hooks (only group heads do work; non-heads are pure queries)
    # ------------------------------------------------------------------

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Queue one retrieve per group; never wait and never copy inline."""
        router = self._require_router()
        # The previous forward's captures already happened; submit any store
        # it did not hand off and drain those batches before blocks are
        # reused.  This is the "wait for the previous step's saves" point.
        router.flush_step()
        metadata = getattr(self, "_connector_metadata", None)
        if metadata is None:
            return
        step = self._build_step_view(metadata)
        if step is None:
            # No retrieves and no stores: there is no live v2 step, so no
            # boundary may be captured or restored for it.
            self._lw_v2_step_noted = False
            self._lw_v2_current_step = None
            return
        self._lw_v2_step_noted = True
        self._lw_v2_current_step = step
        router.note_step(step)

    def wait_for_layer_load(self, layer_name: str) -> None:
        router = self._require_router()
        group = router.plan.group_for_layer(layer_name)
        if group is None:
            # Non-head layer: mapping query only, no adapter/codec/counter.
            return
        self.wait_group_load(group.group_id)
        self.restore_group_boundary(group.group_id)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        router = self._require_router()
        group = router.plan.group_for_layer(layer_name)
        if group is None:
            return
        self.capture_group_boundary(group.group_id)
        self.submit_group_store(group.group_id)

    def wait_for_save(self) -> None:
        """Wait for the previous forward's saves, keeping the live step.

        The current step's captures/submits happen later in the forward; this
        hook must not submit or finish them (see ``GroupTransferRouter.
        wait_for_save``).
        """
        router = self._require_router()
        router.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Merge v2 terminal request ids into the stock connector report.

        The v2 adapter settles each direction asynchronously (retrieves are
        waited per group head, stores are drained in ``wait_for_save``), so
        the stock connector alone never learns about them.  Returning the
        merged ``(finished_sending, finished_receiving)`` pair keeps the stock
        scheduler's request lifecycle intact.
        """
        router = self._lw_v2_router
        if router is not None and router.current_step is not None:
            # The forward and its post-forward captures are done: hand off any
            # remaining group stores and drain them before reporting terminal
            # requests.  This is the only point where the routing step closes.
            router.flush_step()
        self._lw_v2_step_noted = False
        self._lw_v2_current_step = None
        sending, receiving = super().get_finished(finished_req_ids)
        adapter = self.adapter
        if adapter is None:
            return sending, receiving
        # Stock contract: a finished id may only be reported when the request
        # finished (this call or a prior one); the adapter keeps the rest.
        v2_sending, v2_receiving = adapter.take_finished(
            finished_req_ids=finished_req_ids
        )
        return (
            _merge_id_sets(sending, v2_sending),
            _merge_id_sets(receiving, v2_receiving),
        )

    def get_transfer_results(self, finished_req_ids: set[str]) -> Any:
        """Report v2 load failures as ``failed_recving`` so vLLM recomputes.

        A group load that failed after the scheduler had already skipped the
        prefix must not be treated as a completed receive: the request has to
        take the ``BYPASS_LMCACHE`` recompute path with clean pages.
        """
        results = super().get_transfer_results(finished_req_ids)
        adapter = self.adapter
        if adapter is None:
            return results
        failed = adapter.take_failed_receiving()
        if failed:
            results.failed_recving = _merge_id_sets(
                results.failed_recving, failed
            )
        return results

    # ------------------------------------------------------------------
    # Explicit group facades (T3 calls these directly; idempotent with hooks)
    # ------------------------------------------------------------------

    def wait_group_load(
        self, group_id: str, *, stream: Any = None
    ) -> None:
        self._require_router().wait_group_load(group_id, stream=stream)

    def restore_group_boundary(self, group_id: str) -> None:
        self._require_router().restore_group_boundary(group_id)

    def capture_group_boundary(self, group_id: str) -> None:
        self._require_router().capture_group_boundary(group_id)

    def submit_group_store(self, group_id: str) -> None:
        self._require_router().submit_group_store(group_id)

    def drop_request(self, request_id: str, *, reason: str = "dropped") -> None:
        router = self._lw_v2_router
        if router is None:
            return
        router.drop_request(request_id, reason=reason)

    # ------------------------------------------------------------------
    # Stock metadata -> StepView
    # ------------------------------------------------------------------

    def _require_router(self) -> GroupTransferRouter:
        router = self._lw_v2_router
        if router is None:
            raise DSV41LwV2NotReadyError(
                "no grouped plan is bound; the v2 connector must bind one "
                "from the live registration before any forward hook"
            )
        return router

    def _build_step_view(self, metadata: Any) -> StepView | None:
        entries = getattr(metadata, "requests", None)
        if entries is None:
            raise ValueError(
                "connector metadata has no requests field; the v2 connector "
                "requires stock LMCacheMPConnectorMetadata"
            )
        if isinstance(entries, (str, bytes)) or not isinstance(
            entries, Sequence
        ):
            raise ValueError("connector metadata requests must be a sequence")
        synthetic = getattr(metadata, "synthetic", False)
        if type(synthetic) is not bool:
            raise ValueError("connector metadata synthetic flag must be a bool")

        merged: dict[str, dict[str, Any]] = {}
        for entry in entries:
            request_id = _non_empty_str(
                getattr(entry, "request_id", None), "request_id"
            )
            direction = getattr(entry, "direction", None)
            if direction not in ("RETRIEVE", "STORE"):
                raise ValueError(
                    f"request {request_id} has unsupported direction "
                    f"{direction!r}"
                )
            op = getattr(entry, "op", None)
            if op is None:
                raise ValueError(f"request {request_id} has no load/store op")
            start = _non_negative_int(getattr(op, "start", 0), "op.start")
            end = _non_negative_int(getattr(op, "end", 0), "op.end")
            if start > end:
                raise ValueError(
                    f"request {request_id} op.start {start} exceeds op.end "
                    f"{end}"
                )
            skip_first = _non_negative_int(
                getattr(op, "skip_first_n_tokens", 0),
                "op.skip_first_n_tokens",
            )
            token_bytes = getattr(op, "token_bytes", b"")
            if not isinstance(token_bytes, bytes):
                raise ValueError(
                    f"request {request_id} op.token_bytes must be bytes"
                )
            cache_salt = getattr(entry, "cache_salt", "")
            blocks = getattr(op, "block_ids", ())
            state = merged.setdefault(
                request_id,
                {
                    "token_bytes": token_bytes,
                    "cache_salt": cache_salt,
                    "block_ids": blocks,
                    "skip_first_n_tokens": skip_first,
                    "load_prefix_end": 0,
                    "store_prefix_end": 0,
                    "store_start": None,
                },
            )
            if state["token_bytes"] != token_bytes:
                raise ValueError(
                    f"request {request_id} changed token_bytes within a step"
                )
            state["skip_first_n_tokens"] = max(
                state["skip_first_n_tokens"], skip_first
            )
            if direction == "RETRIEVE":
                state["load_prefix_end"] = max(
                    state["load_prefix_end"], end
                )
                # Retrieve blocks are the destination pages; store blocks may
                # differ, so the load target wins when both are present.
                state["block_ids"] = blocks
            else:
                state["store_prefix_end"] = max(
                    state["store_prefix_end"], end
                )
                # The scheduler emits one STORE entry per new window. Keep the
                # earliest window start so the adapter can map this step's
                # block list to absolute token positions.
                if state["store_start"] is None:
                    state["store_start"] = start
                else:
                    state["store_start"] = min(state["store_start"], start)

        step_id = self._lw_v2_step_serial
        requests: list[TransferRequest] = []
        for request_id, state in merged.items():
            load_end = state["load_prefix_end"]
            store_end = state["store_prefix_end"]
            if not load_end and not store_end:
                # No external hit and nothing to store this step.
                continue
            store_start = state["store_start"] or 0
            if store_end and store_start > store_end:
                raise ValueError(
                    f"request {request_id} store_start {store_start} "
                    f"exceeds store_prefix_end {store_end} in one step"
                )
            stock_key = self._stock_key(
                request_id=request_id,
                token_bytes=state["token_bytes"],
                end=max(load_end, store_end),
                cache_salt=state["cache_salt"],
            )
            requests.append(
                TransferRequest(
                    request_id=request_id,
                    step_id=step_id,
                    stock_key=stock_key,
                    load_prefix_end=load_end,
                    store_prefix_end=store_end,
                    block_ids_by_engine_group=_normalize_block_ids(
                        state["block_ids"]
                    ),
                    skip_first_n_tokens=state["skip_first_n_tokens"],
                    store_start=store_start,
                )
            )
        if not requests and not synthetic:
            # Decode steps and requests without a new external load/store do
            # not produce a v2 step at all: every facade then no-ops.
            return None
        self._lw_v2_step_serial += 1
        return StepView(
            step_id=step_id,
            registration_epoch=self._lw_v2_registration_epoch,
            server_generation=self._lw_v2_server_generation,
            synthetic=synthetic,
            requests=tuple(requests),
        )

    def set_generations(
        self, *, registration_epoch: int, server_generation: int
    ) -> None:
        """Advance the registration/generation identity of the worker.

        Pending group operations of the retired generation are dropped right
        away (reported as ``generation_changed``) instead of waiting for the
        next step, then the adapter's own generation is advanced.
        """
        registration_epoch = _non_negative_int(
            registration_epoch, "registration_epoch"
        )
        server_generation = _non_negative_int(
            server_generation, "server_generation"
        )
        router = self._lw_v2_router
        if router is not None:
            router.set_generations(
                registration_epoch=registration_epoch,
                server_generation=server_generation,
            )
        self._lw_v2_registration_epoch = registration_epoch
        self._lw_v2_server_generation = server_generation
        adapter = self.adapter
        setter = getattr(adapter, "set_generations", None)
        if callable(setter):
            setter(
                registration_epoch=self._lw_v2_registration_epoch,
                server_generation=self._lw_v2_server_generation,
            )

    def _stock_key(
        self,
        *,
        request_id: str,
        token_bytes: bytes,
        end: int,
        cache_salt: Any,
    ) -> "IPCCacheServerKey":
        factory = self._lw_v2_stock_key_factory
        if factory is None:
            raise DSV41LwV2NotReadyError(
                f"request {request_id} has a real cache endpoint but no "
                "stock key factory is bound; v2 transfers must use the stock "
                "key, not a private index"
            )
        key = factory(
            request_id=request_id,
            token_bytes=token_bytes,
            start=0,
            end=end,
            cache_salt=cache_salt,
        )
        if key is None:
            raise DSV41LwV2NotReadyError(
                f"stock key factory returned no key for request {request_id}"
            )
        return key


def _normalize_block_ids(
    blocks: Any,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Sequence):
        raise ValueError("block_ids must be a sequence of sequences")
    if not blocks:
        return ()
    if all(type(block) is int for block in blocks):
        # Tolerate the IPC-flattened single-group form: one engine group.
        return (tuple(_non_negative_int(block, "block id") for block in blocks),)
    return tuple(
        tuple(
            _non_negative_int(block, "block id")
            for block in group_blocks
        )
        for group_blocks in blocks
    )
