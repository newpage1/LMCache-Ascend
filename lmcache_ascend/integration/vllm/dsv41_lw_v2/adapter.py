# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layerwise v2 group-level transport adapter.

This module owns the data plane of the v2 design:

* one :class:`GroupEvent` per (group, request, direction) that carries the
  terminal state (``ok`` / ``drop`` / ``timeout`` / ``failed``) of a group
  operation; the engine-side ``wait_group`` either sees it already resolved or
  records a ``host_wait_fallback``;
* a single background worker per adapter for blocking work (device to host
  copies, payload encoding, blocking RPC replies) and one copy stream for
  device writes, so neither the vLLM forward thread nor the connector hook
  performs large ``.cpu()`` / ``.tolist()`` calls;
* the neutral ``resource_blocks_v1`` wire DTOs and a thin
  :class:`MPResourceBlockTransport` wrapper over the stock request client;
* group-level retrieve/store, per-group drop, request drop, stale-generation
  cleanup, ``drain_stores`` and the ``take_finished`` /
  ``take_failed_receiving`` reports the stock connector consumes.

The adapter never invents cache identity: every resource spec carries the
request's ``IPCCacheServerKey`` and the resource id from the registered plan.
``resource_blocks_v1`` is engine neutral; the concrete schema-2 boundary
envelope lives in :mod:`lmcache_ascend.integration.vllm.dsv41_lw_v2.boundary`.

Completion semantics (``docs/v2-api-contract.md`` section 4):

* a load is ``ready`` only after every resource result of the group has been
  validated and every device write is visible to the consumer stream;
* a store is ``committed`` only after the server acknowledged the formal
  write, never after a mere submit;
* a drop is terminal: a late RPC callback can no longer turn the operation
  into ``ready``/``committed``, and a buffer that has an in-flight device copy
  is released only after that copy completed.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
import json
import queue
import struct
import threading
import time
import uuid

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.layerwise import CacheTransferResource
from lmcache_ascend.integration.vllm.dsv41_lw_v2.plan import (
    GroupedTransferPlan,
    TransferGroup,
)
from lmcache_ascend.integration.vllm.dsv41_lw_v2.connector import (
    DSV41GroupLoadError,
    StepView,
    TransferRequest,
)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

RESOURCE_BLOCK_PROTOCOL_VERSION = 1
RESOURCE_BLOCK_LAYOUT_ENCODINGS = ("token_range_v1", "boundary_schema2")
REQUIRED_CAPABILITIES = (
    "resource_blocks_v1",
    "stock_prefix_visibility_v1",
    "boundary_schema2",
)

TOKEN_PAYLOAD_MAGIC = b"DSV2TR1\n"
BOUNDARY_PAYLOAD_MAGIC = b"DSV2BC1\n"

STATUS_OK = "ok"
STATUS_MISS = "miss"
STATUS_INVALID = "invalid"
STATUS_FAILED = "failed"

ALREADY_PRESENT = "ALREADY_PRESENT"
RESOURCE_MISSING = "RESOURCE_MISSING"
PREFIX_INCOMPLETE = "PREFIX_INCOMPLETE"
INVALID_SPEC = "INVALID_SPEC"
STALE_EPOCH = "STALE_EPOCH"
STALE_GENERATION = "STALE_GENERATION"
LAYOUT_MISMATCH = "LAYOUT_MISMATCH"
PLAN_MISMATCH = "PLAN_MISMATCH"
UNSUPPORTED_BOUNDARY_SCHEMA = "UNSUPPORTED_BOUNDARY_SCHEMA"
PAYLOAD_CORRUPT = "PAYLOAD_CORRUPT"
IO_ERROR = "IO_ERROR"
QUEUE_FULL = "QUEUE_FULL"
CANCELLED = "CANCELLED"
TIMEOUT = "TIMEOUT"
INTERNAL_ERROR = "INTERNAL_ERROR"

_LOAD = "load"
_STORE = "store"
_TERMINAL_EVENT_STATUSES = ("ok", "drop", "timeout", "failed")


# ---------------------------------------------------------------------------
# Neutral wire DTOs (resource_blocks_v1)
# ---------------------------------------------------------------------------


class ResourceBlockLayout(msgspec.Struct, frozen=True):
    """Registered geometry of one transferable resource.

    Attributes:
        resource_id: Plan resource id (cache name for the engine side).
        cache_name: Engine cache name the resource resolves to.
        dtype: Torch dtype string of one element (for example ``float32``).
        shape: Per-block element shape of one resource payload plane.
        strides: Element strides of one payload plane (row-major).
        max_payload_bytes: Hard upper bound of one resource payload.
        encoding: ``token_range_v1`` or ``boundary_schema2``.
    """

    resource_id: str
    cache_name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    max_payload_bytes: int
    encoding: str


class ResourceBlockRegistration(msgspec.Struct, frozen=True):
    """Worker registration for engine-neutral resource blocks."""

    protocol_version: int
    instance_id: str
    model_name: str
    world_size: int
    worker_id: int | None
    plan_hash: str
    layout_hash: str
    registration_epoch: int
    resources: tuple[CacheTransferResource, ...]
    layouts: tuple[ResourceBlockLayout, ...]


class ResourceBlockAck(msgspec.Struct, frozen=True):
    """Server acknowledgement of one resource-block registration."""

    instance_id: str
    registration_epoch: int
    plan_hash: str
    layout_hash: str
    server_generation: int
    capabilities: tuple[str, ...]


class ResourceBlockSpec(msgspec.Struct, frozen=True):
    """One resource block store/retrieve request.

    ``stock_key`` is the request's stock cache identity; the server derives
    the chunk hashes from it, so a suffix can never decide the identity of a
    shared prefix.
    """

    operation_id: str
    direction: str
    instance_id: str
    request_id: str
    step_id: int
    registration_epoch: int
    server_generation: int
    plan_hash: str
    layout_hash: str
    stock_key: IPCCacheServerKey
    resource_id: str
    engine_group_id: int | None
    block_ids: tuple[int, ...]
    start_token: int
    end_token: int
    prefix_end: int
    skip_first_n_tokens: int
    boundary: Any = None


class ResourceBlockResult(msgspec.Struct, frozen=True):
    """Echoed terminal result of one resource block operation."""

    operation_id: str
    direction: str
    resource_id: str
    registration_epoch: int
    server_generation: int
    status: str
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class ResourceBlockLookupResult(msgspec.Struct, frozen=True):
    """Derived prefix visibility of one resource-block namespace.

    Attributes:
        prefix_end: Longest chunk-aligned prefix that has every token-range
            resource committed for every kv rank, capped by the boundary
            terminal endpoint when the namespace has boundaries.
        token_prefix_end: Longest prefix with complete token-range resources.
        boundary_prefix_end: Longest complete boundary endpoint (0 without
            boundary resources).
    """

    prefix_end: int
    token_prefix_end: int
    boundary_prefix_end: int


@runtime_checkable
class ResourceBlockTransport(Protocol):
    """Client side of the engine-neutral resource-block RPCs."""

    def register_resource_blocks(
        self, registration: ResourceBlockRegistration
    ) -> Future: ...

    def store_resource_block(
        self, spec: ResourceBlockSpec, payload: bytes
    ) -> Future: ...

    def retrieve_resource_block(self, spec: ResourceBlockSpec) -> Future: ...

    def lookup_resource_blocks(
        self, registration_epoch: int, stock_key: IPCCacheServerKey, limit: int
    ) -> Future: ...


_RESOURCE_BLOCK_CLIENT_METHODS = (
    "register_resource_blocks",
    "store_resource_block",
    "retrieve_resource_block",
    "lookup_resource_blocks",
)


class MPResourceBlockTransport:
    """``ResourceBlockTransport`` over the stock multiprocess request client.

    The stock worker owns a typed ``RequestClient``
    (``ZmqMultiprocessClient``): the resource-block methods are generated from
    the ``@rpc_method`` declarations, so the generated client has no
    ``submit_request``. Older/generic clients only expose
    ``submit_request(operation, payloads)``. Both are accepted; the typed
    methods are preferred when all four are present.

    Args:
        client: Stock multiprocess request client.
        timeout: Blocking timeout for the registration acknowledgement.
    """

    def __init__(self, client: Any, *, timeout: float = 30.0) -> None:
        typed = [
            method
            for method in _RESOURCE_BLOCK_CLIENT_METHODS
            if callable(getattr(client, method, None))
        ]
        has_submit = callable(getattr(client, "submit_request", None))
        if typed and len(typed) != len(_RESOURCE_BLOCK_CLIENT_METHODS):
            missing = sorted(
                set(_RESOURCE_BLOCK_CLIENT_METHODS) - set(typed)
            )
            raise TypeError(
                "request client has a partial resource-block interface; "
                f"missing {missing!r}"
            )
        if not typed and not has_submit:
            raise TypeError(
                "client must expose the typed resource-block RPCs "
                f"{list(_RESOURCE_BLOCK_CLIENT_METHODS)!r} "
                "(stock RequestClient) or submit_request(operation, payloads)"
            )
        self._client = client
        self._typed = bool(typed)
        self._timeout = float(timeout)

    def _submit(self, operation: str, *payloads: Any) -> Future:
        return self._client.submit_request(operation, list(payloads))

    def register_resource_blocks(
        self, registration: ResourceBlockRegistration
    ) -> Future:
        if self._typed:
            return self._client.register_resource_blocks(registration)
        return self._submit("register_resource_blocks", registration)

    def store_resource_block(
        self, spec: ResourceBlockSpec, payload: bytes
    ) -> Future:
        if self._typed:
            return self._client.store_resource_block(spec, payload)
        return self._submit("store_resource_block", spec, payload)

    def retrieve_resource_block(self, spec: ResourceBlockSpec) -> Future:
        if self._typed:
            return self._client.retrieve_resource_block(spec)
        return self._submit("retrieve_resource_block", spec)

    def lookup_resource_blocks(
        self, registration_epoch: int, stock_key: IPCCacheServerKey, limit: int
    ) -> Future:
        request = resource_block_lookup_request(
            registration_epoch, stock_key, limit
        )
        if self._typed:
            return self._client.lookup_resource_blocks(request)
        return self._submit("lookup_resource_blocks", request)


class _ResourceBlockLookupRequest(msgspec.Struct, frozen=True):
    registration_epoch: int
    stock_key: IPCCacheServerKey
    limit: int


def resource_block_lookup_request(
    registration_epoch: int, stock_key: IPCCacheServerKey, limit: int
) -> Any:
    """Build the wire payload of ``lookup_resource_blocks``.

    Kept as a public helper so the server module and clients share one type
    without importing each other.
    """
    return _ResourceBlockLookupRequest(
        registration_epoch=int(registration_epoch),
        stock_key=stock_key,
        limit=int(limit),
    )


# ---------------------------------------------------------------------------
# Group event
# ---------------------------------------------------------------------------


class GroupEvent:
    """Terminal state of one group operation.

    The event starts ``pending``, tracks one outstanding unit per submitted
    resource chunk, and resolves exactly once. ``drop`` is terminal even when
    RPC callbacks are still outstanding: those callbacks observe the terminal
    status and must not change it.

    Args:
        group_id: Transfer group id (``g2`` ... ``g20``).
        request_id: Request the operation belongs to.
        direction: ``load`` or ``store``.
    """

    def __init__(self, group_id: str, request_id: str, direction: str) -> None:
        self.group_id = group_id
        self.request_id = request_id
        self.direction = direction
        self.status = "pending"
        self.error_code = ""
        self.error_message = ""
        self.failed_resource_id: str | None = None
        self.dropped_reason = ""
        self._pending = 0
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._copy_events: list[Any] = []

    # -- producer side ---------------------------------------------------

    def add(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        with self._lock:
            if self.status != "pending":
                return
            self._pending += count

    def complete(self) -> None:
        with self._lock:
            if self.status != "pending":
                return
            if self._pending > 0:
                self._pending -= 1
            if self._pending == 0:
                self.status = STATUS_OK
                self._cv.notify_all()

    def fail(self, *, resource_id: str | None, error_code: str, message: str = "") -> None:
        with self._lock:
            if self.status != "pending":
                return
            self.status = "failed"
            self.error_code = error_code
            self.error_message = message
            self.failed_resource_id = resource_id
            self._cv.notify_all()

    def drop(self, *, reason: str) -> None:
        with self._lock:
            if self.status != "pending":
                return
            self.status = "drop"
            self.dropped_reason = reason
            self.error_code = CANCELLED
            self._cv.notify_all()

    def add_copy_event(self, event: Any) -> None:
        with self._lock:
            self._copy_events.append(event)

    def copy_events(self) -> tuple[Any, ...]:
        with self._lock:
            return tuple(self._copy_events)

    # -- consumer side ---------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self.status in _TERMINAL_EVENT_STATUSES

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def wait(self, timeout: float | None) -> str:
        """Block until the event resolves, the timeout expires, or drop wins."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._cv:
            while self.status == "pending":
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                self._cv.wait(remaining)
            if self.status == "pending":
                self.status = "timeout"
                self.error_code = TIMEOUT
                self._cv.notify_all()
            return self.status


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ChunkRef:
    start_token: int
    end_token: int
    block_ids: tuple[int, ...]


@dataclass
class _StagedStore:
    request_id: str
    group_id: str
    resource_id: str
    chunk_start: int
    chunk_end: int
    block_ids: tuple[int, ...]
    planes: tuple[torch.Tensor, ...]
    operation_id: str
    is_boundary: bool = False
    capture: Any = None


@dataclass
class _StoreBatch:
    event: GroupEvent
    staged: list[_StagedStore] = field(default_factory=list)


@dataclass
class _BoundaryCapture:
    page: torch.Tensor
    event: Any
    encoder: Callable[[torch.Tensor], bytes]
    encode_lock: threading.Lock = field(default_factory=threading.Lock)
    done: threading.Event = field(default_factory=threading.Event)
    encoded: bytes | None = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DSV41LwV2GroupAdapter:
    """Group-level transport adapter for the four DSV4.1 KV source groups.

    Args:
        plan: Validated grouped plan whose resource descriptors define the
            wire resources.
        transport: Engine-neutral resource-block transport.
        instance_id: Worker instance id (registration identity).
        model_name: Served model name (stock key identity component).
        world_size: KV rank shard count.
        worker_id: Optional KV worker id used by the stock key resolver.
        plan_hash: Hash of the registered base plan.
        layout_hash: Hash of the registered tensor layout.
        registration_epoch: Registration epoch of this worker.
        server_generation: Server generation returned by the registration ack.
        chunk_size: LMCache chunk size in tokens.
        block_size: Engine block size in tokens.
        kv_caches: Registered engine cache name -> tensor (or plane tuple).
        cache_names: Registered cache names in plan tensor-index order.
        load_timeout: Per-group load wait budget in seconds.
        store_timeout: ``drain_stores`` wait budget in seconds.
        max_queue: Bound of queued worker jobs; queue-full fails loads and
            drops stores instead of blocking the scheduler.
    """

    def __init__(
        self,
        plan: GroupedTransferPlan,
        *,
        transport: ResourceBlockTransport,
        instance_id: str,
        model_name: str,
        world_size: int,
        worker_id: int | None = None,
        plan_hash: str,
        layout_hash: str,
        registration_epoch: int = 0,
        server_generation: int = 0,
        chunk_size: int = 256,
        block_size: int = 16,
        kv_caches: Mapping[str, Any] | None = None,
        cache_names: Sequence[str] = (),
        load_timeout: float = 30.0,
        store_timeout: float = 60.0,
        max_queue: int = 1024,
    ) -> None:
        if not isinstance(plan, GroupedTransferPlan):
            raise TypeError("plan must be a GroupedTransferPlan")
        if not isinstance(transport, ResourceBlockTransport):
            raise TypeError("transport must implement ResourceBlockTransport")
        if not instance_id:
            raise ValueError("instance_id must be nonempty")
        if not model_name:
            raise ValueError("model_name must be nonempty")
        if int(world_size) < 1:
            raise ValueError("world_size must be >= 1")
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be >= 1")
        if int(block_size) < 1:
            raise ValueError("block_size must be >= 1")
        if int(chunk_size) % int(block_size):
            raise ValueError("chunk_size must be a multiple of block_size")
        if int(max_queue) < 1:
            raise ValueError("max_queue must be >= 1")
        self._plan = plan
        self._transport = transport
        self._instance_id = instance_id
        self._model_name = model_name
        self._world_size = int(world_size)
        self._worker_id = None if worker_id is None else int(worker_id)
        self._plan_hash = plan_hash
        self._layout_hash = layout_hash
        self._registration_epoch = int(registration_epoch)
        self._server_generation = int(server_generation)
        self._chunk_size = int(chunk_size)
        self._block_size = int(block_size)
        self._load_timeout = float(load_timeout)
        self._store_timeout = float(store_timeout)
        self._max_queue = int(max_queue)

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._kv_caches: dict[str, Any] = dict(kv_caches or {})
        self._cache_names: tuple[str, ...] = tuple(cache_names)
        self._layouts: dict[str, ResourceBlockLayout] = {}
        self._resources: dict[str, CacheTransferResource] = {}
        self._boundary_ids: set[str] = set()
        for group in plan.groups:
            for resource in group.transfer_resources:
                self._resources[resource.resource_id] = resource
                if resource.storage_kind == "boundary_state":
                    self._boundary_ids.add(resource.resource_id)
        if not self._layouts:
            self._build_layouts()

        # Operation bookkeeping.
        self._load_events: dict[tuple[str, str, int], GroupEvent] = {}
        self._store_batches: list[_StoreBatch] = []
        self._terminal_ops: set[tuple[str, str, str, int, str, int]] = set()
        self._staged_ends: dict[tuple[str, str], int] = {}
        self._boundary_stage: dict[tuple[str, str, int], bytes] = {}
        self._boundary_payloads: dict[tuple[str, str, int], bytes] = {}
        self._failed_receiving: set[str] = set()
        self._failed_receiving_reported: set[str] = set()
        self._receiving_ready: set[str] = set()
        self._sending_ready: set[str] = set()
        self._request_steps: dict[str, int] = {}
        self._known_requests_store: dict[str, TransferRequest] = {}
        self._step_by_request: dict[str, StepView] = {}
        self._expected_load_groups: dict[str, set[str]] = {}
        self._completed_load_groups: dict[str, set[str]] = {}
        self._copy_stream: Any = None
        self._device: Any = None

        # Metrics: every counter is a terminal-state count, never a submit.
        self._counters: dict[str, int] = {
            "group_load_submitted": 0,
            "group_load_ready": 0,
            "group_load_dropped": 0,
            "group_load_failed": 0,
            "group_load_timeout": 0,
            "group_store_submitted": 0,
            "group_store_committed": 0,
            "group_store_dropped": 0,
            "group_store_failed": 0,
            "boundary_captured": 0,
            "boundary_restored": 0,
            "boundary_restore_failed": 0,
            "host_wait_fallback": 0,
            "swa_transfer_attempts": 0,
            "stale_ops_cleared": 0,
            "queue_full": 0,
            "payload_corrupt": 0,
            "requests_finished_sending": 0,
            "requests_failed_receiving": 0,
        }

        self._jobs: queue.Queue[Any] = queue.Queue(maxsize=self._max_queue)
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"dsv41-lw-v2-adapter-{instance_id}",
            daemon=True,
        )
        self._closed = False
        self._worker.start()

    # ------------------------------------------------------------------
    # Public properties / registration
    # ------------------------------------------------------------------

    @property
    def plan(self) -> GroupedTransferPlan:
        return self._plan

    @property
    def metrics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    @property
    def registration_epoch(self) -> int:
        return self._registration_epoch

    @property
    def server_generation(self) -> int:
        return self._server_generation

    @property
    def boundary_resource_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._boundary_ids))

    @property
    def layouts(self) -> tuple[ResourceBlockLayout, ...]:
        return tuple(self._layouts[rid] for rid in sorted(self._layouts))

    def bind_kv_caches(
        self,
        kv_caches: Mapping[str, Any],
        cache_names: Sequence[str] | None = None,
    ) -> None:
        """Bind the registered engine caches after cache registration."""
        with self._lock:
            self._kv_caches = dict(kv_caches)
            if cache_names is not None:
                self._cache_names = tuple(cache_names)
            self._layouts.clear()
            self._build_layouts()

    def register(self) -> ResourceBlockAck:
        """Register the plan's transferable resources; fail closed on ack mismatch."""
        registration = ResourceBlockRegistration(
            protocol_version=RESOURCE_BLOCK_PROTOCOL_VERSION,
            instance_id=self._instance_id,
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            plan_hash=self._plan_hash,
            layout_hash=self._layout_hash,
            registration_epoch=self._registration_epoch,
            resources=tuple(
                resource
                for group in self._plan.groups
                for resource in group.transfer_resources
            ),
            layouts=self.layouts,
        )
        future = self._transport.register_resource_blocks(registration)
        ack = future.result(timeout=self._load_timeout)
        if not isinstance(ack, ResourceBlockAck):
            raise DSV41GroupLoadError(
                "resource-block registration returned an invalid ack",
                request_id="",
                group_id="",
                error_code=INVALID_SPEC,
            )
        if (
            ack.instance_id != self._instance_id
            or ack.registration_epoch != self._registration_epoch
            or ack.plan_hash != self._plan_hash
            or ack.layout_hash != self._layout_hash
        ):
            raise DSV41GroupLoadError(
                "resource-block registration ack does not echo this worker",
                request_id="",
                group_id="",
                error_code=LAYOUT_MISMATCH,
            )
        missing = [
            capability
            for capability in REQUIRED_CAPABILITIES
            if capability not in ack.capabilities
        ]
        if missing:
            raise DSV41GroupLoadError(
                "server does not advertise required resource-block "
                f"capabilities: {missing!r}",
                request_id="",
                group_id="",
                error_code=LAYOUT_MISMATCH,
            )
        with self._lock:
            self._server_generation = int(ack.server_generation)
        return ack

    def set_generations(
        self, *, registration_epoch: int, server_generation: int
    ) -> None:
        """Advance identity and drop every operation of an older generation."""
        registration_epoch = int(registration_epoch)
        server_generation = int(server_generation)
        with self._lock:
            changed = (
                registration_epoch != self._registration_epoch
                or server_generation != self._server_generation
            )
            self._registration_epoch = registration_epoch
            self._server_generation = server_generation
        if changed:
            self.drop_stale_ops(reason="generation_changed")

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def schedule_group_retrieves(
        self, group_id: str, request_ids: Sequence[str], step: StepView
    ) -> None:
        """Queue one retrieve job per group without waiting or copying inline."""
        group = self._plan.group_by_id(group_id)
        self._remember_requests(step)
        requests = self._requests_for(step, request_ids)
        epoch = step.registration_epoch
        if epoch != self._registration_epoch:
            self._fail_requests(
                requests,
                group_id,
                error_code=STALE_EPOCH,
                message=(
                    f"step epoch {epoch} does not match adapter epoch "
                    f"{self._registration_epoch}"
                ),
            )
            return
        jobs: list[_LoadJob] = []
        with self._lock:
            for request in requests:
                if request.load_prefix_end <= 0:
                    continue
                event = GroupEvent(group_id, request.request_id, _LOAD)
                self._expected_load_groups.setdefault(
                    request.request_id, set()
                ).add(group_id)
                try:
                    chunks = self._load_chunks(request)
                except ValueError as exc:
                    event.fail(
                        resource_id=None,
                        error_code=PREFIX_INCOMPLETE,
                        message=str(exc),
                    )
                else:
                    event.add(len(chunks) * len(group.token_range_resources))
                    if group.has_boundary:
                        event.add(1)
                    self._load_events[
                        (group_id, request.request_id, step.step_id)
                    ] = event
                    jobs.append(
                        _LoadJob(
                            group=group,
                            request=request,
                            step=step,
                            chunks=chunks,
                        )
                    )
        for job in jobs:
            if not self._enqueue(("load", job)):
                self._fail_event(
                    self._load_events.get(
                        (group_id, job.request.request_id, step.step_id)
                    ),
                    resource_id=None,
                    error_code=QUEUE_FULL,
                    message="adapter job queue is full",
                )

    def wait_group(
        self,
        group_id: str,
        request_ids: Sequence[str],
        step: StepView,
        *,
        stream: Any = None,
    ) -> None:
        """Wait one group's loads and hand every copy event to ``stream``."""
        self._remember_requests(step)
        requests = self._requests_for(step, request_ids)
        for request in requests:
            if request.load_prefix_end <= 0:
                continue
            with self._lock:
                event = self._load_events.get(
                    (group_id, request.request_id, step.step_id)
                )
            if event is None:
                raise DSV41GroupLoadError(
                    "group load was not scheduled for this step",
                    request_id=request.request_id,
                    group_id=group_id,
                    error_code=INVALID_SPEC,
                )
            if not event.is_terminal:
                with self._lock:
                    self._counters["host_wait_fallback"] += 1
            status = event.wait(self._load_timeout)
            if status == "timeout":
                with self._lock:
                    self._counters["group_load_timeout"] += 1
                    self._counters["group_load_failed"] += 1
                raise DSV41GroupLoadError(
                    "group load timed out",
                    request_id=request.request_id,
                    group_id=group_id,
                    error_code=TIMEOUT,
                )
            if status != STATUS_OK:
                with self._lock:
                    self._counters["group_load_failed"] += 1
                self._mark_failed_receiving(request.request_id)
                raise DSV41GroupLoadError(
                    f"group load {status}: {event.error_message}",
                    request_id=request.request_id,
                    group_id=group_id,
                    resource_id=event.failed_resource_id,
                    error_code=event.error_code or CANCELLED,
                )
            self._wait_copy_events(event, stream)
            with self._lock:
                self._counters["group_load_ready"] += 1
                self._completed_load_groups.setdefault(
                    request.request_id, set()
                ).add(group_id)
                self._maybe_finish_receiving_locked(request.request_id)

    # ------------------------------------------------------------------
    # Stores
    # ------------------------------------------------------------------

    def submit_group_store(
        self, group_id: str, request_ids: Sequence[str], step: StepView
    ) -> None:
        """Stage the group's new rows and queue the store; never block on RPC."""
        group = self._plan.group_by_id(group_id)
        self._remember_requests(step)
        requests = self._requests_for(step, request_ids)
        stage_failures: list[tuple[TransferRequest, str]] = []
        for request in requests:
            if request.store_prefix_end <= 0:
                continue
            try:
                staged = self._stage_store(group, request)
            except (ValueError, KeyError) as exc:
                stage_failures.append((request, str(exc)))
                continue
            if not staged:
                continue
            event = GroupEvent(group_id, request.request_id, _STORE)
            event.add(len(staged))
            batch = _StoreBatch(event=event, staged=staged)
            with self._lock:
                self._store_batches.append(batch)
            if not self._enqueue(("store", batch)):
                self._fail_event(
                    event,
                    resource_id=None,
                    error_code=QUEUE_FULL,
                    message="adapter job queue is full",
                )
        for request, message in stage_failures:
            self._counters["group_store_dropped"] += 1
            logger_warning(
                "dsv41 lw v2 store staging failed for request %s group %s: %s",
                request.request_id,
                group_id,
                message,
            )

    def drain_stores(self) -> None:
        """Wait every submitted store; failures degrade to dropped, never ok."""
        deadline = time.monotonic() + self._store_timeout
        with self._lock:
            events = [batch.event for batch in self._store_batches]
        for event in events:
            remaining = deadline - time.monotonic()
            status = event.wait(max(0.0, remaining))
            if status == "timeout":
                with self._lock:
                    self._counters["group_store_failed"] += 1
                    self._mark_sending_ready_locked(event.request_id)
                continue
            with self._lock:
                if status == STATUS_OK:
                    self._counters["group_store_committed"] += 1
                elif status == "drop":
                    self._counters["group_store_dropped"] += 1
                else:
                    self._counters["group_store_failed"] += 1
                self._mark_sending_ready_locked(event.request_id)
        with self._lock:
            self._store_batches = [
                batch
                for batch in self._store_batches
                if batch.event.status == "pending"
            ]
            for batch in self._store_batches:
                batch.staged.clear()
        self._release_store_buffers()

    # ------------------------------------------------------------------
    # Drops / terminal reports
    # ------------------------------------------------------------------

    def drop_request(self, request_id: str, *, reason: str) -> None:
        """Drop every non-terminal operation of one request."""
        self._drop_where(
            lambda key: key[1] == request_id, reason=reason
        )

    def drop_group(self, group_id: str, *, reason: str = "dropped") -> None:
        """Drop every non-terminal operation of one group."""
        self._drop_where(lambda key: key[0] == group_id, reason=reason)

    def drop_stale_ops(self, *, reason: str = "stale") -> None:
        """Drop every non-terminal operation of the previous generation."""
        self._drop_where(lambda key: True, reason=reason, counter="stale_ops_cleared")

    def take_finished(self) -> tuple[set[str], set[str]]:
        """Return request ids whose receiving/sending direction is settled."""
        with self._lock:
            finished_receiving = set(self._receiving_ready)
            self._receiving_ready.clear()
            finished_sending = set(self._sending_ready)
            self._sending_ready.clear()
            self._counters["requests_finished_sending"] += len(
                finished_sending
            )
            return finished_receiving, finished_sending

    def take_failed_receiving(self) -> set[str]:
        """Return requests whose load failed and must be recomputed."""
        with self._lock:
            failed = set(self._failed_receiving)
            self._failed_receiving.clear()
            self._failed_receiving_reported.update(failed)
            self._counters["requests_failed_receiving"] += len(failed)
            return failed

    # ------------------------------------------------------------------
    # Boundary payload staging (codec integration)
    # ------------------------------------------------------------------

    def stage_boundary_capture(
        self,
        request_id: str,
        resource_id: str,
        prefix_end: int,
        page: torch.Tensor,
        event: Any | None,
        encoder: Callable[[torch.Tensor], bytes],
        *,
        stream: Any = None,
    ) -> None:
        """Stage an immutable boundary page without encoding on the caller.

        ``page`` must already be a private device copy (the codec clones the
        live ring page before calling). The device-to-host copy and the
        schema-2 encoding happen on the adapter worker.
        """
        self._require_boundary_resource(resource_id)
        capture = _BoundaryCapture(
            page=page, event=event, encoder=encoder
        )
        with self._lock:
            self._boundary_stage[
                (request_id, resource_id, int(prefix_end))
            ] = capture  # type: ignore[assignment]
        if not self._enqueue(("capture", capture)):
            with self._lock:
                self._counters["group_store_dropped"] += 1
                self._boundary_stage.pop(
                    (request_id, resource_id, int(prefix_end)), None
                )

    def record_boundary_captured(self) -> None:
        """Count one boundary page successfully staged for transport."""
        with self._lock:
            self._counters["boundary_captured"] += 1

    def record_boundary_restored(self) -> None:
        """Count one boundary page actually written into a live ring."""
        with self._lock:
            self._counters["boundary_restored"] += 1

    def record_boundary_restore_failed(self) -> None:
        """Count one explicit boundary restore failure (never a no-op)."""
        with self._lock:
            self._counters["boundary_restore_failed"] += 1

    def take_boundary_payload(
        self, request_id: str, resource_id: str, prefix_end: int
    ) -> bytes | None:
        """Consume one retrieved boundary payload (single use)."""
        with self._lock:
            return self._boundary_payloads.pop(
                (request_id, resource_id, int(prefix_end)), None
            )

    def peek_boundary_payload(
        self, request_id: str, resource_id: str, prefix_end: int
    ) -> bytes | None:
        with self._lock:
            return self._boundary_payloads.get(
                (request_id, resource_id, int(prefix_end)), None
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the worker after dropping outstanding operations."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._drop_where(lambda key: True, reason="adapter_closed")
        self._jobs.put(None)
        self._worker.join(timeout=5.0)
        self._release_store_buffers()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            kind, payload = job
            try:
                if kind == "load":
                    self._run_load(payload)
                elif kind == "store":
                    self._run_store(payload)
                elif kind == "capture":
                    self._run_capture(payload)
            except Exception as exc:  # noqa: BLE001 - worker must not die
                self._record_worker_failure(kind, payload, exc)

    def _run_load(self, job: "_LoadJob") -> None:
        group = job.group
        request = job.request
        event = self._load_events.get(
            (group.group_id, request.request_id, job.step.step_id)
        )
        if event is None or event.is_terminal:
            return
        for resource in group.token_range_resources:
            layout = self._layouts.get(resource.resource_id)
            if layout is None:
                event.fail(
                    resource_id=resource.resource_id,
                    error_code=LAYOUT_MISMATCH,
                    message="resource has no registered layout",
                )
                return
            for chunk in job.chunks:
                try:
                    resolved = self._chunk_with_blocks(
                        request, resource, chunk
                    )
                except ValueError as exc:
                    event.fail(
                        resource_id=resource.resource_id,
                        error_code=PREFIX_INCOMPLETE,
                        message=str(exc),
                    )
                    return
                spec = self._spec(
                    direction=_LOAD,
                    request=request,
                    step=job.step,
                    resource_id=resource.resource_id,
                    engine_group_id=resource.engine_group_id,
                    chunk=resolved,
                    prefix_end=request.load_prefix_end,
                )
                self._submit_retrieve(
                    group_id=group.group_id,
                    request=request,
                    resource_id=resource.resource_id,
                    chunk=resolved,
                    spec=spec,
                    boundary=False,
                )
        if group.has_boundary:
            assert group.boundary_resource is not None
            resource = group.boundary_resource
            spec = self._spec(
                direction=_LOAD,
                request=request,
                step=job.step,
                resource_id=resource.resource_id,
                engine_group_id=resource.engine_group_id,
                chunk=None,
                prefix_end=request.load_prefix_end,
            )
            self._submit_retrieve(
                group_id=group.group_id,
                request=request,
                resource_id=resource.resource_id,
                chunk=None,
                spec=spec,
                boundary=True,
            )

    def _submit_retrieve(
        self,
        *,
        group_id: str,
        request: TransferRequest,
        resource_id: str,
        chunk: _ChunkRef | None,
        spec: ResourceBlockSpec,
        boundary: bool,
    ) -> None:
        event = self._load_events[
            (group_id, request.request_id, spec.step_id)
        ]
        try:
            future = self._transport.retrieve_resource_block(spec)
        except Exception as exc:  # noqa: BLE001 - reported through the event
            event.fail(
                resource_id=resource_id,
                error_code=IO_ERROR,
                message=f"retrieve submit failed: {exc}",
            )
            return
        future.add_done_callback(
            lambda done, rid=resource_id: self._on_retrieve_done(
                group_id=group_id,
                request=request,
                resource_id=rid,
                chunk=chunk,
                boundary=boundary,
                future=done,
            )
        )

    def _on_retrieve_done(
        self,
        *,
        group_id: str,
        request: TransferRequest,
        resource_id: str,
        chunk: _ChunkRef | None,
        boundary: bool,
        future: Future,
    ) -> None:
        key = (group_id, request.request_id, request.step_id)
        event = self._load_events.get(key)
        if event is None or event.is_terminal:
            return
        # A resource is transferred in several chunk-sized operations, so the
        # terminal key must include the chunk range; keying on the resource
        # alone would silently leave the second chunk pending forever.
        chunk_start = (
            chunk.start_token
            if chunk is not None
            else int(request.load_prefix_end)
        )
        terminal = (
            group_id,
            request.request_id,
            _LOAD,
            request.step_id,
            resource_id,
            chunk_start,
        )
        if terminal in self._terminal_ops:
            return
        try:
            result, payload = future.result()
        except Exception as exc:  # noqa: BLE001 - transport/result failure
            self._terminal_ops.add(terminal)
            event.fail(
                resource_id=resource_id,
                error_code=IO_ERROR,
                message=f"retrieve failed: {exc}",
            )
            return
        if not isinstance(result, ResourceBlockResult) or not result.ok:
            self._terminal_ops.add(terminal)
            status = getattr(result, "status", STATUS_FAILED)
            code = getattr(result, "error_code", "") or (
                RESOURCE_MISSING if status == STATUS_MISS else IO_ERROR
            )
            event.fail(
                resource_id=resource_id,
                error_code=code,
                message=f"retrieve status {status}",
            )
            return
        if (
            result.registration_epoch != self._registration_epoch
            or result.server_generation != self._server_generation
            or not result.operation_id
        ):
            self._terminal_ops.add(terminal)
            event.fail(
                resource_id=resource_id,
                error_code=STALE_GENERATION,
                message="retrieve result does not echo this generation",
            )
            return
        try:
            self._apply_retrieve(
                event=event,
                group_id=group_id,
                request=request,
                resource_id=resource_id,
                chunk=chunk,
                boundary=boundary,
                payload=payload,
            )
        except (ValueError, RuntimeError, IndexError, KeyError) as exc:
            self._terminal_ops.add(terminal)
            with self._lock:
                self._counters["payload_corrupt"] += 1
            event.fail(
                resource_id=resource_id,
                error_code=PAYLOAD_CORRUPT,
                message=f"retrieved payload rejected: {exc}",
            )
            return
        self._terminal_ops.add(terminal)
        event.complete()

    def _apply_retrieve(
        self,
        *,
        event: GroupEvent,
        group_id: str,
        request: TransferRequest,
        resource_id: str,
        chunk: _ChunkRef | None,
        boundary: bool,
        payload: Any,
    ) -> None:
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError("retrieved payload is not bytes")
        if boundary:
            if not bytes(payload).startswith(BOUNDARY_PAYLOAD_MAGIC):
                raise ValueError("boundary payload has no schema-2 envelope")
            with self._lock:
                self._boundary_payloads[
                    (
                        request.request_id,
                        resource_id,
                        int(request.load_prefix_end),
                    )
                ] = bytes(payload)
            return
        assert chunk is not None
        planes = decode_token_payload(bytes(payload), resource_id=resource_id)
        expected_rows = len(chunk.block_ids)
        if not planes:
            raise ValueError("token payload has no planes")
        targets = self._resource_planes(resource_id)
        if len(targets) != len(planes):
            raise ValueError(
                f"resource {resource_id} has {len(targets)} engine planes but "
                f"payload has {len(planes)}"
            )
        copy_events = []
        for target, plane in zip(targets, planes):
            if target.shape[0] <= chunk.block_ids[-1]:
                raise ValueError(
                    f"block id {chunk.block_ids[-1]} is outside the "
                    f"registered cache {resource_id}"
                )
            value = plane.to(dtype=target.dtype)
            if tuple(plane.shape) != tuple(target.shape[1:]):
                if tuple(plane.shape) != (expected_rows, *target.shape[1:]):
                    raise ValueError(
                        f"payload shape {tuple(plane.shape)} does not match "
                        f"rows {expected_rows} x {tuple(target.shape[1:])}"
                    )
                value = plane.reshape((expected_rows, *target.shape[1:]))
            device = target.device
            index = torch.as_tensor(chunk.block_ids, dtype=torch.long)
            if device.type == "cuda":
                stream = self._copy_stream_for(device)
                with torch.cuda.stream(stream):
                    host_value = value.to(device)
                    target.index_copy_(0, index.to(device), host_value)
                    event_obj = torch.cuda.Event()
                    event_obj.record(stream)
                copy_events.append(event_obj)
            else:
                target.index_copy_(0, index, value.to(device))
        for copy_event in copy_events:
            event.add_copy_event(copy_event)

    def _run_store(self, batch: _StoreBatch) -> None:
        event = batch.event
        for staged in batch.staged:
            spec = self._spec(
                direction=_STORE,
                request=self._request_by_id(staged.request_id),
                step=self._step_for_request(staged.request_id),
                resource_id=staged.resource_id,
                engine_group_id=self._resources[
                    staged.resource_id
                ].engine_group_id,
                chunk=_ChunkRef(
                    start_token=staged.chunk_start,
                    end_token=staged.chunk_end,
                    block_ids=staged.block_ids,
                ),
                prefix_end=staged.chunk_end,
            )
            payload = self._encode_staged(staged)
            if payload is None:
                event.fail(
                    resource_id=staged.resource_id,
                    error_code=PAYLOAD_CORRUPT,
                    message="boundary payload is missing for this store",
                )
                continue
            try:
                future = self._transport.store_resource_block(spec, payload)
            except Exception as exc:  # noqa: BLE001 - terminal store failure
                event.fail(
                    resource_id=staged.resource_id,
                    error_code=IO_ERROR,
                    message=f"store submit failed: {exc}",
                )
                continue
            future.add_done_callback(
                lambda done, rid=staged.resource_id, s=staged: (
                    self._on_store_done(
                        event=event, resource_id=rid, staged=s, future=done
                    )
                )
            )

    def _on_store_done(
        self,
        *,
        event: GroupEvent,
        resource_id: str,
        staged: _StagedStore,
        future: Future,
    ) -> None:
        if event.is_terminal:
            return
        terminal = (
            event.group_id,
            event.request_id,
            _STORE,
            staged.chunk_end,
            resource_id,
        )
        if terminal in self._terminal_ops:
            return
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - reported terminally
            self._terminal_ops.add(terminal)
            event.fail(
                resource_id=resource_id,
                error_code=IO_ERROR,
                message=f"store failed: {exc}",
            )
            return
        if not isinstance(result, ResourceBlockResult) or result.status not in (
            STATUS_OK,
        ):
            self._terminal_ops.add(terminal)
            event.fail(
                resource_id=resource_id,
                error_code=getattr(result, "error_code", "") or IO_ERROR,
                message=f"store status {getattr(result, 'status', '?')}",
            )
            return
        self._terminal_ops.add(terminal)
        with self._lock:
            current = self._staged_ends.get((event.request_id, resource_id), 0)
            self._staged_ends[(event.request_id, resource_id)] = max(
                current, staged.chunk_end
            )
        event.complete()

    def _run_capture(self, capture: _BoundaryCapture) -> None:
        try:
            with capture.encode_lock:
                if capture.encoded is not None:
                    return
                event = capture.event
                if event is not None and not callable(
                    getattr(event, "synchronize", None)
                ):
                    event = None
                if event is not None:
                    event.synchronize()
                cpu_page = capture.page.detach().to("cpu", copy=True)
                capture.encoded = capture.encoder(cpu_page)
        finally:
            capture.done.set()

    # ------------------------------------------------------------------
    # Staging helpers
    # ------------------------------------------------------------------

    def _load_chunks(self, request: TransferRequest) -> tuple[_ChunkRef, ...]:
        end = request.load_prefix_end
        if end % self._chunk_size:
            raise ValueError(
                f"load prefix {end} is not chunk aligned "
                f"(chunk_size={self._chunk_size})"
            )
        chunks = []
        for start in range(0, end, self._chunk_size):
            chunks.append(
                _ChunkRef(
                    start_token=start,
                    end_token=start + self._chunk_size,
                    block_ids=(),
                )
            )
        return tuple(chunks)

    def _chunk_with_blocks(
        self,
        request: TransferRequest,
        resource: CacheTransferResource,
        chunk: _ChunkRef,
    ) -> _ChunkRef:
        """Resolve one chunk's destination rows for a resource's engine group."""
        blocks = self._blocks_for(request, resource.engine_group_id)
        lo = chunk.start_token // self._block_size
        hi = chunk.end_token // self._block_size
        if hi > len(blocks):
            raise ValueError(
                f"resource {resource.resource_id} needs blocks [{lo}, {hi}) "
                f"but the request registered only {len(blocks)}"
            )
        return _ChunkRef(
            start_token=chunk.start_token,
            end_token=chunk.end_token,
            block_ids=tuple(blocks[lo:hi]),
        )

    def _stage_store(
        self, group: TransferGroup, request: TransferRequest
    ) -> list[_StagedStore]:
        if request.store_prefix_end % self._chunk_size:
            raise ValueError(
                f"store prefix {request.store_prefix_end} is not chunk aligned"
            )
        staged: list[_StagedStore] = []
        groups = request.block_ids_by_engine_group
        for resource in group.token_range_resources:
            resource_blocks = self._blocks_for(
                request, resource.engine_group_id, groups=groups
            )
            if not resource_blocks:
                raise ValueError(
                    f"request {request.request_id} has no block ids for "
                    f"resource {resource.resource_id}"
                )
            stored_end = self._staged_ends.get(
                (request.request_id, resource.resource_id), 0
            )
            start = max(stored_end, 0)
            while start < request.store_prefix_end:
                chunk_end = min(start + self._chunk_size, request.store_prefix_end)
                if chunk_end - start != self._chunk_size:
                    raise ValueError(
                        "store prefix is not chunk aligned at the tail"
                    )
                blocks = resource_blocks[
                    start // self._block_size : chunk_end // self._block_size
                ]
                if len(blocks) != self._chunk_size // self._block_size:
                    raise ValueError(
                        f"resource {resource.resource_id} has too few blocks "
                        f"for tokens [{start}, {chunk_end})"
                    )
                planes = self._resource_planes(resource.resource_id)
                cloned = tuple(
                    plane.index_select(
                        0,
                        torch.as_tensor(blocks, dtype=torch.long, device=plane.device),
                    )
                    .detach()
                    .clone()
                    for plane in planes
                )
                staged.append(
                    _StagedStore(
                        request_id=request.request_id,
                        group_id=group.group_id,
                        resource_id=resource.resource_id,
                        chunk_start=start,
                        chunk_end=chunk_end,
                        block_ids=tuple(blocks),
                        planes=cloned,
                        operation_id=uuid.uuid4().hex,
                    )
                )
                start = chunk_end
        if group.has_boundary:
            assert group.boundary_resource is not None
            resource = group.boundary_resource
            with self._lock:
                capture = self._boundary_stage.pop(
                    (
                        request.request_id,
                        resource.resource_id,
                        int(request.store_prefix_end),
                    ),
                    None,
                )
            if capture is not None:
                staged.append(
                    _StagedStore(
                        request_id=request.request_id,
                        group_id=group.group_id,
                        resource_id=resource.resource_id,
                        chunk_start=request.store_prefix_end,
                        chunk_end=request.store_prefix_end,
                        block_ids=(),
                        planes=(),
                        operation_id=uuid.uuid4().hex,
                        is_boundary=True,
                        capture=capture,
                    )
                )
        return staged

    def _await_capture(self, capture: Any) -> bytes:
        if isinstance(capture, bytes):
            return capture
        if not isinstance(capture, _BoundaryCapture):
            raise ValueError("boundary staging slot is not a capture")
        # Wait until the worker encoded the page; captures are small and this
        # only blocks the group head after the store was already staged for
        # the token resources.
        if not capture.done.wait(self._store_timeout):
            raise ValueError("boundary capture encoding timed out")
        if capture.encoded is None:
            raise ValueError("boundary capture encoding failed")
        return capture.encoded

    def _encode_staged(self, staged: _StagedStore) -> bytes | None:
        if staged.is_boundary:
            capture = staged.capture
            if isinstance(capture, (bytes, bytearray)):
                return bytes(capture)
            if not isinstance(capture, _BoundaryCapture):
                return None
            # Encoding happens on the worker, never on the forward thread.
            return self._await_capture(capture)
        return encode_token_payload(
            resource_id=staged.resource_id,
            block_ids=staged.block_ids,
            planes=staged.planes,
        )

    def _spec(
        self,
        *,
        direction: str,
        request: TransferRequest,
        step: StepView,
        resource_id: str,
        engine_group_id: int | None,
        chunk: _ChunkRef | None,
        prefix_end: int,
    ) -> ResourceBlockSpec:
        if chunk is None:
            start = end = prefix_end
            blocks: tuple[int, ...] = ()
        else:
            start, end, blocks = (
                chunk.start_token,
                chunk.end_token,
                chunk.block_ids,
            )
        boundary = (
            {"schema": 2, "resource_id": resource_id, "prefix_end": prefix_end}
            if resource_id in self._boundary_ids
            else None
        )
        return ResourceBlockSpec(
            operation_id=uuid.uuid4().hex,
            direction=direction,
            instance_id=self._instance_id,
            request_id=request.request_id,
            step_id=step.step_id,
            registration_epoch=self._registration_epoch,
            server_generation=self._server_generation,
            plan_hash=self._plan_hash,
            layout_hash=self._layout_hash,
            stock_key=request.stock_key,
            resource_id=resource_id,
            engine_group_id=engine_group_id,
            block_ids=blocks,
            start_token=start,
            end_token=end,
            prefix_end=prefix_end,
            skip_first_n_tokens=request.skip_first_n_tokens,
            boundary=boundary,
        )

    def _blocks_for(
        self,
        request: TransferRequest,
        engine_group_id: int | None,
        *,
        groups: tuple[tuple[int, ...], ...] | None = None,
    ) -> tuple[int, ...]:
        groups = (
            request.block_ids_by_engine_group if groups is None else groups
        )
        if not groups:
            return ()
        if engine_group_id is None:
            return groups[0]
        index = int(engine_group_id)
        if index < 0 or index >= len(groups):
            if len(groups) == 1:
                return groups[0]
            return ()
        return groups[index]

    def _resource_planes(self, resource_id: str) -> tuple[torch.Tensor, ...]:
        resource = self._resources.get(resource_id)
        if resource is None:
            raise KeyError(resource_id)
        name = self._cache_names_for(resource)
        value = self._kv_caches.get(name)
        if value is None:
            raise KeyError(
                f"registered kv cache {name!r} for resource "
                f"{resource_id!r} is not bound"
            )
        if isinstance(value, (tuple, list)):
            return tuple(value)
        return (value,)

    def _cache_names_for(self, resource: CacheTransferResource) -> str:
        if not resource.tensor_indices:
            raise KeyError(f"resource {resource.resource_id} has no tensors")
        index = resource.tensor_indices[0]
        if index < 0 or index >= len(self._cache_names):
            raise KeyError(
                f"resource {resource.resource_id} tensor index {index} is "
                f"outside the {len(self._cache_names)} registered caches"
            )
        return self._cache_names[index]

    def _build_layouts(self) -> None:
        for resource in self._resources.values():
            if resource.storage_kind == "boundary_state":
                self._layouts[resource.resource_id] = ResourceBlockLayout(
                    resource_id=resource.resource_id,
                    cache_name=resource.resource_id,
                    dtype="float32",
                    shape=(),
                    strides=(),
                    max_payload_bytes=8 * 1024 * 1024,
                    encoding="boundary_schema2",
                )
                continue
            try:
                planes = self._resource_planes(resource.resource_id)
            except KeyError:
                # Registration happens after cache binding; a placeholder
                # keeps the resource visible and is replaced on bind.
                self._layouts[resource.resource_id] = ResourceBlockLayout(
                    resource_id=resource.resource_id,
                    cache_name=resource.resource_id,
                    dtype="",
                    shape=(),
                    strides=(),
                    max_payload_bytes=0,
                    encoding="token_range_v1",
                )
                continue
            plane = planes[0]
            block_shape = tuple(int(dim) for dim in plane.shape[1:])
            per_block = 1
            for dim in block_shape:
                per_block *= dim
            per_block *= plane.element_size()
            self._layouts[resource.resource_id] = ResourceBlockLayout(
                resource_id=resource.resource_id,
                cache_name=self._cache_names_for(resource),
                dtype=str(plane.dtype).removeprefix("torch."),
                shape=block_shape,
                strides=tuple(int(s) for s in plane.stride()[1:]),
                max_payload_bytes=(
                    per_block
                    * (self._chunk_size // self._block_size)
                    * len(planes)
                    + 4096
                ),
                encoding="token_range_v1",
            )

    def _copy_stream_for(self, device: torch.device) -> Any:
        with self._lock:
            if self._copy_stream is None or self._device != device:
                self._copy_stream = torch.cuda.Stream(device=device)
                self._device = device
            return self._copy_stream

    def _wait_copy_events(self, event: GroupEvent, stream: Any) -> None:
        for copy_event in event.copy_events():
            if stream is not None and hasattr(stream, "wait_event"):
                stream.wait_event(copy_event)
            else:
                copy_event.synchronize()

    # ------------------------------------------------------------------
    # Bookkeeping helpers
    # ------------------------------------------------------------------

    def _requests_for(
        self, step: StepView, request_ids: Sequence[str]
    ) -> list[TransferRequest]:
        by_id = {request.request_id: request for request in step.requests}
        requests: list[TransferRequest] = []
        for request_id in request_ids:
            request = by_id.get(request_id)
            if request is None:
                raise DSV41GroupLoadError(
                    f"request {request_id} is not part of step {step.step_id}",
                    request_id=request_id,
                    group_id="",
                    error_code=INVALID_SPEC,
                )
            requests.append(request)
        return requests

    def _fail_requests(
        self,
        requests: Sequence[TransferRequest],
        group_id: str,
        *,
        error_code: str,
        message: str,
    ) -> None:
        for request in requests:
            if request.load_prefix_end > 0:
                self._mark_failed_receiving(request.request_id)
                self._counters["group_load_failed"] += 1
                raise DSV41GroupLoadError(
                    message,
                    request_id=request.request_id,
                    group_id=group_id,
                    error_code=error_code,
                )

    def _enqueue(self, job: Any) -> bool:
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            with self._lock:
                self._counters["queue_full"] += 1
            return False

    def _fail_event(
        self,
        event: GroupEvent | None,
        *,
        resource_id: str | None,
        error_code: str,
        message: str,
    ) -> None:
        if event is not None:
            event.fail(
                resource_id=resource_id,
                error_code=error_code,
                message=message,
            )

    def _mark_failed_receiving(self, request_id: str) -> None:
        with self._lock:
            self._failed_receiving.add(request_id)
            self._receiving_ready.discard(request_id)

    def _maybe_finish_receiving_locked(self, request_id: str) -> None:
        expected = self._expected_load_groups.get(request_id, set())
        completed = self._completed_load_groups.get(request_id, set())
        if not expected or completed != expected:
            return
        if request_id in self._failed_receiving:
            return
        if request_id in self._failed_receiving_reported:
            self._failed_receiving_reported.discard(request_id)
        self._receiving_ready.add(request_id)

    def _mark_sending_ready_locked(self, request_id: str) -> None:
        self._sending_ready.add(request_id)

    def _drop_where(
        self,
        predicate: Callable[[tuple[str, str, int]], bool],
        *,
        reason: str,
        counter: str | None = None,
    ) -> None:
        dropped = 0
        with self._lock:
            for (group_id, request_id, step_id), event in list(
                self._load_events.items()
            ):
                if event.is_terminal:
                    continue
                if not predicate((group_id, request_id, step_id)):
                    continue
                event.drop(reason=reason)
                dropped += 1
                self._counters["group_load_dropped"] += 1
                if request_id in self._expected_load_groups:
                    self._mark_failed_receiving(request_id)
            for batch in self._store_batches:
                if batch.event.is_terminal:
                    continue
                if not predicate(
                    (batch.event.group_id, batch.event.request_id, -1)
                ):
                    continue
                batch.event.drop(reason=reason)
                dropped += 1
                self._counters["group_store_dropped"] += 1
            if counter is not None:
                self._counters[counter] += dropped

    def _release_store_buffers(self) -> None:
        with self._lock:
            for batch in self._store_batches:
                batch.staged.clear()

    def _record_worker_failure(
        self, kind: str, payload: Any, exc: BaseException
    ) -> None:
        logger_warning(
            "dsv41 lw v2 adapter worker failed for %s job: %s", kind, exc
        )
        if kind == "load":
            job: _LoadJob = payload
            event = self._load_events.get(
                (job.group.group_id, job.request.request_id, job.step.step_id)
            )
            self._fail_event(
                event,
                resource_id=None,
                error_code=INTERNAL_ERROR,
                message=str(exc),
            )
        elif kind == "store":
            batch: _StoreBatch = payload
            self._fail_event(
                batch.event,
                resource_id=None,
                error_code=INTERNAL_ERROR,
                message=str(exc),
            )

    def _request_by_id(self, request_id: str) -> TransferRequest:
        with self._lock:
            request = self._known_requests_store.get(request_id)
        if request is None:
            raise KeyError(request_id)
        return request

    def _step_for_request(self, request_id: str) -> StepView:
        with self._lock:
            step = self._step_by_request.get(request_id)
        if step is None:
            raise KeyError(request_id)
        return step

    def _require_boundary_resource(self, resource_id: str) -> None:
        if resource_id not in self._boundary_ids:
            raise ValueError(
                f"resource {resource_id!r} is not a registered boundary "
                "resource of this plan"
            )

    # Registries populated by the routing entry points.
    def _remember_requests(self, step: StepView) -> None:
        with self._lock:
            for request in step.requests:
                self._known_requests_store[request.request_id] = request
                self._request_steps[request.request_id] = request.step_id
                self._step_by_request[request.request_id] = step


def logger_warning(message: str, *args: Any) -> None:
    """Emit a module warning through the stock logger when available."""
    try:
        # First Party
        from lmcache.logging import init_logger

        init_logger(__name__).warning(message, *args)
    except Exception:  # pragma: no cover - logging must never mask a failure
        return


# ---------------------------------------------------------------------------
# Token payload framing
# ---------------------------------------------------------------------------

_DTYPE_NAMES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def encode_token_payload(
    *,
    resource_id: str,
    block_ids: Sequence[int],
    planes: Sequence[torch.Tensor],
) -> bytes:
    """Encode one chunk's token-range planes into a validated envelope.

    The envelope is ``magic + u32 header length + JSON header + plane bytes``.
    Every plane is written contiguously in its public dtype; the header keeps
    the exact per-plane shape so a decoder can reject geometry mismatches
    before it allocates or writes anything.
    """
    if not resource_id:
        raise ValueError("resource_id must be nonempty")
    if not planes:
        raise ValueError("a token payload needs at least one plane")
    header_planes = []
    chunks = []
    for plane in planes:
        if not isinstance(plane, torch.Tensor):
            raise TypeError("payload planes must be torch tensors")
        tensor = plane.detach().to("cpu").contiguous()
        raw = tensors_to_bytes(tensor)
        header_planes.append(
            {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "bytes": len(raw),
            }
        )
        chunks.append(raw)
    header = {
        "schema": "token_range_v1",
        "resource_id": resource_id,
        "block_ids": list(block_ids),
        "planes": header_planes,
    }
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    body = b"".join(chunks)
    return (
        TOKEN_PAYLOAD_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + body
    )


def decode_token_payload(
    payload: bytes, *, resource_id: str | None = None
) -> tuple[torch.Tensor, ...]:
    """Decode and validate a token-range envelope.

    Raises:
        ValueError: on truncation, trailing bytes, unknown dtype, bad plane
            sizes or a resource-id mismatch.
    """
    if not payload.startswith(TOKEN_PAYLOAD_MAGIC):
        raise ValueError("token payload has no token-range magic")
    offset = len(TOKEN_PAYLOAD_MAGIC)
    if len(payload) < offset + 4:
        raise ValueError("token payload header length is truncated")
    (header_length,) = struct.unpack(">I", payload[offset : offset + 4])
    offset += 4
    if header_length > 1 << 20:
        raise ValueError("token payload header is unreasonably large")
    if len(payload) < offset + header_length:
        raise ValueError("token payload header is truncated")
    header = json.loads(payload[offset : offset + header_length])
    offset += header_length
    if not isinstance(header, dict) or header.get("schema") != "token_range_v1":
        raise ValueError("token payload header schema is not token_range_v1")
    if resource_id is not None and header.get("resource_id") != resource_id:
        raise ValueError(
            f"token payload is for resource {header.get('resource_id')!r}, "
            f"expected {resource_id!r}"
        )
    planes = []
    for plane_header in header.get("planes", ()):
        dtype_name = plane_header.get("dtype")
        dtype = _DTYPE_NAMES.get(dtype_name)
        if dtype is None:
            raise ValueError(f"unsupported payload dtype {dtype_name!r}")
        shape = tuple(int(dim) for dim in plane_header.get("shape", ()))
        if not shape:
            raise ValueError("payload plane has an empty shape")
        byte_count = int(plane_header.get("bytes", -1))
        itemsize = torch.empty(0, dtype=dtype).element_size()
        expected = itemsize
        for dim in shape:
            expected *= dim
        if expected != byte_count:
            raise ValueError(
                f"payload plane declares {byte_count} bytes but its shape "
                f"needs {expected}"
            )
        if len(payload) < offset + byte_count:
            raise ValueError("token payload plane is truncated")
        raw = payload[offset : offset + byte_count]
        offset += byte_count
        if dtype == torch.bfloat16:
            tensor = torch.frombuffer(
                bytearray(raw), dtype=torch.int16
            ).view(torch.bfloat16).reshape(shape)
        else:
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
        planes.append(tensor)
    if offset != len(payload):
        raise ValueError("token payload has trailing bytes")
    if not planes:
        raise ValueError("token payload has no planes")
    return tuple(planes)


def tensors_to_bytes(tensor: torch.Tensor) -> bytes:
    """Return the raw bytes of a CPU-contiguous tensor."""
    if tensor.device.type != "cpu":
        raise ValueError("tensors_to_bytes expects a CPU tensor")
    contiguous = tensor.contiguous()
    if contiguous.dtype == torch.bfloat16:
        # numpy has no bfloat16 dtype; the bit pattern is what the wire keeps.
        return contiguous.view(torch.int16).numpy().tobytes()
    return contiguous.numpy().tobytes()


@dataclass
class _LoadJob:
    group: TransferGroup
    request: TransferRequest
    step: StepView
    chunks: tuple[_ChunkRef, ...]
