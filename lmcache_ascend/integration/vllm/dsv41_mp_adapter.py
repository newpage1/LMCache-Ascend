# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layerwise worker/scheduler adapters (LMCache-Ascend side).

The worker adapter owns the layerwise wiring state of one vLLM worker: the
transfer plan, the attention bindings, the per-resource runtime index and
the admission registry handed to the vLLM-Ascend custom op. The scheduler
adapter validates registration acks and builds prefix lookups.

The S0 transport closes the loop over the real MP request wire:
registration fetches a real server generation, stores carry per-block
tensor bytes to the server, and dispatched stores are committed as
cold-only snapshots (empty boundaries — lookup cannot hit without the C2
boundary codec, and says so). One request may be in flight per step;
overlapping multi-request scheduling is a fail-closed boundary reserved
for the scheduler-MQ milestone.

Since t94 the boundary pass and the snapshot commits are per request: a
multi-request step stores each attributable request's boundary page and
commits one snapshot per request (own snapshot id, own prefix key), so
sibling requests never overwrite each other server-side and a request
that cannot be attributed stays cold alone.
"""

# Standard
import hashlib
import itertools
import json
import os
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

# Third Party
from lmcache.logging import init_logger

# First Party (path-loaded sibling module)
from lmcache_ascend.integration.vllm.dsv41_prefix_key import (
    dsv41_prefix_key,
)

from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.multiprocess.layerwise_protocol import (
    BOUNDARY_STATE_SCHEMA_V2,
    LAYERWISE_PROTOCOL_VERSION,
    BoundaryStateDescriptor,
    LayerwiseLookupRequest,
    LayerwiseLookupResult,
    LayerwiseRegistration,
    LayerwiseRegistrationAck,
    LayerwiseSnapshotCommit,
    ResourceTransferResult,
    ResourceTransferSpec,
    compute_boundary_checksum,
)

try:  # msgspec.structs.replace needs msgspec >= 0.18
    from msgspec.structs import replace as _msgspec_replace
except Exception:  # pragma: no cover - image ships msgspec 0.22
    _msgspec_replace = None

from lmcache_ascend.integration.vllm.dsv41_boundary_codec import (
    decode_boundary_rows,
    encode_boundary_rows,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    # Third Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.layerwise import (
        CacheTransferResource,
        LayerwiseTransferPlan,
    )
    from lmcache.v1.multiprocess.layerwise_runtime import (
        LayerwiseSnapshotStore,
        LayerwiseTransferIndex,
    )
    from vllm_ascend.models.deepseek_v41.topology import DSV41LayerBinding

# Generation carried before the layerwise registration RPC ran; the real
# server generation replaces it on registration (design 4.3).
_SERVER_GENERATION_PLACEHOLDER = 0


def compute_plan_hash(plan: "LayerwiseTransferPlan") -> str:
    """Hash the canonical msgspec encoding of ``plan`` (stable identity)."""
    # Third Party
    import msgspec

    return hashlib.sha256(msgspec.json.encode(plan)).hexdigest()


def encode_resource_payload(segments: Sequence[tuple[str, Any]]) -> bytes:
    """Frame per-cache tensor slices into self-describing wire bytes (S0).

    Layout: 4-byte big-endian header length, one JSON header
    ``{"v": 1, "segments": [{"name", "dtype", "shape"}, ...]}``, then each
    segment's raw bytes in order. Portability over throughput: the
    torch-only byte path (no numpy) keeps the adapter usable on hosts
    without numpy, and S0 payloads are block-sized, not cache-sized.

    Args:
        segments: ``(name, tensor_slice)`` pairs; slices are copied to
            host memory when they live on a device.

    Returns:
        The framed payload bytes.
    """
    # Third Party
    import torch

    meta: list[dict[str, Any]] = []
    bodies: list[bytes] = []
    for name, slice_ in segments:
        host = slice_.detach().contiguous().cpu()
        # view(uint8) keeps the bytes exact; flatten before tolist() so
        # multi-dimensional slices serialize as one flat byte string.
        raw = host.view(torch.uint8).reshape(-1)
        meta.append(
            {"name": name, "dtype": str(host.dtype), "shape": list(host.shape)}
        )
        bodies.append(bytes(raw.tolist()))
    header = json.dumps({"v": 1, "segments": meta}, sort_keys=True).encode(
        "utf-8"
    )
    return len(header).to_bytes(4, "big") + header + b"".join(bodies)


def decode_resource_payload(
    payload: bytes,
) -> tuple[dict[str, Any], list[Any]]:
    """Unframe :func:`encode_resource_payload` wire bytes.

    Args:
        payload: Framed payload bytes.

    Returns:
        ``(header, tensors)``: the parsed JSON header and one CPU tensor
        per segment, in header order.
    """
    # Third Party
    import torch

    header_len = int.from_bytes(payload[:4], "big")
    header = json.loads(payload[4 : 4 + header_len])
    tensors: list[Any] = []
    offset = 4 + header_len
    for meta in header["segments"]:
        # dtype strings are torch's repr ("torch.float32"): resolve
        # through the attribute table; torch.dtype(str) is not callable.
        dtype = getattr(torch, meta["dtype"].split(".")[-1])
        numel = 1
        for dim in meta["shape"]:
            numel *= dim
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()
        body = payload[offset : offset + nbytes]
        offset += nbytes
        if body:
            tensor = torch.frombuffer(bytearray(body), dtype=dtype)
        else:
            # torch.frombuffer rejects zero-length buffers (empty row
            # selections): materialize an empty tensor of the right shape.
            tensor = torch.empty(tuple(meta["shape"]), dtype=dtype)
        tensors.append(tensor.reshape(meta["shape"]) if meta["shape"] else tensor)
    return header, tensors


@dataclass(frozen=True)
class TensorLayoutDesc:
    """Engine-side tensor geometry of one registered cache tensor.

    Args:
        shape: Logical shape of the tensor.
        dtype: dtype name (e.g. ``"torch.bfloat16"``).
        stride: Stride tuple in elements.
        planes: Registered planes in this tensor; K and scale tensors that
            were registered as one entry with two planes report ``2``.
    """

    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    planes: int = 1


def tensor_layouts_from_kv_caches(
    kv_caches: Mapping[str, Any],
) -> dict[str, TensorLayoutDesc]:
    """Derive the per-tensor layout descriptors from the real tensors.

    A value that is a tuple describes one registered multi-plane entry
    (K + scale): its geometry is read from the first plane and the plane
    count from the tuple length. A value without tensor attributes cannot
    be hashed honestly, so the registration fails closed instead of
    pretending the layout is known.

    Args:
        kv_caches: Ordered name->tensor mapping exactly as registered.

    Returns:
        One :class:`TensorLayoutDesc` per cache name.

    Raises:
        RuntimeError: If a registered value exposes neither tensor
            geometry (``shape``/``dtype``/``stride``) nor a tuple of such
            planes.
    """
    layouts: dict[str, TensorLayoutDesc] = {}
    for name, value in kv_caches.items():
        planes = value if isinstance(value, tuple) else (value,)
        if not planes:
            raise RuntimeError(
                f"cannot derive a tensor layout for {name}: empty tuple "
                "registration has no planes"
            )
        first = planes[0]
        shape = getattr(first, "shape", None)
        dtype = getattr(first, "dtype", None)
        stride = getattr(first, "stride", None)
        if callable(stride):
            # torch.Tensor exposes ``stride`` as a *method* returning the
            # per-dimension tuple; iterating the bound method itself
            # raises TypeError. Calling it keeps real tensors and tuple
            # attribute carriers (fake values in tests) both working
            # (T56-01).
            stride = stride()
        if shape is None or dtype is None or stride is None:
            raise RuntimeError(
                f"cannot derive a tensor layout for {name}: registered "
                "values must expose tensor geometry (shape/dtype/stride)"
            )
        layouts[name] = TensorLayoutDesc(
            shape=tuple(int(dim) for dim in shape),
            dtype=str(dtype),
            stride=tuple(int(step) for step in stride),
            planes=len(planes),
        )
    return layouts


def compute_layout_hash(
    cache_names: Sequence[str],
    engine_group_infos: Sequence["EngineGroupInfo"],
    tensor_layouts: Mapping[str, TensorLayoutDesc] | None = None,
) -> str:
    """Hash the full cache layout that transfers are laid out against.

    The digest changes whenever any fact that changes transfer geometry
    changes: cache identity/order, engine group membership *and* block
    geometry (``tokens_per_block``/``sw_size_tokens``), and, when provided,
    each tensor's shape/dtype/stride/plane count. Two workers whose layouts
    differ must never hash equal, or a receiver would accept blocks laid
    out against a foreign geometry.

    Args:
        cache_names: Registered cache names in registration order.
        engine_group_infos: Engine group view of the registration.
        tensor_layouts: Optional per-tensor geometry; when given it must
            cover every cache name, because a partial view would silently
            weaken the digest.

    Returns:
        The sha256 hex digest of the canonical JSON payload.

    Raises:
        ValueError: If ``tensor_layouts`` is given but does not cover every
            cache name.
    """
    if tensor_layouts is not None:
        missing = [name for name in cache_names if name not in tensor_layouts]
        if missing:
            raise ValueError(f"tensor layouts are missing cache names: {missing}")
    payload: dict[str, Any] = {
        "cache_names": list(cache_names),
        "groups": [
            [
                group.engine_group_id,
                list(group.layer_indices),
                group.tokens_per_block,
                group.sw_size_tokens,
            ]
            for group in engine_group_infos
        ],
    }
    if tensor_layouts is not None:
        payload["tensor_layouts"] = [
            [
                name,
                list(tensor_layouts[name].shape),
                tensor_layouts[name].dtype,
                list(tensor_layouts[name].stride),
                tensor_layouts[name].planes,
            ]
            for name in cache_names
        ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


# Sentinel request id for a step-level cold window (t71): when the stock
# scheduler emits an entirely empty connector metadata (``requests=[]``,
# the engine-warmup / first cold prefill step carries no RETRIEVE and no
# STORE op), the step still needs one declared window per layer so the
# fail-closed forward observes a settled step. The id can never collide
# with engine request ids (chatcmpl-* / req-*).
_COLD_STEP_REQUEST = "__cold_step__"


@dataclass(frozen=True)
class StepRequestView:
    """Per-request transfer geometry of one scheduler step.

    Built by the connector purely from the stock connector-metadata ops
    (``LoadStoreOp`` semantics): the committed token bound, the block ids
    per engine KV cache group and the retrieve window. No engine-side
    index guessing is involved.

    Args:
        committed_tokens: Token bound of the retrieve window (the op
            ``end`` — for stock RETRIEVE ops this is the LMCache hit
            prefix bound, *not* the step's computed-token bound).
        blocks: Block ids indexed by engine KV cache group, in engine
            group order.
        start_token: First token of the retrieve window (op ``start``).
        skip_first_n_tokens: Tokens to skip writing at the beginning of
            the retrieve range (op ``skip_first_n_tokens``).
        end_token: Retrieve window end; defaults to ``committed_tokens``
            when not given.
        has_retrieve: The step metadata carried a RETRIEVE op for this
            request. False means a cold/no-hit request whose admission
            window opens empty (T56-03).
        store_start_token: First token of this step's newly computed KV
            to persist (the STORE op ``start``), when a STORE op exists.
        store_end_token: End of the store range (the STORE op ``end``) —
            the actual committed token range of this step, never the
            RETRIEVE hit bound (T56-02).
        store_blocks: Block ids of the store range, per engine group.
        has_store: The step metadata carried a STORE op for this
            request; a request without one persists nothing.
        prefix_key: Cross-request prefix key injected into the stock
            entry's ``request_configs`` by the wrapper connector
            (``dsv41.prefix_key``; t106). Present when the scheduler
            side derived a stable token-hash key for this request's
            prompt; transfer specs then commit under that key so a
            later request with the same prefix can hit the snapshot.
            ``None`` keeps the request-scoped ``<request>:layerwise``
            key (no cross-request reuse).
    """

    committed_tokens: int
    prefix_key: str | None = None
    hit_identity: dict | None = None
    blocks: tuple[tuple[int, ...], ...] = ()
    start_token: int = 0
    skip_first_n_tokens: int = 0
    end_token: int | None = None
    has_retrieve: bool = False
    store_start_token: int = 0
    store_end_token: int | None = None
    store_blocks: tuple[tuple[int, ...], ...] = ()
    has_store: bool = False

    @property
    def effective_end_token(self) -> int:
        """The retrieve window end actually used for specs."""
        return self.committed_tokens if self.end_token is None else self.end_token

    @property
    def flat_blocks(self) -> tuple[int, ...]:
        """All block ids across groups (order-insensitive views only)."""
        return tuple(block for group in self.blocks for block in group)


class _DispatchAdmissionView:
    """Admission registry handed to the vLLM-Ascend custom op.

    Implements the three hooks of
    ``vllm_ascend.attention.dsv41_layerwise_admission.AdmissionRegistry``
    against the adapter state.
    """

    def __init__(self, adapter: "DSV41LayerwiseMPWorkerAdapter") -> None:
        self._adapter = adapter

    def wait_for_layer(self, dispatch_layer_name: str, timeout: float) -> None:
        """Block until the layer's whole current step is admitted.

        The wait covers *every* request of the layer's newest step window,
        in scheduling order; a batch whose first request is still pending
        blocks even when later requests already settled.

        Raises:
            KeyError: The dispatch key is not part of the registered
                backbone.
            RuntimeError: No admission window was ever opened for the
                layer, or one of the requests is still pending or was
                rejected.
        """
        binding = self._adapter.binding_for_dispatch(dispatch_layer_name)
        windows = self._adapter.current_step_windows(binding.execution_layer_name)
        if not windows:
            raise RuntimeError(
                f"layer {dispatch_layer_name} has never opened an "
                "admission window"
            )
        for request_id, step_id in windows:
            try:
                self._adapter.index.wait_for_admission(
                    request_id,
                    step_id,
                    binding.execution_layer_name,
                    timeout=timeout,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"admission for {dispatch_layer_name} request "
                    f"{request_id} still pending after {timeout}s"
                ) from exc

    def check_admitted(self, dispatch_layer_name: str) -> None:
        """Zero-timeout gate used on the hot forward path."""
        self.wait_for_layer(dispatch_layer_name, timeout=0.0)

    def notify_layer_produced(self, dispatch_layer_name: str) -> None:
        """Record that the layer finished writing (producer side)."""
        self._adapter.notify_layer_produced(dispatch_layer_name)


@dataclass
class _ProducerWindow:
    """The step window a layer last produced, for stores (all requests)."""

    step_id: int = 0
    request_ids: tuple[str, ...] = ()


class DSV41LayerwiseMPWorkerAdapter(LMCacheMPWorkerAdapter):
    """Worker adapter carrying the DSV4.1 layerwise plan and runtime index.

    LMCache-Ascend extension point: stock LMCache creates it through the
    connector's ``create_lmcache_mp_worker_adapter`` factory when the
    ``dsv41.layerwise`` flag is set, so no stock class is modified here.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Layerwise state stays unset until register_layerwise_kv_caches
        # commits it after the stock registration RPC succeeded (T52-12).
        self._plan: "LayerwiseTransferPlan | None" = None
        self._bindings: tuple["DSV41LayerBinding", ...] = ()
        self._by_dispatch: dict[str, "DSV41LayerBinding"] = {}
        self._by_execution: dict[str, "DSV41LayerBinding"] = {}
        # (execution layer, step id) -> request ids in scheduling order;
        # one entry per begin_step_admission call (T54-01).
        self._admission_windows: dict[tuple[str, int], list[str]] = {}
        self._layer_current_step: dict[str, int] = {}
        self._producer_windows: dict[str, _ProducerWindow] = {}
        self._op_counter = itertools.count()
        # Set at registration from the vLLM worker rank: operation ids
        # must be unique across the DP workers of one engine (each worker
        # owns its shard of the KV caches and stores concurrently; a
        # shared counter collided as DUPLICATE_OPERATION on the MP
        # server, t75).
        self._op_owner = ""
        self.index: "LayerwiseTransferIndex | None" = None
        self.snapshots: "LayerwiseSnapshotStore | None" = None
        self.registration_epoch: int = -1
        self.server_generation: int = _SERVER_GENERATION_PLACEHOLDER
        self.plan_hash: str = ""
        self.layout_hash: str = ""
        self._engine_group_ids: tuple[int, ...] = ()
        self._step_specs: dict[tuple[str, int, str, str], ResourceTransferSpec] = {}
        # (step id, request id) -> (start, end, per-group blocks) of the
        # step's committed store range (the STORE op semantics), which is
        # the only source a store spec may use (T56-02).
        self._step_store_ranges: dict[
            tuple[int, str], tuple[int, int, tuple[tuple[int, ...], ...]]
        ] = {}
        self._cache_names: tuple[str, ...] = ()
        # S0 transport dispatch bookkeeping: op ids that already crossed
        # the wire, and the dispatched store ops (spec, future) per step
        # awaiting snapshot commit.
        self._dispatched_ops: set[str] = set()
        self._step_store_ops: dict[
            int, dict[str, tuple[ResourceTransferSpec, Any]]
        ] = {}
        self._uncommitted_store_steps: set[int] = set()
        # t111 run20: a step commits only after the engine sealed it —
        # wait_for_pending_stores marks the step complete. Committing on
        # ``outstanding == 0`` alone raced ahead of later layers of the
        # same step submitting their stores, so the first partial commit
        # published an op set missing whole layers and every later
        # same-id commit hit "reused with different content" (9x
        # layers.1.swa_cache NOT_COMMITTED).
        self._sealed_steps: set[int] = set()
        # Steps whose store pass already ran (wait-time or sync path), so a
        # second wait never resubmits the same range (t75 async fix).
        self._wait_store_handled: set[int] = set()
        # Steps whose metadata carried at least one real request entry —
        # the wait-time pass warns once when such a step ends with no
        # recorded store range (chunk granularity miss), while sentinel
        # cold steps stay silent (t75).
        self._step_real_requests: dict[int, set[str]] = {}
        # t106 throughput guards: per-(request, resource) stored token
        # watermark for overlap dedup, and the lost-election log counter.
        self._stored_token_end: dict[tuple[str, str], int] = {}
        self._lost_election_count: int = 0
        # t106 bridge: request id -> cross-request prefix key delivered
        # through the stock metadata's request_configs.
        self._request_prefix_keys: dict[str, str] = {}
        # t111 run19: exact serving identity per request, injected by the
        # scheduler through request_configs once its lookup HIT.
        self._request_hit_identity: dict[str, dict] = {}
        # t111 run13: cross-DP prefix retrieves. A published snapshot
        # lives in the *storing* DP group's registration state, so a
        # warm request routed to another DP must address its retrieve
        # RPC with that publishing identity instead of its own.
        self._worker_identity_cache: list[dict] | None = None
        self._group_identity_adopted = False
        # S1 boundary support: the engine-side provider of real boundary
        # rows (ring cursor, compression phase, row indices) per
        # boundary-state resource. ``None`` (default) leaves snapshots
        # cold-only — the adapter never invents row indices.
        self._boundary_provider: Any = None
        # step id -> request id -> resource id -> descriptor of the
        # boundary rows dispatched for that step (t94: per request), each
        # request's descriptors folded into that request's snapshot commit.
        self._step_boundary_descriptors: dict[
            int, dict[str, dict[str, BoundaryStateDescriptor]]
        ] = {}
        # t111 async store drain: store RPCs are submitted in the engine
        # thread (cheap enqueue on the client's outbound queue) but settled
        # and committed by a background drain thread, so the engine never
        # pays a synchronous per-store settle RTT (the t103/t106 c01 runs
        # measured ~90s per 8192-token window at 48 in-flight requests).
        # The queue is bounded: backpressure settles the oldest items in
        # the caller's thread instead of dropping them.
        self._drain_pending: "deque[tuple[ResourceTransferSpec, Any, bytes | None, int]]" = deque()
        self._drain_lock = threading.Lock()
        self._drain_wake = threading.Event()
        self._drain_stop = False
        self._drain_thread: threading.Thread | None = None
        self._drain_failures = 0
        # step id -> store settles still in flight (enqueued minus
        # settled). A step commits once its outstanding count hits zero.
        self._step_outstanding_stores: dict[int, int] = {}
        # t111: the engine thread (opportunistic commit after
        # wait_for_pending_stores) and the drain thread both reach
        # _commit_settled_steps; commits are serialized and a commit
        # failure in either thread is logged (honest miss), never
        # raised into the engine.
        self._commit_lock = threading.Lock()
        self._committing_steps: set[int] = set()
        # Steps whose stores already went over the wire in any lane
        # (hook or wait-time). The wait-time pass never resubmits them:
        # after a single-owner election loss the winning sibling worker
        # already stored the key, and a fresh operation id for the same
        # ResourceOpKey is a conflicting re-submission server-side
        # (observed on c01 t111 run2: 45 commit refusals, wedged steps).
        self._step_dispatched: set[int] = set()

    def set_boundary_provider(self, provider: Any) -> None:
        """Install the engine-side boundary-row provider (S1).

        Args:
            provider: Callable ``(resource_id, snapshot_id, boundary_token)
                -> (ring_cursor, compression_phase, row_indices) | None``.
                Returning ``None`` for a resource leaves that resource out
                of the snapshot (a loud warning, never fabricated rows).
                Row indices must be the engine's real slot mapping: the
                C2 codec rejects guesses.

        Provider protocol v2 (t84 full-page snapshots). Optional hooks on
        the provider object enable the page-geometry aware paths:

        * ``provider.ring_rows(resource_id) -> int`` — rows per ring
          page. Required for boundary *stores*: with it the adapter
          frames the complete ring page (schema v2 payload, unwritten
          rows normalized to zero); without it the snapshot stays
          cold-only with a loud warning, exactly like a missing
          provider.
        * ``provider.restore_target(resource_id, boundary_token) ->
          (target_page, ring_rows) | None`` — the new request's own ring
          page for a *restore*. Required for boundary *restores*: the
          captured physical page is provenance only (plan §6), so the
          adapter refuses to write back at the captured rows and
          rebases the full page onto the target page instead. A missing
          hook or ``None`` fails the transfer closed rather than
          restoring onto a guessed page.
        * ``provider.step_end_token(request_id) -> int | None`` — the
          engine's true step end for one request (optional, F5). When
          present the boundary rides that token instead of the store's
          chunk-aligned end, so a boundary only ever leaves the worker
          under the token the engine will match on a later hit.

        Provider protocol v3 (t92/t94 per-request attribution). The
        engine provider may accept a ``request_id`` keyword on both the
        call body and ``restore_target``: the adapter always passes it so
        a multi-request step resolves each request's own rows and own
        restore page. Providers built before t92 take only three
        arguments and keep working — the adapter retries the call
        positionally, preserving their single-request semantics.

        Raises:
            TypeError: If ``provider`` is not callable.
        """
        if not callable(provider):
            raise TypeError("boundary provider must be callable")
        self._boundary_provider = provider

    # ------------------------------------------------------------------
    # Registration (design section 4.1 ordering; T52-12 commit-after-RPC)
    # ------------------------------------------------------------------
    def register_layerwise_kv_caches(
        self,
        kv_caches: Mapping[str, Any],
        *,
        engine_group_infos: Sequence["EngineGroupInfo"],
        layout_hints: Any,
        plan: "LayerwiseTransferPlan",
        bindings: tuple["DSV41LayerBinding", ...],
    ) -> None:
        """Commit the layerwise plan after the stock registration succeeds.

        The stock registration RPC runs first, then the layerwise
        registration RPC fetches the server generation; only when both
        return is the local layerwise state committed and the admission
        registry bound. A failing RPC therefore leaves this adapter
        unregistered and retryable instead of half-committed with a live
        registry.

        The layout digest hashes the real registered tensor geometry
        (:func:`tensor_layouts_from_kv_caches`), so two registrations that
        agree on names and groups but differ in shape/dtype/stride/planes
        can never hash equal (T54-04).

        Args:
            kv_caches: Ordered name->tensor mapping exactly as registered.
            engine_group_infos: Engine group view matching those tensors.
            layout_hints: Opaque stock layout hints (passed through).
            plan: The validated transfer plan for this registration.
            bindings: Attention bindings, in ``plan.layer_order`` order.

        Raises:
            ValueError: On double registration or bindings/order mismatch.
            RuntimeError: If a registered value exposes no tensor geometry.
            Exception: Whatever the stock registration RPC raises; no
                layerwise state is committed in that case (see above).
        """
        if self._plan is not None:
            raise ValueError(
                "DSV4.1 layerwise caches are already active on this worker"
            )
        if tuple(b.execution_layer_name for b in bindings) != tuple(plan.layer_order):
            raise ValueError(
                "bindings must line up with plan.layer_order one-to-one"
            )
        # Local
        from lmcache.v1.multiprocess.layerwise_runtime import (
            LayerwiseSnapshotStore,
            LayerwiseTransferIndex,
        )

        from vllm_ascend.attention import dsv41_layerwise_admission as admission

        by_dispatch = {b.dispatch_layer_name: b for b in bindings}
        by_execution = {b.execution_layer_name: b for b in bindings}
        next_epoch = self.registration_epoch + 1
        next_plan_hash = compute_plan_hash(plan)
        next_layout_hash = compute_layout_hash(
            list(kv_caches),
            engine_group_infos,
            tensor_layouts=tensor_layouts_from_kv_caches(kv_caches),
        )
        next_engine_group_ids = tuple(
            group.engine_group_id for group in engine_group_infos
        )
        # Stock registration RPC first; everything below commits only after
        # it returned, so a raise leaves nothing to roll back.
        super().register_kv_caches(
            dict(kv_caches),
            engine_group_infos=engine_group_infos,
            layout_hints=layout_hints,
        )
        # Layerwise registration RPC: the MP server owns a real server
        # generation for this (plan, layout) identity, replacing the
        # pre-registration placeholder (design 4.1).
        strategy = getattr(self, "parallel_strategy", None)
        vllm_rank = getattr(strategy, "vllm_worker_id", None)
        kv_worker_id = getattr(strategy, "kv_worker_id", None)
        if vllm_rank is None or kv_worker_id is None:
            raise RuntimeError(
                "layerwise registration requires a ParallelStrategy with "
                "vllm_worker_id/kv_worker_id (v1 connector); legacy int "
                "worker ids are not supported"
            )
        registration = LayerwiseRegistration(
            protocol_version=LAYERWISE_PROTOCOL_VERSION,
            plan_hash=next_plan_hash,
            layout_hash=next_layout_hash,
            registration_epoch=next_epoch,
            plan=plan,
            cache_names=tuple(kv_caches),
            vllm_rank=int(vllm_rank),
            kv_worker_id=str(kv_worker_id),
            topology_hash=hashlib.sha256(
                json.dumps(
                    {
                        "engine_group_ids": list(next_engine_group_ids),
                        "cache_names": list(kv_caches),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        )
        ack = self.req_client.register_layerwise_cache(registration).result(
            timeout=self._mq_timeout
        )
        if (
            ack.registration_epoch != next_epoch
            or ack.plan_hash != next_plan_hash
            or ack.layout_hash != next_layout_hash
            or "token_range_host_storage" not in ack.capabilities
        ):
            raise RuntimeError(
                "layerwise registration ack mismatch: "
                f"epoch={ack.registration_epoch} "
                f"plan={ack.plan_hash[:8]} layout={ack.layout_hash[:8]} "
                f"capabilities={sorted(ack.capabilities)}"
            )
        self._plan = plan
        self._bindings = bindings
        self._by_dispatch = by_dispatch
        self._by_execution = by_execution
        self._engine_group_ids = next_engine_group_ids
        self._cache_names = tuple(kv_caches)
        # The ack's server generation is unique per registering worker
        # process (the server increments it for every registration), so it
        # disambiguates even the TP workers that share one vllm_rank.
        self._op_owner = f"g{int(ack.server_generation)}-"
        self.registration_epoch = next_epoch
        self.plan_hash = next_plan_hash
        self.layout_hash = next_layout_hash
        self.server_generation = ack.server_generation
        self.index = LayerwiseTransferIndex(registration_epoch=next_epoch)
        self.snapshots = LayerwiseSnapshotStore()
        admission.set_admission_registry(_DispatchAdmissionView(self))
        self._publish_registration_identity()

    def _publish_registration_identity(self) -> None:
        """Publish this registration identity for the scheduler sidecar.

        The scheduler-role connector runs in a different process and
        cannot see the worker's registration ack, so a prefix lookup
        submitted there has no identity to carry (t111 run8: the B3
        bridge silently never activated because ``_plan_hash`` stayed
        empty). The worker drops a tiny JSON sidecar under
        ``DSV41_LAYERWISE_IDENTITY_DIR`` (default
        ``/tmp/dsv41-layerwise-identity``); the scheduler adapter lazily
        adopts whatever identities exist. No sidecar, no layerwise
        lookup — fail-closed, the stock path is untouched.
        """
        root = os.environ.get(
            "DSV41_LAYERWISE_IDENTITY_DIR",
            "/tmp/dsv41-layerwise-identity",
        )
        try:
            os.makedirs(root, exist_ok=True)
            payload = json.dumps(
                {
                    "plan_hash": self.plan_hash,
                    "layout_hash": self.layout_hash,
                    "registration_epoch": int(self.registration_epoch),
                    "server_generation": int(self.server_generation),
                }
            )
            name = f"{self.plan_hash[:8]}-{self.layout_hash[:8]}.json"
            tmp = os.path.join(root, f".{name}.{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, os.path.join(root, name))
            logger.info(
                "[dsv41_layerwise] published registration identity "
                "plan=%s layout=%s epoch=%d generation=%d to %s/%s "
                "(scheduler-side prefix lookups can now adopt it)",
                self.plan_hash[:8],
                self.layout_hash[:8],
                int(self.registration_epoch),
                int(self.server_generation),
                root,
                name,
            )
        except OSError as exc:
            logger.warning(
                "[dsv41_layerwise] could not publish the registration "
                "identity sidecar (%s); scheduler-side prefix lookups "
                "stay off (fail-closed, stock path unaffected)",
                exc,
            )

    # ------------------------------------------------------------------
    # Binding lookup
    # ------------------------------------------------------------------
    def binding_for_dispatch(self, dispatch_layer_name: str) -> "DSV41LayerBinding":
        """Return the binding owning ``dispatch_layer_name``.

        Raises:
            KeyError: The key is not part of the registered backbone; an
                unknown key must never fall through to another layer.
        """
        try:
            return self._by_dispatch[dispatch_layer_name]
        except KeyError:
            raise KeyError(
                f"{dispatch_layer_name} is not part of the registered "
                "DSV4.1 layerwise backbone"
            ) from None

    def current_step_windows(
        self, execution_layer_name: str
    ) -> tuple[tuple[str, int], ...]:
        """All ``(request_id, step_id)`` windows of the layer's current step.

        A scheduler step schedules a batch of requests through one layer;
        every request of that batch gets its own admission window, in
        scheduling order (T54-01).
        """
        step_id = self._layer_current_step.get(execution_layer_name)
        if step_id is None:
            return ()
        requests = self._admission_windows.get((execution_layer_name, step_id), ())
        return tuple((request_id, step_id) for request_id in requests)

    # ------------------------------------------------------------------
    # Runtime plumbing (delegates to the per-resource index)
    # ------------------------------------------------------------------
    def next_operation_id(self, direction: str) -> str:
        """Fresh operation id; the direction prefix drives index dispatch."""
        return f"{direction}:{self._op_owner}{next(self._op_counter)}"

    def submit_retrieve(self, spec: ResourceTransferSpec) -> Any:
        """Submit one retrieve into the runtime index (idempotent per spec)."""
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        return self.index.submit_transfer(spec)

    def begin_step_admission(
        self,
        request_id: str,
        step_id: int,
        execution_layer_name: str,
        required_resource_ids: Sequence[str],
        retrieve_specs: Sequence[ResourceTransferSpec],
    ) -> None:
        """Open the admission window for one request of one layer step."""
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        self.index.begin_layer_step(
            request_id,
            step_id,
            execution_layer_name,
            required_resource_ids,
            retrieve_specs=retrieve_specs,
        )
        self._layer_current_step[execution_layer_name] = step_id
        self._admission_windows.setdefault(
            (execution_layer_name, step_id), []
        ).append(request_id)

    def complete_transfer(self, result: Any) -> bool:
        """Feed one transfer result into the runtime index."""
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        return self.index.complete_transfer(result)

    # ------------------------------------------------------------------
    # Step scheduling (design section 5.2; called by the connector)
    # ------------------------------------------------------------------
    def schedule_step_retrieves(
        self,
        *,
        step_id: int,
        requests: Mapping[str, StepRequestView],
    ) -> None:
        """Submit retrieves and open admission for every layer of one step.

        For each layer in ``plan.layer_order`` and each request in the
        batch, the token-range resources the layer consumes are submitted
        into the runtime index and the per-request admission window is
        opened, so the custom-op wait covers every request of the step
        (T54-01). Retrieve specs carry the stock op semantics —
        ``start``/``end``/``skip_first_n_tokens`` and the per-engine-group
        block ids (T54-03). Requests without a RETRIEVE op (cold/no-hit)
        still open an empty, immediately-admitted window so forward
        admission and stores cover them too (T56-03). An entirely empty
        ``requests`` mapping — the stock scheduler's metadata for a fully
        cold step such as the engine warmup — opens one step-level
        trivially-admitted window per layer (t71): the step is declared
        and settled, it simply has nothing to wait for. Boundary-state and
        recompute resources are excluded: they are not token-range
        retrievable.

        Args:
            step_id: Scheduler step index.
            requests: request id -> per-request transfer geometry built
                from the stock connector metadata ops.

        Raises:
            RuntimeError: When called before ``register_layerwise_kv_caches``.
        """
        if self._plan is None or self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        # t111 run18: adopt the group identity before ANY spec is built.
        # The hook lane (submit_layer_store during forward) may already
        # have built specs by the time the async start_load_kv lane gets
        # here, so both entry points adopt first.
        self._ensure_group_identity()
        assert self.index is not None
        # Steps below the incoming one can no longer schedule or store:
        # their records are retired up front so a long serving run does
        # not accumulate per-step state (T56-04).
        self._retire_steps_below(step_id)
        if requests:
            self._step_real_requests.setdefault(step_id, set()).update(
                requests.keys()
            )
            for request_id, view in requests.items():
                if view.prefix_key:
                    # t106: keep the latest cross-request key for this
                    # request; specs and snapshots commit under it.
                    if view.prefix_key:
                        self._request_prefix_keys[request_id] = (
                            view.prefix_key
                        )
                        hit_identity = getattr(view, "hit_identity", None)
                        if hit_identity:
                            self._request_hit_identity[request_id] = (
                                hit_identity
                            )
                        logger.info(
                            "[dsv41_layerwise] worker adopted prefix key "
                            "%s for request %s",
                            view.prefix_key,
                            request_id,
                        )
        try:
            for request_id, view in requests.items():
                # The store range comes from the STORE op semantics — the
                # step's actual committed token range — never from the
                # RETRIEVE hit bound (T56-02).
                if view.has_store:
                    if view.store_end_token is None:
                        raise RuntimeError(
                            f"request {request_id} has a store marker but "
                            "no store end token; engine integration "
                            "pending"
                        )
                    self._step_store_ranges[(step_id, request_id)] = (
                        int(view.store_start_token),
                        int(view.store_end_token),
                        view.store_blocks,
                    )
            for layer in self._plan.layer_order:
                required = [
                    resource
                    for resource in self._plan.resources
                    if resource.storage_kind == "token_range"
                    and layer in resource.consumer_layers
                ]
                if not required:
                    continue
                if not requests:
                    # Fully cold step (t71): the stock scheduler metadata
                    # carried no request entry at all, so there is no
                    # per-request geometry. The step is still declared --
                    # one trivially-admitted window for the layer -- so
                    # the fail-closed forward observes a settled step
                    # instead of "never opened an admission window".
                    # No hit is faked and nothing is transferred; store
                    # ops arrive in later steps' metadata.
                    self.begin_step_admission(
                        _COLD_STEP_REQUEST, step_id, layer, [], []
                    )
                    continue
                for request_id, view in requests.items():
                    if not view.has_retrieve:
                        # Cold/no-hit request: nothing to fetch, but the
                        # admission window must still exist so the
                        # forward wait/guard covers the whole batch and a
                        # later store can run (T56-03). An empty window
                        # resolves immediately.
                        self.begin_step_admission(
                            request_id, step_id, layer, [], []
                        )
                        continue
                    # F4: a required resource whose engine group is not
                    # covered by this step's recorded block groups has no
                    # bytes to fetch (zero-allocated groups are omitted
                    # by the stock scheduler) — skip it instead of
                    # building an empty-blocks spec that would fail the
                    # protocol validation and the whole step.
                    covered = [
                        resource
                        for resource in required
                        if self._store_covers_resource(resource, view.blocks)
                    ]
                    skipped = len(required) - len(covered)
                    if skipped:
                        logger.debug(
                            "[dsv41_layerwise] retrieve skip: %d resource(s) "
                            "of request %s in step %d have engine groups not "
                            "covered by the recorded %d block group(s)",
                            skipped,
                            request_id,
                            step_id,
                            len(view.blocks or ()),
                        )
                    specs = tuple(
                        self._step_spec_for(
                            step_id=step_id,
                            request_id=request_id,
                            resource=resource,
                            per_group_blocks=view.blocks,
                            start_token=view.start_token,
                            end_token=view.effective_end_token,
                            skip_first_n_tokens=view.skip_first_n_tokens,
                        )
                        for resource in covered
                    )
                    # Submit before opening admission: the window resolves
                    # as the submitted ops' results arrive. Admission only
                    # requires the COVERED resources: skipped groups have
                    # no transfer to wait for, so requiring them would
                    # leave the window forever unsettled (t109).
                    for spec in specs:
                        self.index.submit_transfer(spec)
                    self.begin_step_admission(
                        request_id,
                        step_id,
                        layer,
                        [resource.resource_id for resource in covered],
                        specs,
                    )
        except BaseException:
            # A failed schedule must not strand half-open windows or
            # store ranges for the step; the engine will retry the step
            # as a whole (T56-04).
            self._drop_step_records(step_id)
            raise

    def enqueue_step_transports(self, step_id: int) -> None:
        """Dispatch a step's submitted transfers over the MP wire.

        Every submitted-but-undispatched spec of ``step_id`` is sent
        synchronously through the request client: stores carry the real
        per-block tensor bytes, retrieves fetch committed bytes and write
        them back into the registered caches. Each dispatched op settles
        into the runtime index before returning, so admission gates and
        store waits observe terminal state, not promises.

        Multi-request steps dispatch since t94. The S0 milestone refused
        to interleave requests a single index could not order; the
        runtime index keys every transfer by (epoch, request, step,
        resource, direction), the server-side single-owner election
        (t75) dedups replicated sibling workers, and snapshot commits
        are per request (t94), so sibling requests of one step now
        dispatch in deterministic (request, resource) order — each with
        its own pages, ranges and prefix key.

        Raises:
            RuntimeError: No runtime registered, or a dispatched
                transfer failed (propagated from the S0 wire path).
        """
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        specs = [
            spec
            for (_, spec_step_id, _, _), spec in self._step_specs.items()
            if spec_step_id == step_id
                and spec.operation_id not in self._dispatched_ops
        ]
        if not specs:
            logger.debug(
                "[dsv41_layerwise] step %s has no pending transfers to "
                "dispatch over the wire",
                step_id,
            )
            return
        ordered = sorted(
            specs, key=lambda s: (s.request_id, s.resource_id, s.operation_id)
        )
        if ordered:
            self._step_dispatched.add(step_id)
        # t111: retrieves settle inline - their admission window must
        # close (and bytes land) before the engine's wait_for_layer_load
        # unblocks, and a step carries at most a handful of them.
        for spec in ordered:
            if not spec.operation_id.startswith("store:"):
                self._dispatch_retrieve(spec)
        # Stores only submit here; the bounded background drain settles
        # them and commits the finished steps (t111). Submitting is a
        # cheap enqueue on the client's outbound queue, so the engine
        # thread stops paying a synchronous settle RTT per store spec
        # (t103/t106 measured ~90s per 8192-token window at 48 in-flight
        # requests on that barrier).
        for spec in ordered:
            if spec.operation_id.startswith("store:"):
                payload = self._extract_resource_payload(spec)
                future = self.req_client.store_resource(spec, payload)
                self._enqueue_store_drain(spec, future, payload, len(payload))

    def notify_layer_produced(self, layer_name: str) -> None:
        """Record the producing layer's whole current step window.

        The producer window covers *every* request of the layer's newest
        step, so a later store submits one transfer per request instead
        of only the last one (T54-01).

        Raises:
            KeyError: The layer is not part of the registered backbone.
        """
        binding = self._binding_for_engine_layer(layer_name)
        windows = self.current_step_windows(binding.execution_layer_name)
        if not windows:
            return
        step_id = windows[0][1]
        self._producer_windows[binding.execution_layer_name] = _ProducerWindow(
            step_id=step_id,
            request_ids=tuple(request_id for request_id, _ in windows),
        )

    def submit_layer_store(self, layer_name: str) -> tuple[Any, ...]:
        """Submit stores for every request the layer just produced.

        One store spec per (request, token-range resource the layer
        writes). The token range comes from the step's recorded *store*
        range — the STORE op semantics, i.e. the tokens actually computed
        in this step — never from the RETRIEVE hit bound, so a partial
        hit persists exactly the new KV beyond the hit prefix and a cold
        request persists its full range (T56-02). Requests without a
        recorded store range had no STORE op and persist nothing. The
        specs are really submitted into the runtime index; moving them is
        the transport seam (:meth:`enqueue_step_transports`).

        Args:
            layer_name: Dispatch key or attention prefix of the layer.

        Returns:
            Futures of the submitted store transfers.

        Raises:
            KeyError: The layer is not part of the registered backbone.
            RuntimeError: No runtime, or the layer has no recorded
                producing window.
        """
        if self._plan is None or self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        self._ensure_group_identity()
        if not self._is_sole_group_storer():
            logger.debug(
                "[dsv41_layerwise] skipping hook-lane stores for %s: a "
                "sibling TP worker is the elected sole storer",
                layer_name,
            )
            return ()
        binding = self._binding_for_engine_layer(layer_name)
        window = self._producer_windows.get(binding.execution_layer_name)
        if window is None or not window.request_ids:
            raise RuntimeError(
                f"layer {layer_name} has no recorded producing window; "
                "notify_layer_produced must run after its forward"
            )
        futures = []
        requests_without_store_op = []
        for request_id in window.request_ids:
            store_range = self._step_store_ranges.get(
                (window.step_id, request_id)
            )
            if store_range is None:
                # No STORE op for this request: nothing newly computed to
                # persist (e.g. a fully-served replay).
                requests_without_store_op.append(request_id)
                continue
            start_token, end_token, blocks = store_range
            for resource in self._plan.written_by(binding.execution_layer_name):
                if resource.storage_kind != "token_range":
                    continue
                # t106 overlap dedup, keyed per (request, resource) — see
                # the wait-time pass for the rationale.
                watermark = self._stored_token_end.get(
                    (request_id, resource.resource_id), 0
                )
                if end_token <= watermark:
                    logger.debug(
                        "[dsv41_layerwise] store skip: request %s layer %s "
                        "resource %s range end %d is already stored "
                        "(watermark %d); overlapping range not re-sent",
                        request_id,
                        layer_name,
                        resource.resource_id,
                        end_token,
                        watermark,
                    )
                    continue
                # F4: a resource whose engine group sits outside the
                # groups this step's STORE op recorded has no bytes to
                # move (the stock scheduler omits zero-allocated groups,
                # e.g. the hybrid SWA scratch group). Skip it — the same
                # coverage rule the wait-time pass applies — instead of
                # letting a coverage miss kill the worker.
                if not self._store_covers_resource(resource, blocks):
                    logger.debug(
                        "[dsv41_layerwise] store skip: engine group %s of "
                        "resource %s is not covered by the %d block "
                        "group(s) recorded for request %s in step %d",
                        resource.engine_group_id,
                        resource.resource_id,
                        len(blocks or ()),
                        request_id,
                        window.step_id,
                    )
                    continue
                spec = self._step_spec_for(
                    step_id=window.step_id,
                    request_id=request_id,
                    resource=resource,
                    per_group_blocks=blocks,
                    start_token=start_token,
                    end_token=end_token,
                    skip_first_n_tokens=0,
                    direction="store",
                )
                futures.append(self.index.submit_transfer(spec))
                self._stored_token_end[
                    (request_id, resource.resource_id)
                ] = end_token
                logger.info(
                    "[dsv41_layerwise] store submitted op=%s request=%s "
                    "tokens=[%d, %d) blocks=%d",
                    spec.operation_id,
                    request_id,
                    start_token,
                    end_token,
                    len(blocks or ()),
                )
        if futures:
            logger.info(
                "[dsv41_layerwise] layer %s produced %d store spec(s) for "
                "step %d (requests without a STORE op: %s)",
                layer_name,
                len(futures),
                window.step_id,
                ", ".join(requests_without_store_op) or "none",
            )
        elif requests_without_store_op:
            # Debug-only on the hook path: with async scheduling the
            # engine binds the step's connector metadata *after* the
            # forward (0.30 start-after-forward lane), so this call
            # legitimately observes the previous step's window before the
            # real store range exists. The wait-time pass below is the
            # authoritative store path and warns once, loudly, when a
            # real step ends with no STORE op at all (t75).
            logger.debug(
                "[dsv41_layerwise] hook-path store for step %d saw no "
                "STORE op for request(s) %s yet (async metadata binds "
                "after forward; the wait-time pass owns the store)",
                window.step_id,
                ", ".join(requests_without_store_op),
            )
        # Dispatch moves the submitted stores over the wire now (S0);
        # wait_for_pending_stores later commits them as a snapshot.
        self.enqueue_step_transports(window.step_id)
        return tuple(futures)

    def wait_for_pending_stores(self) -> None:
        """Commit every dispatched store step as a server snapshot (S0).

        Store dispatch is synchronous (see
        :meth:`enqueue_step_transports`), so the wires are already
        settled when this runs; each step with dispatched stores is
        committed to the MP server as a snapshot with empty boundaries —
        S0's honest cold-only contract: the snapshot exists, but without
        the C2 boundary codec a lookup can never hit it, and nothing
        pretends otherwise.

        t111 async contract: store settlement and snapshot commits run in
        the background drain thread; this call submits everything the
        engine knows about (wait-time store pass, boundary pass for steps
        whose stores are already settled), opportunistically settles the
        futures that have already completed (so a fully-synchronous wire
        behaves exactly like the old blocking version), and returns
        without waiting on unresolved futures. The engine thread stops
        paying a synchronous settle RTT per store spec (t103/t106 c01:
        ~90s per 8192-token window at 48 in-flight requests).

        Raises:
            RuntimeError: No runtime registered, or a synchronous part
                (submit / retrieve settle) failed.
        """
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        self._wait_time_store_pass()
        self._settle_ready_drain_inline()
        for (step_id, _rid) in self._step_store_ranges:
            self._sealed_steps.add(step_id)
        self._commit_settled_steps()

    def _store_step_boundaries_for_commit(self, step_id: int) -> None:
        """Run the boundary pass for a step about to commit (S1).

        Sources the step's store ranges for the boundary token; a step
        without recorded ranges (sentinel step) commits cold-only.
        """
        assert self._plan is not None
        entries = [
            (request_id, *self._step_store_ranges[(step_id, request_id)])
            for (sid, request_id) in sorted(self._step_store_ranges)
            if sid == step_id
        ]
        if entries:
            self._store_step_boundaries(step_id, entries)
        else:
            self._step_boundary_descriptors.setdefault(step_id, {})

    def _wait_time_store_pass(self) -> None:
        """Submit and dispatch the steps whose stores the hook could not.

        With async scheduling the engine runs the step's forward *before*
        :meth:`schedule_step_retrieves` binds the connector metadata (the
        0.30 start-after-forward lane), so the per-layer hook observes the
        previous step's producer window and cannot submit yet. By the time
        the engine calls ``wait_for_save`` the metadata is bound: every
        recorded ``(step, request)`` store range is submitted for the full
        plan backbone — exactly the bytes the forward just wrote into the
        registered caches — dispatched over the MP wire, and left for the
        commit loop as a server snapshot. Steps the hook already handled
        (sync lane) are skipped, a step never stores twice, and a step
        that carried real requests but recorded no STORE range warns once,
        loudly: chunk granularity miss, not a silent hole (t65/t75).

        Raises:
            RuntimeError: No runtime registered, or dispatch/commit
                refused (propagated from the S0 wire path).
        """
        assert self._plan is not None and self.index is not None
        if not self._is_sole_group_storer():
            logger.debug(
                "[dsv41_layerwise] skipping wait-time stores: a sibling "
                "TP worker (lower server_generation in the same DP "
                "group) is the elected sole storer"
            )
            self._wait_store_handled.update(
                step_id
                for (step_id, _request_id) in self._step_store_ranges
            )
            return
        by_step: dict[int, list[tuple[str, int, int, tuple]]] = {}
        for (step_id, request_id), store_range in sorted(
            self._step_store_ranges.items()
        ):
            if step_id in self._wait_store_handled:
                continue
            if (
                step_id in self._step_store_ops
                # t111: the hook lane (or an earlier wait pass) already
                # queued this step's stores for the background drain.
                or self._step_outstanding_stores.get(step_id)
                # t111: any lane already put this step's stores on the
                # wire (including election losers - the winner stored
                # the key; a resubmit is a conflicting re-submission).
                or step_id in self._step_dispatched
            ):
                # Stores already on the wire for this step.
                self._wait_store_handled.add(step_id)
                continue
            by_step.setdefault(step_id, []).append(
                (request_id, *store_range)
            )
        for step_id in sorted(by_step):
            specs = []
            for request_id, start_token, end_token, blocks in by_step[
                step_id
            ]:
                for resource in self._plan.resources:
                    if resource.storage_kind != "token_range":
                        continue
                    # t106 overlap dedup, keyed per (request, resource):
                    # the same token range re-presented on later steps
                    # (decode steps re-present the stored prompt range)
                    # must not re-send bytes the server already holds —
                    # the t103 c01 run re-sent 27 unique ranges as ~527
                    # ops. A DIFFERENT resource of the same range is not
                    # covered: each resource moves its own bytes.
                    watermark = self._stored_token_end.get(
                        (request_id, resource.resource_id), 0
                    )
                    if end_token <= watermark:
                        logger.debug(
                            "[dsv41_layerwise] store skip: request %s "
                            "step %d resource %s range end %d is already "
                            "stored (watermark %d); overlapping range "
                            "not re-sent",
                            request_id,
                            step_id,
                            resource.resource_id,
                            end_token,
                            watermark,
                        )
                        continue
                    if not self._store_covers_resource(resource, blocks):
                        # The step's STORE op allocated no blocks for this
                        # resource's engine group (scratch / zero-alloc
                        # group, e.g. the hybrid SWA group the stock
                        # scheduler omits): there is nothing computed for
                        # that group in this step, so nothing to persist.
                        # Skipping is the honest store; inventing empty
                        # blocks would fake coverage.
                        logger.debug(
                            "[dsv41_layerwise] store skip: request %s "
                            "resource %s engine group %s has no blocks in "
                            "this step's STORE op (step %d)",
                            request_id,
                            resource.resource_id,
                            resource.engine_group_id,
                            step_id,
                        )
                        continue
                    spec = self._step_spec_for(
                        step_id=step_id,
                        request_id=request_id,
                        resource=resource,
                        per_group_blocks=blocks,
                        start_token=start_token,
                        end_token=end_token,
                        skip_first_n_tokens=0,
                        direction="store",
                    )
                    specs.append(spec)
                    self.index.submit_transfer(spec)
                    self._stored_token_end[
                        (request_id, resource.resource_id)
                    ] = end_token
                    logger.info(
                        "[dsv41_layerwise] store submitted (wait-time) "
                        "op=%s request=%s tokens=[%d, %d) blocks=%d",
                        spec.operation_id,
                        request_id,
                        start_token,
                        end_token,
                        len(blocks or ()),
                    )
            self._wait_store_handled.add(step_id)
            if specs:
                self.enqueue_step_transports(step_id)
        for step_id in sorted(self._step_real_requests):
            if step_id in self._wait_store_handled:
                continue
            recorded = any(
                sid == step_id for (sid, _) in self._step_store_ranges
            )
            if not recorded:
                logger.warning(
                    "[dsv41_layerwise] step %d carried real request(s) %s "
                    "but the connector metadata had no STORE op; no KV "
                    "stored (chunk granularity / S0 cold-only)",
                    step_id,
                    ", ".join(sorted(self._step_real_requests[step_id])),
                )
                self._wait_store_handled.add(step_id)

    def _snapshot_id_for(self, step_id: int, request_id: str | None = None) -> str:
        """Deterministic per-worker snapshot id (t75), per request (t94).

        The base id stays derived from (plan, epoch, step, worker owner)
        so a retried commit is idempotent server-side. A named request
        appends an 8-hex digest: sibling requests of the same step
        commit their own snapshots, and the t75 "reused with different
        content" collision can no longer eat a batch mate.
        """
        base = (
            f"snap-{self.plan_hash[:8]}-{self.registration_epoch}"
            f"-{step_id}-{self._op_owner.rstrip('-')}"
        )
        if request_id is None:
            return base
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8]
        return f"{base}-r{digest}"

    def _call_boundary_provider(
        self,
        resource_id: str,
        snapshot_id: str,
        boundary_token: str,
        request_id: str,
    ) -> tuple[int, int, tuple[int, ...]] | None:
        """Ask the provider for one request's boundary rows (t94).

        The engine provider (t92) records boundaries per request and
        accepts a ``request_id`` keyword; naming the request resolves a
        multi-request step to each request's own rows. A provider built
        before t92 takes only the three positional arguments — the
        keyword call fails to bind and the positional retry keeps its
        single-request semantics working.
        """
        assert self._boundary_provider is not None
        try:
            return self._boundary_provider(
                resource_id, snapshot_id, boundary_token, request_id=request_id
            )
        except TypeError:
            return self._boundary_provider(resource_id, snapshot_id, boundary_token)

    def _store_step_boundaries(
        self,
        step_id: int,
        entries: list[tuple[str, int, int, tuple]],
    ) -> None:
        """Encode and dispatch the step's boundary-state pages (S1, t94).

        Boundary rows are the engine-side ring-buffer slots a later hit
        must restore, so they ride the same snapshot as the token-range
        bytes. Rows always come from the installed provider — the
        engine's real slot mapping — and a missing provider, a provider
        ``None``, or a provider without ring geometry leaves the
        resource out of the snapshot with a loud warning: the adapter
        never invents row indices and never guesses pages.

        Every recorded request of the step gets its own boundary spec
        (t94): the provider is called with that request's id, its answer
        frames that request's page under that request's prefix key, and
        a request the provider cannot attribute stays cold alone — the
        rest of the batch is unaffected.

        Stores frame the **complete ring page** (schema v2): the
        provider's written rows prove which slots hold real bytes, the
        unwritten slots travel normalized to zero (plan §6.2), and the
        descriptor's row identity spans the whole page so a restore can
        rebase it onto a new request page without stale tail slots.

        Raises:
            RuntimeError: No runtime registered, or the server refused a
                boundary store outside the single-owner election.
        """
        assert self._plan is not None and self.index is not None
        boundary_resources = [
            resource
            for resource in self._plan.resources
            if resource.storage_kind == "boundary_state"
        ]
        if not boundary_resources:
            return
        if step_id in self._step_boundary_descriptors:
            # Already dispatched for this step (idempotent wait pass).
            return
        if self._boundary_provider is None:
            logger.warning(
                "[dsv41_layerwise] step %d has boundary-state resources "
                "but no boundary provider is installed; snapshot stays "
                "cold-only (no fabricated rows)",
                step_id,
            )
            self._step_boundary_descriptors[step_id] = {}
            return
        ring_rows_hook = getattr(self._boundary_provider, "ring_rows", None)
        if not callable(ring_rows_hook):
            logger.warning(
                "[dsv41_layerwise] boundary provider exposes no ring "
                "geometry (ring_rows hook); a sparse snapshot cannot be "
                "rebased onto a new request page, so step %d stays "
                "cold-only (no fabricated pages)",
                step_id,
            )
            self._step_boundary_descriptors[step_id] = {}
            return
        dispatched: dict[str, dict[str, BoundaryStateDescriptor]] = {}
        # F5 alignment hook (optional, engine-side): when the provider
        # knows the engine's true step end for a request, the boundary
        # rides that token instead of the store's chunk-aligned end.
        step_end_hook = getattr(
            self._boundary_provider, "step_end_token", None
        )
        for request_id, start_token, end_token, _blocks in entries:
            boundary_token = str(end_token)
            if callable(step_end_hook):
                engine_end = step_end_hook(request_id)
                if engine_end is not None and int(engine_end) != end_token:
                    boundary_token = str(int(engine_end))
                    logger.info(
                        "[dsv41_layerwise] boundary token for request %s "
                        "step %d aligned to the engine step end %s "
                        "(store chunk end was %d)",
                        request_id,
                        step_id,
                        boundary_token,
                        end_token,
                    )
            snapshot_id = self._snapshot_id_for(step_id, request_id)
            for resource in boundary_resources:
                provided = self._call_boundary_provider(
                    resource.resource_id, snapshot_id, boundary_token, request_id
                )
                if provided is None:
                    logger.warning(
                        "[dsv41_layerwise] boundary stays cold for "
                        "resource %s of request %s in step %d: the "
                        "provider did not accept store end %s as this "
                        "request's engine step end (a boundary only "
                        "rides a store whose end equals the step end; "
                        "partial-chunk tails stay cold by design — "
                        "cold-only for this request, the rest of the "
                        "batch is unaffected)",
                        resource.resource_id,
                        request_id,
                        step_id,
                        boundary_token,
                    )
                    continue
                ring_cursor, compression_phase, row_indices = provided
                ring_length = int(ring_rows_hook(resource.resource_id))
                tensor_index = resource.tensor_indices[0]
                name = self._cache_names[tensor_index]
                value = self.kv_caches[name]
                tensor = value[0] if isinstance(value, tuple) else value
                descriptor, payload = encode_boundary_rows(
                    tensor,
                    row_indices,
                    snapshot_id=snapshot_id,
                    resource_id=resource.resource_id,
                    boundary_token=boundary_token,
                    ring_cursor=ring_cursor,
                    compression_phase=compression_phase,
                    ring_length=ring_length,
                )
                # t111: the boundary spec is deterministic per (request,
                # step, resource) — a stable operation id over stable
                # fields — so re-entering the boundary pass (engine and
                # drain threads, or a retried commit) re-submits the
                # *identical* spec. Both the client-side and server-side
                # single-owner indexes treat an identical re-submission
                # as idempotent and hand back the original future, where
                # a fresh counter-based operation id made every re-entry
                # a fatal "conflicting re-submission" (t111 run3/run4:
                # 20+20 boundary commit failures per request pair).
                spec = ResourceTransferSpec(
                    operation_id=(
                        f"store:boundary:{request_id}:{step_id}:"
                        f"{resource.resource_id}"
                    ),
                    request_id=request_id,
                    step_id=step_id,
                    registration_epoch=self.registration_epoch,
                    server_generation=self.server_generation,
                    plan_hash=self.plan_hash,
                    layout_hash=self.layout_hash,
                    resource_id=resource.resource_id,
                    key=self._request_prefix_keys.get(request_id)
            or f"{request_id}:layerwise",
                    engine_group_id=None,
                    block_ids=(),
                    start_token=start_token,
                    end_token=end_token,
                    skip_first_n_tokens=0,
                    boundary=descriptor,
                )
                self._step_specs[
                    (request_id, step_id, resource.resource_id, "store")
                ] = spec
                self.index.submit_transfer(spec)
                result = self.req_client.store_resource(spec, payload).result(
                    timeout=self._mq_timeout
                )
                error_code = getattr(result, "error_code", "") or ""
                if (
                    getattr(result, "status", None) == "invalid"
                    and error_code == "DUPLICATE_OPERATION"
                ):
                    logger.warning(
                        "[dsv41_layerwise] boundary store lost the "
                        "single-owner election for (request %s, step %d, "
                        "resource %s); another worker already stored this "
                        "key - skipping",
                        request_id,
                        step_id,
                        resource.resource_id,
                    )
                    self.complete_transfer(result)
                    continue
                if getattr(result, "status", None) != "ok":
                    raise RuntimeError(
                        "boundary store failed with status "
                        f"{getattr(result, 'status', None)} ({error_code})"
                    )
                if not self.complete_transfer(result):
                    # t111: the boundary operation id is deterministic
                    # per (request, step, resource), so a re-entry of the
                    # boundary pass settles the *same* op the first pass
                    # already completed — the client index answers False
                    # (already terminal) for the identical completion.
                    # That is idempotent success, not a stale epoch (the
                    # pre-deterministic-id run5 showed 28 false failures
                    # here); a genuinely stale epoch still fails the
                    # identity checks above.
                    logger.debug(
                        "[dsv41_layerwise] boundary store completion "
                        "already recorded for op=%s (idempotent "
                        "re-entry); continuing",
                        spec.operation_id,
                    )
                self._dispatched_ops.add(spec.operation_id)
                self._step_store_ops.setdefault(step_id, {})[
                    spec.operation_id
                ] = (spec, None)
                self._uncommitted_store_steps.add(step_id)
                dispatched.setdefault(request_id, {})[
                    resource.resource_id
                ] = descriptor
                logger.info(
                    "[dsv41_layerwise] boundary store dispatched over the "
                    "wire op=%s request=%s resource=%s full_page_rows=%d "
                    "written_rows=%d bytes=%d token=%s",
                    spec.operation_id,
                    request_id,
                    resource.resource_id,
                    len(descriptor.row_indices),
                    len(row_indices),
                    len(payload),
                    boundary_token,
                )
        self._step_boundary_descriptors[step_id] = dispatched

    def _published_identities(self) -> list[dict]:
        """Own identity first, then the worker sidecars, newest epoch last.

        Each entry carries the identity fields a retrieve RPC must
        carry for the server to route it to the *publishing*
        registration state (``_state_for`` matches plan, layout and
        epoch). Unreadable sidecars are skipped; a missing directory
        degrades to the own identity only.
        """
        own = {
            "plan_hash": self.plan_hash,
            "layout_hash": self.layout_hash,
            "registration_epoch": self.registration_epoch,
            "server_generation": int(self.server_generation),
        }
        if self._worker_identity_cache is not None:
            return [own, *self._worker_identity_cache]
        root = os.environ.get(
            "DSV41_LAYERWISE_IDENTITY_DIR",
            "/tmp/dsv41-layerwise-identity",
        )
        others: dict[tuple, dict] = {}
        try:
            names = sorted(os.listdir(root))
        except OSError:
            names = []
        for name in names:
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    payload = json.load(fh)
                identity = {
                    "plan_hash": str(payload["plan_hash"]),
                    "layout_hash": str(payload["layout_hash"]),
                    "registration_epoch": int(payload["registration_epoch"]),
                    "server_generation": int(payload["server_generation"]),
                }
            except (OSError, ValueError, KeyError, TypeError):
                continue
            others.setdefault(
                (
                    identity["plan_hash"],
                    identity["layout_hash"],
                    identity["registration_epoch"],
                ),
                identity,
            )
        cache = sorted(
            others.values(), key=lambda i: i["registration_epoch"]
        )
        self._worker_identity_cache = cache
        return [own, *cache]

    def _ensure_group_identity(self) -> None:
        """Adopt the group's canonical registration identity once.

        All sibling TP workers of a DP group register separately, but
        the server keeps ONE registration state per (plan, layout) —
        last registrant wins — so only specs carrying the group's final
        (epoch, generation) pass ``matches()``. Every worker reads the
        group sidecar (written by the last registrant) and adopts that
        identity before building specs; combined with deterministic
        store op ids, all siblings then emit *identical* specs, which
        the server replays idempotently: one complete op set, one
        commit, no election fragmentation (t111 run17 root cause).
        """
        if self._group_identity_adopted:
            return
        self._group_identity_adopted = True
        for identity in self._published_identities():
            if (
                identity["plan_hash"] == self.plan_hash
                and identity["layout_hash"] == self.layout_hash
                and identity["registration_epoch"] != self.registration_epoch
            ):
                logger.info(
                    "[dsv41_layerwise] adopting group identity: epoch "
                    "%d -> %d, generation %s -> %s (sibling TP workers "
                    "emit identical specs against one registration "
                    "state)",
                    self.registration_epoch,
                    identity["registration_epoch"],
                    int(self.server_generation),
                    identity["server_generation"],
                )
                self.registration_epoch = identity["registration_epoch"]
                self.server_generation = identity["server_generation"]
                self._op_owner = f"g{int(self.server_generation)}-"
                self.index = LayerwiseTransferIndex(
                    registration_epoch=self.registration_epoch
                )
                self._dispatched_ops.clear()
                self._step_store_ops.clear()
                self._uncommitted_store_steps.clear()
                self._sealed_steps.clear()
                self._step_specs.clear()
                break

    def _is_sole_group_storer(self) -> bool:
        """Whether this worker is the elected storer for its DP group.

        Sibling TP workers of one DP group share a registration state,
        and the server elects one owner per (key, request, step,
        resource) op — with every sibling racing, winners scatter across
        workers and each sibling's snapshot commit then lists only its
        own winning ops. A retrieve resolves against *one* commit, so a
        scattered commit reads as NOT_COMMITTED for the resources a
        sibling lost (t111 run16: op -0 hit, op -1 missed on the same
        step). Electing a single storer per group — the registered
        identity of the group with the lowest ``server_generation`` —
        keeps every step's op set in one commit. Workers parse the same
        sidecar directory, so the election is deterministic and needs no
        extra RPC. Losing siblings skip storing (their bytes are
        replicated by construction; the sole storer ships the full set).
        """
        identities = self._published_identities()
        own_key = (self.plan_hash, self.layout_hash)
        peers = [
            i["server_generation"]
            for i in identities
            if (i["plan_hash"], i["layout_hash"]) == own_key
        ]
        if not peers:
            return True
        return int(self.server_generation) == min(peers)

    def _dispatch_retrieve(self, spec: ResourceTransferSpec) -> None:
        """Dispatch one retrieve, settling it inline (admission contract).

        Own-identity retrieves settle through the standard path. A
        *prefix-key* retrieve (cross-request reuse) whose own-identity
        attempt misses retries with every published identity: the
        snapshot was committed under the storing DP group's
        registration state, and only a spec carrying that identity
        reaches it (t111 run13: warm requests round-robin across DP
        ranks, so cross-identity is the norm, not the exception).
        """
        exact = self._request_hit_identity.get(spec.request_id)
        if exact and _msgspec_replace is not None:
            variant = _msgspec_replace(
                spec,
                plan_hash=exact["plan_hash"],
                layout_hash=exact["layout_hash"],
                registration_epoch=exact["registration_epoch"],
                server_generation=exact["server_generation"],
            )
            settled = self.req_client.retrieve_resource(variant).result(
                timeout=self._mq_timeout
            )
            result, payload = settled[0], settled[1]
            if result.status == "ok" and payload:
                self._restore_resource_payload(spec, payload)
                ok = ResourceTransferResult(
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    registration_epoch=self.registration_epoch,
                    server_generation=int(self.server_generation),
                    status="ok",
                )
                self.index.complete_transfer(ok)
                self._dispatched_ops.add(spec.operation_id)
                logger.info(
                    "[dsv41_layerwise] exact-identity prefix retrieve "
                    "op=%s request=%s served by the scheduler's hit "
                    "identity %s/%s",
                    spec.operation_id,
                    spec.request_id,
                    exact["plan_hash"][:8],
                    exact["layout_hash"][:8],
                )
                return
            logger.info(
                "[dsv41_layerwise] exact-identity retrieve %s missed "
                "(status=%s); falling back to own + published identities",
                spec.operation_id,
                result.status,
            )
        future = self.req_client.retrieve_resource(spec)
        try:
            self._settle_dispatched_spec(spec, future, None, 0)
            return
        except RuntimeError as exc:
            if not spec.key.startswith("dsv41-"):
                raise
            logger.debug(
                "[dsv41_layerwise] own-identity retrieve %s missed; "
                "trying published identities (%s)",
                spec.operation_id,
                exc,
            )
        # Own attempt missed: the runtime index recorded the failure
        # (complete_transfer ran before the raise), so re-submit before
        # retrying, keeping one live index entry per operation id.
        self.index.submit_transfer(spec)
        for identity in self._published_identities()[1:]:
            if (
                identity["plan_hash"] == self.plan_hash
                and identity["layout_hash"] == self.layout_hash
                and identity["registration_epoch"] == self.registration_epoch
            ):
                continue
            if _msgspec_replace is None:
                break
            variant = _msgspec_replace(
                spec,
                plan_hash=identity["plan_hash"],
                layout_hash=identity["layout_hash"],
                registration_epoch=identity["registration_epoch"],
                server_generation=identity["server_generation"],
            )
            settled = self.req_client.retrieve_resource(variant).result(
                timeout=self._mq_timeout
            )
            result, payload = settled[0], settled[1]
            if result.status == "ok" and payload:
                self._restore_resource_payload(spec, payload)
                ok = ResourceTransferResult(
                    operation_id=spec.operation_id,
                    resource_id=spec.resource_id,
                    registration_epoch=self.registration_epoch,
                    server_generation=int(self.server_generation),
                    status="ok",
                )
                if not self.index.complete_transfer(ok):
                    logger.debug(
                        "[dsv41_layerwise] cross-identity retrieve %s "
                        "completion was ignored (stale or duplicate)",
                        spec.operation_id,
                    )
                self._dispatched_ops.add(spec.operation_id)
                logger.info(
                    "[dsv41_layerwise] cross-identity prefix retrieve "
                    "op=%s request=%s served by published identity "
                    "%s/%s (own layout %s)",
                    spec.operation_id,
                    spec.request_id,
                    identity["plan_hash"][:8],
                    identity["layout_hash"][:8],
                    str(self.layout_hash)[:8],
                )
                return
        miss = ResourceTransferResult(
            operation_id=spec.operation_id,
            resource_id=spec.resource_id,
            registration_epoch=self.registration_epoch,
            server_generation=int(self.server_generation),
            status="miss",
            error_code="NOT_COMMITTED",
        )
        self.index.complete_transfer(miss)
        tried = ", ".join(
            f"{i['plan_hash'][:6]}/{i['layout_hash'][:6]}"
            for i in self._published_identities()
        )
        raise RuntimeError(
            f"layerwise transfer {spec.operation_id} (request="
            f"{spec.request_id} step={spec.step_id} resource="
            f"{spec.resource_id} key={spec.key[:24]}...) failed with "
            "status miss (NOT_COMMITTED) after trying every published "
            f"identity: {tried}"
        )

    def _settle_dispatched_spec(
        self,
        spec: ResourceTransferSpec,
        future: Any,
        payload: bytes | None,
        payload_len: int,
    ) -> bool:
        """Wait one already-submitted transfer and fold in its result.

        The RPC itself went out in :meth:`enqueue_step_transports`'s
        batched submit loop (t106); this settles the future, restores
        retrieved bytes, and records the outcome in the runtime index. A
        non-ok status is recorded first and then raised, so the engine
        stops on the first failed transfer instead of scheduling on top
        of it. The one exception is a store that loses the server-side
        single-owner election (DUPLICATE_OPERATION): the winning sibling
        worker already stored the key, so the loser skips (returns
        ``False``) instead of failing the step.

        Returns:
            True when this worker's transfer completed on the wire;
            False when a store lost the single-owner election and was
            skipped.

        Raises:
            RuntimeError: Wire failure, non-ok status, or the index
                rejected the completion (stale epoch or duplicate).
        """
        assert self.index is not None
        is_store = spec.operation_id.startswith("store:")
        if is_store:
            result = future.result(timeout=self._mq_timeout)
        else:
            settled = future.result(timeout=self._mq_timeout)
            result, payload = settled[0], settled[1]
            if result.status == "ok" and payload:
                self._restore_resource_payload(spec, payload)
        if result.status != "ok" and result.error_code == "DUPLICATE_OPERATION":
            # The MP server's index keys stores by (epoch, request, step,
            # resource, direction): with DP+TP the sibling TP workers of
            # one engine submit the same key for their shards, and the
            # first store wins. Losing a store race means another worker
            # already moved this (request, step, resource) - the S0
            # protocol is single-owner per key. Skip loudly instead of
            # tearing the engine down; this worker simply contributes no
            # bytes for that key (S0 scope: one real owner per store).
            # t106: the t103 c01 run logged 217k of these - one per
            # losing sibling op - which is a log storm, not signal. The
            # first occurrence and every Nth thereafter stay WARNING so
            # operators still see the election working; the rest drop to
            # debug. The running counter is in the message either way.
            self._lost_election_count += 1
            count = self._lost_election_count
            if count == 1 or count % 200 == 0:
                logger.warning(
                    "[dsv41_layerwise] store op=%s lost the single-owner "
                    "election for (request=%s, step=%d, resource=%s); "
                    "another worker already stored this key - skipping "
                    "(occurrence #%d)",
                    spec.operation_id,
                    spec.request_id,
                    spec.step_id,
                    spec.resource_id,
                    count,
                )
            else:
                logger.debug(
                    "[dsv41_layerwise] store op=%s lost the single-owner "
                    "election (occurrence #%d); skipping",
                    spec.operation_id,
                    count,
                )
            self.index.complete_transfer(result)
            return False
        if not self.index.complete_transfer(result):
            raise RuntimeError(
                f"layerwise transfer {spec.operation_id} completion was "
                "ignored (stale epoch or duplicate); refusing to continue"
            )
        if result.status != "ok":
            raise RuntimeError(
                f"layerwise transfer {spec.operation_id} failed with "
                f"status {result.status}"
                + (f" ({result.error_code})" if result.error_code else "")
            )
        self._dispatched_ops.add(spec.operation_id)
        if is_store:
            self._step_store_ops.setdefault(spec.step_id, {})[
                spec.operation_id
            ] = (spec, future)
            self._uncommitted_store_steps.add(spec.step_id)
            logger.info(
                "[dsv41_layerwise] store dispatched over the wire op=%s "
                "request=%s tokens=[%d, %d) blocks=%d bytes=%d",
                spec.operation_id,
                spec.request_id,
                spec.start_token,
                spec.end_token,
                len(spec.block_ids or ()),
                payload_len,
            )
        return True

    # ------------------------------------------------------------------
    # t111: bounded background drain for store settlement and commits.
    # ------------------------------------------------------------------
    _DRAIN_QUEUE_MAX = 256
    _DRAIN_LOG_EVERY = 50

    def _ensure_drain_thread(self) -> None:
        """Start the drain thread on first use (never at import time)."""
        if self._drain_thread is not None and self._drain_thread.is_alive():
            return
        self._drain_stop = False
        self._drain_wake.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_loop,
            name="dsv41-layerwise-store-drain",
            daemon=True,
        )
        self._drain_thread.start()

    def _enqueue_store_drain(
        self,
        spec: ResourceTransferSpec,
        future: Any,
        payload: bytes | None,
        payload_len: int,
    ) -> None:
        """Queue one already-submitted store for background settlement.

        Bounded: when the queue is at capacity the oldest items are
        settled in the caller's thread first (honest backpressure — the
        engine blocks before the queue grows without limit, never by
        silently dropping a store).
        """
        if getattr(future, "query", None) is not None and future.query():
            # Already resolved: settle inline (no wait, just bookkeeping —
            # the barrier t111 removes is the *wait*, not the folding).
            self._settle_drain_item((spec, future, payload, payload_len))
            return
        self._ensure_drain_thread()
        while True:
            with self._drain_lock:
                if len(self._drain_pending) < self._DRAIN_QUEUE_MAX:
                    self._drain_pending.append(
                        (spec, future, payload, payload_len)
                    )
                    self._step_outstanding_stores[spec.step_id] = (
                        self._step_outstanding_stores.get(spec.step_id, 0) + 1
                    )
                    break
                oldest = self._drain_pending.popleft()
            self._settle_drain_item(oldest)
        self._drain_wake.set()

    def _settle_ready_drain_inline(self) -> None:
        """Settle queued stores whose futures already resolved (fast path).

        Keeps fully-synchronous wires (tests, a responsive server)
        behaving like the pre-t111 blocking implementation: by the time
        ``wait_for_pending_stores`` returns, every already-completed
        store is settled.
        """
        while True:
            with self._drain_lock:
                ready = None
                for item in self._drain_pending:
                    if getattr(item[1], "query", None)():
                        self._drain_pending.remove(item)
                        ready = item
                        break
                if ready is None:
                    return
            self._settle_drain_item(ready)

    def _settle_drain_item(
        self, item: tuple[ResourceTransferSpec, Any, bytes | None, int]
    ) -> None:
        """Settle one queued store and mark its step one step closer to commit."""
        spec = item[0]
        try:
            self._settle_dispatched_spec(spec, item[1], item[2], item[3])
        except Exception as exc:  # noqa: BLE001 - background settle: log + drop
            # The engine already moved on (async contract); a failed store
            # is recorded loudly and its snapshot simply never commits —
            # the next lookup misses honestly. Nothing is fabricated.
            self._drain_failures += 1
            if (
                self._drain_failures == 1
                or self._drain_failures % self._DRAIN_LOG_EVERY == 0
            ):
                logger.error(
                    "[dsv41_layerwise] background store settle failed "
                    "(occurrence #%d): op=%s request=%s step=%d: %s; that "
                    "snapshot will not commit (honest miss next lookup)",
                    self._drain_failures,
                    spec.operation_id,
                    spec.request_id,
                    spec.step_id,
                    exc,
                )
        finally:
            outstanding = self._step_outstanding_stores.get(spec.step_id, 0)
            if outstanding <= 1:
                self._step_outstanding_stores.pop(spec.step_id, None)
            else:
                self._step_outstanding_stores[spec.step_id] = outstanding - 1

    def _drain_loop(self) -> None:
        """Background: settle queued stores, commit finished steps."""
        while True:
            with self._drain_lock:
                item = (
                    self._drain_pending.popleft()
                    if self._drain_pending
                    else None
                )
            if item is None:
                if self._drain_stop:
                    return
                self._drain_wake.wait(timeout=0.05)
                self._drain_wake.clear()
                continue
            self._settle_drain_item(item)
            try:
                # _commit_settled_steps is serialized and fail-soft; the
                # drain thread keeps looping whatever a commit did.
                self._commit_settled_steps()
            except Exception as exc:  # noqa: BLE001 - never kill drain
                self._drain_failures += 1
                logger.error(
                    "[dsv41_layerwise] background commit pass failed "
                    "(occurrence #%d): %s",
                    self._drain_failures,
                    exc,
                )

    def _commit_settled_steps(self) -> None:
        """Commit every step whose queued stores have all settled.

        Serialized across the engine and drain threads; a step already
        committing under the other thread is skipped. Commit failures
        (server refusal, RPC timeout) are logged and dropped — that
        snapshot simply never commits, the next lookup misses honestly,
        and the engine never dies on a background store.
        """
        with self._commit_lock:
            ready = [
                step_id
                for step_id in sorted(self._uncommitted_store_steps)
                if not self._step_outstanding_stores.get(step_id)
                and step_id in self._sealed_steps
                and step_id not in self._committing_steps
            ]
            self._committing_steps.update(ready)
        try:
            for step_id in ready:
                try:
                    # S1: fold the step's boundary rows (engine-provided)
                    # into the snapshot before the commit, whichever lane
                    # dispatched the token-range stores. The engine thread
                    # only spends a short bounded wait here; a slow or
                    # refused commit lands in the except path (honest
                    # miss) instead of wedging the engine for the full
                    # RPC timeout (t111 run2: 300s per refused commit).
                    wait = (
                        self._DRAIN_COMMIT_WAIT_S
                        if threading.current_thread()
                        is self._drain_thread
                        else self._ENGINE_COMMIT_WAIT_S
                    )
                    self._store_step_boundaries_for_commit(step_id)
                    self._commit_step_snapshot(step_id, wait_timeout=wait)
                except Exception as exc:  # noqa: BLE001 - never kill engine
                    self._drain_failures += 1
                    logger.error(
                        "[dsv41_layerwise] background snapshot commit "
                        "failed (occurrence #%d) for step %d: %s; that "
                        "snapshot stays uncommitted (honest miss next "
                        "lookup)",
                        self._drain_failures,
                        step_id,
                        exc,
                    )
                    # One-shot commit: the server's view of a timed-out or
                    # refused commit is unknowable, and re-submitting the
                    # boundary store under a fresh operation id trips the
                    # client-side single-owner index (t111 run3: retry
                    # storm - 504 commit timeouts and 94 conflicting
                    # re-submissions from re-entering steps). Drop the
                    # step; the lookup misses honestly.
                    self._uncommitted_store_steps.discard(step_id)
                    self._sealed_steps.discard(step_id)
                    self._step_store_ops.pop(step_id, None)
                    self._step_boundary_descriptors.pop(step_id, None)
        finally:
            with self._commit_lock:
                for step_id in ready:
                    self._committing_steps.discard(step_id)

    def _extract_resource_payload(self, spec: ResourceTransferSpec) -> bytes:
        """Frame the real tensor bytes one store moves (S0).

        Multi-plane registrations (K + scale tuples) are framed as one
        segment per plane with a ``name[plane]`` suffix so the decode
        side can address the exact plane again.

        Raises:
            RuntimeError: Unknown resource id or tensor name.
        """
        assert self._plan is not None
        resource = next(
            (
                candidate
                for candidate in self._plan.resources
                if candidate.resource_id == spec.resource_id
            ),
            None,
        )
        if resource is None:
            raise RuntimeError(
                f"unknown layerwise resource {spec.resource_id}"
            )
        rows = list(spec.block_ids)
        segments: list[tuple[str, Any]] = []
        for tensor_index in resource.tensor_indices:
            name = self._cache_names[tensor_index]
            value = self.kv_caches[name]
            planes = value if isinstance(value, tuple) else (value,)
            for plane_index, plane in enumerate(planes):
                segment_name = (
                    name
                    if len(planes) == 1
                    else f"{name}[{plane_index}]"
                )
                segments.append((segment_name, plane[rows] if rows else plane[:0]))
        return encode_resource_payload(segments)

    def _restore_resource_payload(
        self, spec: ResourceTransferSpec, payload: bytes
    ) -> None:
        """Write fetched bytes back into the registered caches (S0/S1).

        Token-range payloads scatter into the addressed cache rows.
        Boundary payloads (S1, schema v2) are **full ring pages**: the
        captured physical page is provenance only (plan §6), so the
        restore asks the provider for the *new request's own target
        page* and copies the whole page there — never back at the
        captured rows, never only the rows the captured step wrote. The
        full-page copy overwrites every target slot (ring tail
        included), so no previous request's bytes survive the rebase.

        Raises:
            RuntimeError: Unknown segment name in the payload header, an
                unknown resource, a v1 (sparse) boundary payload, ring
                integrity mismatches, or no resolvable target page.
        """
        if spec.boundary is not None:
            assert self._plan is not None
            resource = next(
                (
                    candidate
                    for candidate in self._plan.resources
                    if candidate.resource_id == spec.resource_id
                ),
                None,
            )
            if resource is None:
                raise RuntimeError(
                    f"unknown layerwise resource {spec.resource_id}"
                )
            decoded = decode_boundary_rows(payload)
            header = decoded["header"]
            if header.get("v") != 2:
                raise RuntimeError(
                    "boundary payload for resource "
                    f"{spec.resource_id} is v1 (sparse rows): it carries "
                    "no ring geometry, so it cannot be rebased onto a new "
                    "request page - refusing to restore; re-store the "
                    "snapshot with the v2 full-page codec"
                )
            ring_length = int(header["ring_length"])
            page_offset = int(header["page_offset"])
            # Schema integrity: a v2 payload must travel with a v2
            # descriptor. A v1 descriptor cannot prove its ring identity
            # (no geometry fields), so pairing it with a v2 payload is a
            # forged or misrouted frame and is refused fail-closed.
            descriptor = spec.boundary
            if descriptor.schema_version != BOUNDARY_STATE_SCHEMA_V2:
                raise RuntimeError(
                    "boundary descriptor for resource "
                    f"{spec.resource_id} is schema "
                    f"v{descriptor.schema_version} but its payload is v2; "
                    "identity fields cannot be cross-checked - refusing "
                    "to restore"
                )
            # Ring integrity: every identity field the descriptor
            # checksums must agree with the payload header it traveled
            # with; a forged cursor, phase, or page describes a ring
            # state the capture never wrote.
            for field_name, header_key in (
                ("ring_length", "ring_length"),
                ("page_offset", "page_offset"),
                ("ring_cursor", "ring_cursor"),
                ("compression_phase", "compression_phase"),
            ):
                header_value = int(header[header_key])
                descriptor_value = int(getattr(descriptor, field_name))
                if header_value != descriptor_value:
                    raise RuntimeError(
                        f"boundary payload {header_key} {header_value} "
                        f"disagrees with descriptor "
                        f"{descriptor_value} for resource "
                        f"{spec.resource_id}; refusing to restore"
                    )
            # Checksum integrity: recompute over the descriptor's own
            # identity so a hand-edited descriptor cannot ride a payload
            # it was never encoded with.
            expected_checksum = compute_boundary_checksum(
                descriptor.snapshot_id,
                descriptor.resource_id,
                descriptor.boundary_token,
                descriptor.ring_cursor,
                descriptor.compression_phase,
                descriptor.row_indices,
                descriptor.schema_version,
            )
            if descriptor.checksum != expected_checksum:
                raise RuntimeError(
                    "boundary descriptor checksum for resource "
                    f"{spec.resource_id} does not cover its identity "
                    "fields; refusing to restore a forged descriptor"
                )
            # Ring integrity: the descriptor's row identity must span
            # exactly the payload's captured page, and the payload's
            # boundary token must match the descriptor it traveled with.
            descriptor_rows = tuple(spec.boundary.row_indices)
            expected_rows = tuple(range(page_offset, page_offset + ring_length))
            if descriptor_rows != expected_rows:
                raise RuntimeError(
                    f"boundary descriptor rows {descriptor_rows[:4]}... do "
                    f"not span the payload ring page {expected_rows[:4]}... "
                    f"for resource {spec.resource_id}; refusing to restore "
                    "mismatched identity"
                )
            if str(header.get("boundary_token")) != str(
                spec.boundary.boundary_token
            ):
                raise RuntimeError(
                    f"boundary payload token {header.get('boundary_token')!r} "
                    f"disagrees with descriptor token "
                    f"{spec.boundary.boundary_token!r} for resource "
                    f"{spec.resource_id}; refusing to restore"
                )
            target_hook = getattr(
                self._boundary_provider, "restore_target", None
            )
            if not callable(target_hook):
                raise RuntimeError(
                    "the installed boundary provider cannot name a restore "
                    "target page (no restore_target hook); refusing to "
                    "restore boundary bytes onto a guessed page fail-closed"
                )
            # t94: name the restoring request so a multi-request step
            # resolves to this request's own ring page. Pre-t92 hooks take
            # only two arguments and keep their single-request semantics.
            try:
                target = target_hook(
                    spec.resource_id,
                    spec.boundary.boundary_token,
                    request_id=spec.request_id,
                )
            except TypeError:
                target = target_hook(
                    spec.resource_id, spec.boundary.boundary_token
                )
            if target is None:
                raise RuntimeError(
                    "boundary provider returned no target page for "
                    f"resource {spec.resource_id} (no new request page to "
                    "rebase onto); refusing to restore fail-closed"
                )
            target_page, target_ring = int(target[0]), int(target[1])
            if target_ring != ring_length:
                raise RuntimeError(
                    f"restore target ring has {target_ring} rows but the "
                    f"payload page has {ring_length}; cross-topology "
                    "rebase refused fail-closed"
                )
            tensor_index = resource.tensor_indices[0]
            name = self._cache_names[tensor_index]
            value = self.kv_caches[name]
            cache = value[0] if isinstance(value, tuple) else value
            target_start = target_page * ring_length
            target_end = target_start + ring_length
            if target_start < 0 or target_end > int(cache.shape[0]):
                raise RuntimeError(
                    f"restore target page {target_page} rows "
                    f"[{target_start}, {target_end}) exceed the registered "
                    f"cache with {int(cache.shape[0])} rows for resource "
                    f"{spec.resource_id}"
                )
            page = decoded["rows"].to(device=cache.device, dtype=cache.dtype)
            cache[target_start:target_end] = page
            logger.info(
                "[dsv41_layerwise] boundary page restored op=%s "
                "resource=%s captured_page=%d -> target_page=%d "
                "ring_length=%d written_rows=%d token=%s",
                spec.operation_id,
                spec.resource_id,
                page_offset // ring_length,
                target_page,
                ring_length,
                len(header.get("written_rows", ())),
                spec.boundary.boundary_token,
            )
            return
        header, tensors = decode_resource_payload(payload)
        for meta, tensor in zip(header["segments"], tensors, strict=True):
            name = meta["name"]
            plane_index = 0
            if name.endswith("]") and "[" in name:
                base, _, tail = name.rpartition("[")
                name = base
                plane_index = int(tail[:-1])
            if name not in self.kv_caches:
                raise RuntimeError(
                    f"layerwise payload names unknown cache {name!r}"
                )
            value = self.kv_caches[name]
            planes = list(value) if isinstance(value, tuple) else [value]
            target = planes[plane_index]
            rows = list(spec.block_ids)
            if rows:
                target[rows] = tensor.to(device=target.device)

    _ENGINE_COMMIT_WAIT_S = 2.0
    _DRAIN_COMMIT_WAIT_S = 30.0

    def _commit_step_snapshot(
        self, step_id: int, wait_timeout: float | None = None
    ) -> None:
        """Commit one step's dispatched stores as server snapshots (S0).

        One snapshot per request (t94): a multi-request step's ops are
        grouped by request, and each group commits under its own prefix
        key with its own snapshot id (plan/epoch/step/owner plus a
        request digest) carrying that request's boundary descriptors.
        Sibling requests therefore publish independent snapshots — one
        request's boundary can never overwrite a batch mate's — and a
        retried commit stays idempotent server-side. Steps without
        recorded boundaries still commit cold-only: lookups refuse to
        hit them, by design.

        Raises:
            RuntimeError: If the server refuses any request's commit.
        """
        ops = self._step_store_ops.get(step_id)
        if not ops:
            self._uncommitted_store_steps.discard(step_id)
            return
        by_request: dict[str, list[ResourceTransferSpec]] = {}
        for spec, _future in ops.values():
            by_request.setdefault(spec.request_id, []).append(spec)
        for request_id in sorted(by_request):
            request_specs = by_request[request_id]
            # The id stays deterministic per worker and request
            # (plan/epoch/step/owner plus the request digest): sibling
            # TP workers commit their own op sets and sibling requests
            # commit their own snapshots — a shared id collided on the
            # server as "reused with different content" (t75, t94).
            boundaries = tuple(
                sorted(
                    self._step_boundary_descriptors.get(step_id, {})
                    .get(request_id, {})
                    .values(),
                    key=lambda item: item.resource_id,
                )
            )
            commit = LayerwiseSnapshotCommit(
                snapshot_id=self._snapshot_id_for(step_id, request_id),
                key=request_specs[0].key,
                registration_epoch=self.registration_epoch,
                server_generation=self.server_generation,
                plan_hash=self.plan_hash,
                layout_hash=self.layout_hash,
                operation_ids=tuple(
                    sorted(spec.operation_id for spec in request_specs)
                ),
                # S1: boundary descriptors ride the commit when the
                # engine provided real rows; empty boundaries keep the
                # snapshot cold-only (lookups refuse to hit it, by
                # design).
                boundaries=boundaries,
            )
            ok = self.req_client.commit_layerwise_snapshot(commit).result(
                timeout=self._mq_timeout if wait_timeout is None else wait_timeout,
            )
            if ok is not True:
                raise RuntimeError(
                    f"layerwise snapshot commit for step {step_id} "
                    f"request {request_id} was refused"
                )
            logger.info(
                "[dsv41_layerwise] store snapshot committed id=%s "
                "request=%s ops=%d boundaries=%d%s",
                commit.snapshot_id,
                request_id,
                len(commit.operation_ids),
                len(commit.boundaries),
                ""
                if commit.boundaries
                else " (cold-only: lookups cannot hit S0-style snapshots)",
            )
        self._uncommitted_store_steps.discard(step_id)
        self._sealed_steps.discard(step_id)
        del self._step_store_ops[step_id]
        self._step_boundary_descriptors.pop(step_id, None)

    def recorded_step_spec(
        self, request_id: str, step_id: int, resource_id: str, direction: str
    ) -> ResourceTransferSpec | None:
        """Return the memoized spec of one transfer slot (inspection seam).

        Args:
            request_id: Request the transfer belongs to.
            step_id: Scheduler step the transfer was scheduled in.
            resource_id: Cache resource the transfer moves.
            direction: ``"retrieve"`` or ``"store"``.

        Returns:
            The spec memoized for that slot, or ``None`` when the slot was
            never scheduled.
        """
        return self._step_specs.get((request_id, step_id, resource_id, direction))

    # ------------------------------------------------------------------
    # State lifecycle (design section 4.5: state must be reclaimable)
    # ------------------------------------------------------------------
    def prune_finished_requests(
        self, request_ids: Iterable[str]
    ) -> tuple[str, ...]:
        """Drop per-request layerwise state once the runtime settled it.

        Requests the runtime index reports as finished (through
        :meth:`LayerwiseTransferIndex.get_finished`) can no longer
        transfer in that direction; their adapter-side records are
        dropped here so repeated requests do not accumulate state
        (T56-04).

        Args:
            request_ids: Candidate requests to settle this call.

        Returns:
            The request ids whose state was dropped this call.

        Raises:
            RuntimeError: No runtime registered.
        """
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        finished_receiving, finished_sending = self.index.get_finished(
            request_ids
        )
        finished = finished_receiving | finished_sending
        for request_id in finished:
            for key in [k for k in self._step_specs if k[0] == request_id]:
                del self._step_specs[key]
            for key in [
                k for k in self._step_store_ranges if k[1] == request_id
            ]:
                del self._step_store_ranges[key]
            for window_key in list(self._admission_windows):
                ids = self._admission_windows[window_key]
                if request_id in ids:
                    self._admission_windows[window_key] = [
                        rid for rid in ids if rid != request_id
                    ]
            for layer_key in list(self._producer_windows):
                window = self._producer_windows[layer_key]
                if request_id in window.request_ids:
                    self._producer_windows[layer_key] = _ProducerWindow(
                        step_id=window.step_id,
                        request_ids=tuple(
                            rid
                            for rid in window.request_ids
                            if rid != request_id
                        ),
                    )
        return tuple(sorted(finished))

    def _retire_steps_below(self, step_id: int) -> None:
        """Drop records of steps strictly below ``step_id`` (T56-04).

        Once a newer step is scheduled, an older step can neither
        schedule nor store anything new; its windows, store ranges and
        memoized specs are dead weight. Already-submitted runtime-index
        operations are *not* touched: they settle through the index's own
        completion/cancel/epoch lifecycle.
        """
        for key in [k for k in self._admission_windows if k[1] < step_id]:
            del self._admission_windows[key]
        for key in [k for k in self._step_store_ranges if k[0] < step_id]:
            del self._step_store_ranges[key]
        for key in [k for k in self._step_specs if k[1] < step_id]:
            del self._step_specs[key]
        for step in [s for s in self._wait_store_handled if s < step_id]:
            self._wait_store_handled.discard(step)
        for step in [s for s in self._step_real_requests if s < step_id]:
            del self._step_real_requests[step]
        for step in [s for s in self._step_boundary_descriptors if s < step_id]:
            del self._step_boundary_descriptors[step]

    def _drop_step_records(self, step_id: int) -> None:
        """Drop every adapter record of one step (schedule rollback).

        Used when :meth:`schedule_step_retrieves` raises midway: a
        half-open step must leave no windows or ranges behind, or the
        engine's retry would collide with stale state (T56-04). The
        runtime index keeps whatever operations were already submitted;
        they settle through the index's own failure lifecycle.
        """
        for key in [k for k in self._admission_windows if k[1] == step_id]:
            del self._admission_windows[key]
        for key in [k for k in self._step_store_ranges if k[0] == step_id]:
            del self._step_store_ranges[key]
        for key in [k for k in self._step_specs if k[1] == step_id]:
            del self._step_specs[key]
        self._wait_store_handled.discard(step_id)
        self._step_real_requests.pop(step_id, None)
        self._step_boundary_descriptors.pop(step_id, None)
        for layer_key in list(self._producer_windows):
            if self._producer_windows[layer_key].step_id == step_id:
                del self._producer_windows[layer_key]
        self._layer_current_step = {
            layer: step
            for layer, step in self._layer_current_step.items()
            if step != step_id
        }

    def _binding_for_engine_layer(self, layer_name: str) -> "DSV41LayerBinding":
        if layer_name in self._by_dispatch:
            return self._by_dispatch[layer_name]
        if layer_name in self._by_execution:
            return self._by_execution[layer_name]
        raise KeyError(
            f"{layer_name} is not part of the registered DSV4.1 layerwise "
            "backbone"
        )

    def _blocks_for_engine_group(
        self,
        per_group_blocks: tuple[tuple[int, ...], ...],
        engine_group_id: int | None,
    ) -> tuple[int, ...]:
        """Select the blocks of one engine group from a per-group record.

        A group the record does not cover has no bytes this step (the
        stock scheduler omits zero-allocated groups, e.g. the hybrid SWA
        scratch group), so the miss is logged and empty blocks are
        returned (F4): a coverage gap must never kill a serving worker.
        Only a group outside the registration remains a hard error —
        that is a configuration mistake, not a scheduling shape.

        Raises:
            RuntimeError: When the resource references an engine group
                outside the registration.
        """
        if engine_group_id is None:
            return ()
        if engine_group_id not in self._engine_group_ids:
            raise RuntimeError(
                f"resource references engine group {engine_group_id} "
                "outside the registered engine groups "
                f"{self._engine_group_ids}"
            )
        index = self._engine_group_ids.index(engine_group_id)
        if index >= len(per_group_blocks):
            logger.warning(
                "[dsv41_layerwise] recorded block groups (%d) do not "
                "cover engine group %s at index %d; treating that group "
                "as empty for this transfer (zero-allocated groups are "
                "omitted by the stock scheduler)",
                len(per_group_blocks),
                engine_group_id,
                index,
            )
            return ()
        return per_group_blocks[index]

    def _store_covers_resource(
        self,
        resource: "CacheTransferResource",
        per_group_blocks: tuple[tuple[int, ...], ...],
    ) -> bool:
        """Whether a step's STORE op carries blocks for a resource's group.

        Two distinct outcomes (t109):

        - The resource's engine group IS registered but the step's STORE
          op recorded no blocks for it (the stock scheduler omits
          zero-allocated groups, e.g. the hybrid SWA scratch group):
          ``False`` — the caller skips the transfer because there are no
          bytes to move this step.
        - The resource's engine group is NOT in the registration at all:
          the plan and the registration disagree, which is a
          configuration error. Failing closed beats silently dropping a
          group nobody registered, so this raises ``RuntimeError``.

        Args:
            resource: Token-range resource referenced by the store.
            per_group_blocks: Per-group block ids recorded from the step's
                STORE op.

        Returns:
            True when the recorded groups cover the resource's engine
            group.

        Raises:
            RuntimeError: When the resource's engine group is not part of
                the registration (the caller must not silently skip it).
        """
        if resource.engine_group_id is None:
            return True
        if resource.engine_group_id not in self._engine_group_ids:
            raise RuntimeError(
                "[dsv41_layerwise] resource "
                f"{resource.resource_id} references engine group "
                f"{resource.engine_group_id}, which is not among the "
                "registered engine groups "
                f"{self._engine_group_ids}; the plan and the engine "
                "registration disagree — refusing to silently skip an "
                "unregistered group"
            )
        index = self._engine_group_ids.index(resource.engine_group_id)
        return index < len(per_group_blocks)

    def _step_spec_for(
        self,
        *,
        step_id: int,
        request_id: str,
        resource: "CacheTransferResource",
        per_group_blocks: tuple[tuple[int, ...], ...],
        start_token: int,
        end_token: int,
        skip_first_n_tokens: int,
        direction: str = "retrieve",
    ) -> ResourceTransferSpec:
        """Return the memoized transfer spec for one transfer slot.

        Memoization is keyed by (request, step, resource, direction): the
        runtime index treats a repeated identical spec as the same future,
        and a store must never reuse a retrieve spec.
        """
        key = (request_id, step_id, resource.resource_id, direction)
        cached = self._step_specs.get(key)
        if cached is not None:
            return cached
        if direction == "store":
            # Deterministic across sibling TP workers: identical specs
            # replay idempotently on the server instead of racing the
            # single-owner election with drifting per-worker counters
            # (t111 run17: per-worker numbering fragmented each step's
            # op set across siblings, so one commit never held them
            # all and retrieves read NOT_COMMITTED).
            operation_id = (
                f"store:{request_id}:{step_id}:{resource.resource_id}"
            )
        else:
            operation_id = self.next_operation_id(direction)
        spec = ResourceTransferSpec(
            operation_id=operation_id,
            request_id=request_id,
            step_id=step_id,
            registration_epoch=self.registration_epoch,
            server_generation=self.server_generation,
            plan_hash=self.plan_hash,
            layout_hash=self.layout_hash,
            resource_id=resource.resource_id,
            key=self._request_prefix_keys.get(request_id)
            or f"{request_id}:layerwise",
            engine_group_id=resource.engine_group_id,
            block_ids=self._blocks_for_engine_group(
                per_group_blocks, resource.engine_group_id
            ),
            start_token=start_token,
            end_token=end_token,
            skip_first_n_tokens=skip_first_n_tokens,
            boundary=None,
        )
        self._step_specs[key] = spec
        return spec

    # ------------------------------------------------------------------
    # Engine-facing hooks
    # ------------------------------------------------------------------
    def wait_for_layer_load(self, layer_name: str, timeout: float = 60.0) -> None:
        """Wait until the layer's whole current step is KV-ready.

        Every request of the layer's newest admission window is waited on
        in scheduling order; a batch is admitted only when all of its
        requests are (T54-01).

        Raises:
            KeyError: The layer is unknown or has no admission window yet.
            TimeoutError/RuntimeError: A wait timed out or a request's
                step was rejected (fail-closed).
        """
        binding = self._binding_for_engine_layer(layer_name)
        windows = self.current_step_windows(binding.execution_layer_name)
        if not windows:
            raise KeyError(
                f"layer {layer_name} has no layerwise admission window yet"
            )
        assert self.index is not None
        for request_id, step_id in windows:
            self.index.wait_for_admission(
                request_id, step_id, binding.execution_layer_name, timeout=timeout
            )

    def shutdown_layerwise(self) -> None:
        """Drop every layerwise runtime state (worker shutdown).

        Clears the process-level admission registry *and* all per-step /
        per-request maps this adapter owns, so a shutdown worker leaks no
        layerwise state (T56-04). Registration identity (plan/hash/epoch)
        stays committed so a double registration on the same adapter is
        still refused.

        Idempotent: clearing an unset registry and empty maps is a no-op,
        and the connector's ``shutdown`` routes here before the stock
        shutdown.
        """
        # Local
        from vllm_ascend.attention import dsv41_layerwise_admission as admission

        self._admission_windows.clear()
        self._layer_current_step.clear()
        self._producer_windows.clear()
        self._step_specs.clear()
        self._step_store_ranges.clear()
        self._dispatched_ops.clear()
        self._step_store_ops.clear()
        self._uncommitted_store_steps.clear()
        self._step_boundary_descriptors.clear()
        self._stored_token_end.clear()
        self._request_prefix_keys.clear()
        self._request_hit_identity.clear()
        self._boundary_provider = None
        # t111: stop the background drain; queued stores are settled
        # synchronously here so shutdown stays deterministic.
        self._drain_stop = True
        self._drain_wake.set()
        thread = self._drain_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        while True:
            with self._drain_lock:
                if not self._drain_pending:
                    break
                item = self._drain_pending.popleft()
            self._settle_drain_item(item)
        self._step_outstanding_stores.clear()
        self._step_dispatched.clear()
        admission.clear_admission_registry()


class DSV41LayerwiseMPSchedulerAdapter(LMCacheMPSchedulerAdapter):
    """Scheduler-side adapter: registration identity + prefix lookups.

    The S0 transport keeps stores cold-only (empty boundaries), so lookups
    built here cannot hit yet; this class still delivers the real path it
    owns: recording the expected registration identity, validating the
    worker ack (now echoing a real server generation), and
    building/applying prefix lookups against that identity.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._plan_hash: str = ""
        self._layout_hash: str = ""
        self._cache_names: tuple[str, ...] = ()
        self._registration_epoch: int = -1
        # t106 bridge: in-flight layerwise lookups keyed by request id —
        # futures, the derived prefix key, and the settled hit (whose
        # lease is released as soon as the hit is consumed).
        self._dsv41_lookups: dict[str, dict[str, Any]] = {}
        # t111: identities adopted from the worker sidecar (plan, layout,
        # epoch); empty until the first lookup finds at least one.
        self._lookup_identities: list[tuple[str, str, int]] = []

    def expect_registration(
        self,
        plan_hash: str,
        layout_hash: str,
        cache_names: tuple[str, ...] = (),
    ) -> None:
        """Record the registration identity the worker must echo back.

        Args:
            plan_hash: Digest of the transfer plan the scheduler expects.
            layout_hash: Digest of the cache layout the scheduler expects.
            cache_names: Registered cache names from the worker
                registration (the wire :class:`LayerwiseRegistration`
                carries them); lookup results naming anything outside
                this set are rejected. Empty means the names are not
                known yet and no scope check is applied.
        """
        self._plan_hash = plan_hash
        self._layout_hash = layout_hash
        self._cache_names = tuple(cache_names)
        self._registration_epoch += 1

    def validate_registration_ack(self, ack: Any) -> None:
        """Fail unless the ack matches the expected registration identity.

        Raises:
            ValueError: On plan/layout hash mismatch; a stale or foreign
                registration must never be accepted.
        """
        if ack.plan_hash != self._plan_hash:
            raise ValueError(
                f"worker ack plan hash {ack.plan_hash} != expected {self._plan_hash}"
            )
        if ack.layout_hash != self._layout_hash:
            raise ValueError(
                f"worker ack layout hash {ack.layout_hash} != expected "
                f"{self._layout_hash}"
            )

    def maybe_submit_lookup_request(
        self,
        request_id: str,
        packed_token_ids: bytes,
        cache_salt: str = "",
        request_configs: dict[str, Any] | None = None,
        reserve_last_token: bool = False,
        start: int = 0,
    ) -> bool:
        """Stock lookup first, plus a layerwise lookup on the same prompt.

        The stock path is untouched: ``super()`` submits the standard
        lookup exactly as before (t106 item ①) and its return value is
        what this returns. Additionally, when a layerwise registration
        identity is expected, a ``lookup_layerwise_prefix`` RPC fan-out
        goes out under the stable ``dsv41_prefix_key`` digest — the same
        key the wrapper connector injects into the stock metadata and
        the worker commits snapshots under — without waiting here.
        """
        submitted = super().maybe_submit_lookup_request(
            request_id,
            packed_token_ids=packed_token_ids,
            cache_salt=cache_salt,
            request_configs=request_configs,
            reserve_last_token=reserve_last_token,
            start=start,
        )
        if request_id in self._dsv41_lookups:
            return submitted
        if not self._lookup_identities:
            self._lookup_identities = self._load_lookup_identities()
            if self._lookup_identities:
                self._plan_hash, self._layout_hash, self._registration_epoch = (
                    self._lookup_identities[0]
                )
                # Cache names are worker-private; the scheduler-side
                # scope check stays off (no names adopted -> no check).
                self._cache_names = ()
                logger.info(
                    "[dsv41_layerwise] adopted %d published layerwise "
                    "registration identit%s for prefix lookups "
                    "(lead plan=%s layout=%s)",
                    len(self._lookup_identities),
                    "y" if len(self._lookup_identities) == 1 else "ies",
                    self._plan_hash[:8],
                    self._layout_hash[:8],
                )
        if not self._lookup_identities:
            return submitted
        key = dsv41_prefix_key(
            self.model_name, cache_salt, packed_token_ids
        )
        entries: list[tuple[str, str, Any]] = []
        for plan_h, layout_h, epoch in self._lookup_identities:
            request = LayerwiseLookupRequest(
                protocol_version=LAYERWISE_PROTOCOL_VERSION,
                plan_hash=plan_h,
                layout_hash=layout_h,
                registration_epoch=epoch,
                key=key,
            )
            for client in self.req_clients.values():
                entries.append(
                    (plan_h, layout_h, client.lookup_layerwise_prefix(request))
                )
        logger.info(
            "[dsv41_layerwise] scheduler layerwise lookup submitted "
            "request=%s key=%s identities=%d rpcs=%d stock_submitted=%s",
            request_id,
            key,
            len(self._lookup_identities),
            len(entries),
            submitted,
        )
        self._dsv41_lookups[request_id] = {
            "key": key,
            "pending": entries,
            "hit": None,
        }
        return submitted

    def _load_lookup_identities(self) -> list[tuple[str, str, int]]:
        """Read the worker-published registration identity sidecars.

        Returns a de-duplicated, deterministic ``(plan, layout, epoch)``
        list; any unreadable or malformed sidecar is skipped (fail-closed
        to "no identity for that file"), and a missing directory yields
        an empty list — the layerwise lookup simply stays off.
        """
        root = os.environ.get(
            "DSV41_LAYERWISE_IDENTITY_DIR",
            "/tmp/dsv41-layerwise-identity",
        )
        found: dict[tuple[str, str, int], None] = {}
        try:
            names = sorted(os.listdir(root))
        except OSError:
            return []
        for name in names:
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as fh:
                    payload = json.load(fh)
                identity = (
                    str(payload["plan_hash"]),
                    str(payload["layout_hash"]),
                    int(payload["registration_epoch"]),
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue
            found.setdefault(identity, None)
        return list(found)

    def check_lookup_result(self, request_id: str) -> int | None:
        """Stock result first; fall through to the layerwise snapshot.

        A stock hit wins outright and the layerwise lookup is dropped.
        While the stock lookup is still pending (``None``) the layerwise
        futures are polled non-blockingly. A layerwise hit is validated
        through :meth:`apply_layerwise_lookup`, reported as the matched
        token count (stock then schedules the RETRIEVE the worker serves
        through the layerwise transport), and its lease is released
        immediately — retrieves never consume the lease. All-layerwise
        miss defers to the stock answer (0 stays 0, None stays None).
        """
        ret = super().check_lookup_result(request_id)
        state = self._dsv41_lookups.get(request_id)
        if state is None:
            return ret
        if ret is not None and ret > 0:
            # The standard storage already answered: drop the layerwise
            # lookup without consuming it.
            self._drop_layerwise_lookup(request_id, drain=True)
            return ret
        still_pending = []
        for entry in state.get("pending", []):
            plan_h, layout_h, future = entry
            if not future.query():
                still_pending.append(entry)
                continue
            result = future.result(timeout=0)
            if result is None:
                continue
            # Validate against the identity this RPC targeted (the
            # scheduler's lead identity may differ when several DP-group
            # layouts published sidecars).
            if result.plan_hash != plan_h or result.layout_hash != layout_h:
                logger.warning(
                    "[dsv41_layerwise] rejecting layerwise lookup result "
                    "for request %s: result identity does not match the "
                    "targeted registration",
                    request_id,
                )
                continue
            state["hit"] = result
            break
        state["pending"] = still_pending
        hit = state.get("hit")
        if hit is not None:
            # The boundary token is the engine's real step end, which is
            # NOT chunk aligned (t111 run12: stock asserts
            # ``ret % lmcache_tokens_per_chunk == 0`` and died on the
            # first genuine hit). Align the matched prefix DOWN to the
            # chunk grid the stock scheduler expects; a sub-chunk prefix
            # aligns to zero and is reported as an honest miss.
            chunk = int(getattr(self, "lmcache_tokens_per_chunk", 0) or 0)
            matched = int(hit.matched_end)
            aligned = matched // chunk * chunk if chunk > 0 else matched
            self._release_lease_quietly(hit.lease_id)
            if aligned <= 0:
                logger.info(
                    "[dsv41_layerwise] scheduler layerwise lookup HIT "
                    "request=%s matched_end=%d but aligns to 0 tokens "
                    "under chunk=%d; reporting a miss",
                    request_id,
                    matched,
                    chunk,
                )
                return ret
            logger.info(
                "[dsv41_layerwise] scheduler layerwise lookup HIT "
                "request=%s matched_end=%d aligned=%d (chunk=%d) "
                "snapshot=%s",
                request_id,
                matched,
                aligned,
                chunk,
                hit.snapshot_id,
            )
            return aligned
        if state["pending"]:
            return None
        logger.info(
            "[dsv41_layerwise] scheduler layerwise lookup MISS "
            "request=%s (no boundary-complete snapshot under the "
            "request's prefix key)",
            request_id,
        )
        return ret

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Stock cleanup plus layerwise lookup drain and lease release."""
        state = self._dsv41_lookups.pop(request_id, None)
        if state is not None:
            hit = state.get("hit")
            if hit is not None:
                self._release_lease_quietly(hit.lease_id)
            else:
                for entry in state.get("pending", []):
                    future = entry[-1]
                    try:
                        result = future.result(timeout=self._mq_timeout)
                    except Exception:  # noqa: BLE001 - best-effort drain
                        continue
                    if result is not None:
                        self._release_lease_quietly(result.lease_id)
        super().cleanup_lookup_result(request_id)

    def _drop_layerwise_lookup(self, request_id: str, drain: bool) -> None:
        """Discard a request's layerwise lookup without using its result."""
        state = self._dsv41_lookups.pop(request_id, None)
        if state is None:
            return
        if not drain:
            return
        for entry in state.get("pending", []):
            future = entry[-1]
            try:
                result = future.result(timeout=self._mq_timeout)
            except Exception:  # noqa: BLE001 - best-effort drain
                continue
            if result is not None:
                self._release_lease_quietly(result.lease_id)

    def _release_lease_quietly(self, lease_id: str | None) -> None:
        """Fan out a lease release; failures are logged, never raised."""
        if not lease_id:
            return
        for client in self.req_clients.values():
            try:
                client.release_layerwise_lease(lease_id)
            except Exception as exc:  # noqa: BLE001 - release is best effort
                logger.debug(
                    "[dsv41_layerwise] lease release failed for %s: %s",
                    lease_id,
                    exc,
                )

    def build_layerwise_lookup(self, key: str) -> LayerwiseLookupRequest:
        """Build the prefix-lookup request for ``key`` (selected RPC).

        Args:
            key: The prefix key to look up.

        Returns:
            The request carrying the registration identity, so the server
            can reject a lookup from a stale epoch/layout.

        Raises:
            RuntimeError: No registration identity recorded yet.
        """
        if not self._plan_hash:
            raise RuntimeError("no registration has been expected yet")
        return LayerwiseLookupRequest(
            protocol_version=LAYERWISE_PROTOCOL_VERSION,
            plan_hash=self._plan_hash,
            layout_hash=self._layout_hash,
            registration_epoch=self._registration_epoch,
            key=key,
        )

    def apply_layerwise_lookup(self, result: LayerwiseLookupResult) -> tuple[str, ...]:
        """Validate a lookup result and return the selected resource ids.

        Raises:
            ValueError: On plan/layout hash mismatch (the result belongs
                to a different registration) or when the result names a
                cache outside the registered set.
        """
        if result.plan_hash != self._plan_hash:
            raise ValueError(
                f"lookup plan hash {result.plan_hash} != expected {self._plan_hash}"
            )
        if result.layout_hash != self._layout_hash:
            raise ValueError(
                f"lookup layout hash {result.layout_hash} != expected "
                f"{self._layout_hash}"
            )
        if self._cache_names:
            outside = set(result.resource_ids) - set(self._cache_names)
            if outside:
                raise ValueError(
                    f"lookup selected caches outside the registered set: "
                    f"{sorted(outside)}"
                )
        return tuple(result.resource_ids)

    def lookup_layerwise_prefix(
        self, key: str, timeout: float | None = None
    ) -> LayerwiseLookupResult | None:
        """Look ``key`` up over the real MP request clients (S1).

        The request goes to every configured server; the first server
        holding a boundary-complete snapshot for this registration
        identity wins and its result is validated through
        :meth:`apply_layerwise_lookup`. A miss on every server returns
        ``None`` — a miss is honest, never fabricated.

        Args:
            key: The prefix key to look up (``"<request>:layerwise"``).
            timeout: Per-server RPC timeout; defaults to the adapter's
                configured MQ timeout.

        Returns:
            The validated lookup result, or ``None`` when every server
            missed.

        Raises:
            RuntimeError: No registration identity recorded yet.
            ValueError: A server answered with a foreign result.
        """
        request = self.build_layerwise_lookup(key)
        for client in self.req_clients.values():
            result = client.lookup_layerwise_prefix(request).result(
                timeout=timeout if timeout is not None else self._mq_timeout
            )
            if result is None:
                continue
            self.apply_layerwise_lookup(result)
            return result
        return None

    def release_layerwise_lease(self, lease_id: str) -> bool:
        """Release a lookup lease on every configured server.

        The lease lives on the server that granted it; the scheduler
        does not track which one, so the release is fanned out and a
        single ``True`` (some server actually released) is reported.

        Args:
            lease_id: Lease id from a prior
                :meth:`lookup_layerwise_prefix` hit.

        Returns:
            ``True`` when at least one server released the lease,
            ``False`` when every server reported it already released.
        """
        released = False
        for client in self.req_clients.values():
            if client.release_layerwise_lease(lease_id).result(
                timeout=self._mq_timeout
            ):
                released = True
        return released
