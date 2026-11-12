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
"""

# Standard
import hashlib
import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

# Third Party
from lmcache.logging import init_logger

from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.multiprocess.layerwise_protocol import (
    LAYERWISE_PROTOCOL_VERSION,
    LayerwiseLookupRequest,
    LayerwiseLookupResult,
    LayerwiseRegistration,
    LayerwiseRegistrationAck,
    LayerwiseSnapshotCommit,
    ResourceTransferSpec,
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
    """

    committed_tokens: int
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
        # Steps whose store pass already ran (wait-time or sync path), so a
        # second wait never resubmits the same range (t75 async fix).
        self._wait_store_handled: set[int] = set()
        # Steps whose metadata carried at least one real request entry —
        # the wait-time pass warns once when such a step ends with no
        # recorded store range (chunk granularity miss), while sentinel
        # cold steps stay silent (t75).
        self._step_real_requests: dict[int, set[str]] = {}

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
        assert self.index is not None
        # Steps below the incoming one can no longer schedule or store:
        # their records are retired up front so a long serving run does
        # not accumulate per-step state (T56-04).
        self._retire_steps_below(step_id)
        if requests:
            self._step_real_requests.setdefault(step_id, set()).update(
                requests.keys()
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
                        for resource in required
                    )
                    # Submit before opening admission: the window resolves
                    # as the submitted ops' results arrive.
                    for spec in specs:
                        self.index.submit_transfer(spec)
                    self.begin_step_admission(
                        request_id,
                        step_id,
                        layer,
                        [resource.resource_id for resource in required],
                        specs,
                    )
        except BaseException:
            # A failed schedule must not strand half-open windows or
            # store ranges for the step; the engine will retry the step
            # as a whole (T56-04).
            self._drop_step_records(step_id)
            raise

    def enqueue_step_transports(self, step_id: int) -> None:
        """Dispatch a step's submitted transfers over the MP wire (S0).

        Every submitted-but-undispatched spec of ``step_id`` is sent
        synchronously through the request client: stores carry the real
        per-block tensor bytes, retrieves fetch committed bytes and write
        them back into the registered caches. Each dispatched op settles
        into the runtime index before returning, so admission gates and
        store waits observe terminal state, not promises.

        Fail-closed S0 boundary: exactly one request may be in flight per
        step. A step carrying specs of several requests raises instead of
        interleaving transfers the index cannot order yet; overlapping
        multi-request scheduling arrives with the scheduler-MQ milestone.

        Raises:
            RuntimeError: No runtime registered, or the step carries more
                than one request's transfers (S0 boundary).
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
        requests = sorted({spec.request_id for spec in specs})
        if len(requests) > 1:
            raise RuntimeError(
                "layerwise S0 transport dispatches one request per step; "
                f"step {step_id} carries {len(requests)}: "
                f"{', '.join(requests)}. Overlapping multi-request "
                "scheduling arrives with the scheduler-MQ milestone; "
                "refusing to interleave fail-closed"
            )
        for spec in specs:
            self._dispatch_spec(spec)

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

        Raises:
            RuntimeError: No runtime registered, or the server refused a
                snapshot commit.
        """
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        self._wait_time_store_pass()
        for step_id in sorted(self._uncommitted_store_steps):
            self._commit_step_snapshot(step_id)

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
        by_step: dict[int, list[tuple[str, int, int, tuple]]] = {}
        for (step_id, request_id), store_range in sorted(
            self._step_store_ranges.items()
        ):
            if step_id in self._wait_store_handled:
                continue
            if step_id in self._step_store_ops:
                # The sync-path hook already dispatched this step's stores.
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
                    if not self._store_covers_resource(resource, blocks):
                        # The step's STORE op allocated no blocks for this
                        # resource's engine group (scratch / zero-alloc
                        # group, e.g. the hybrid SWA group the stock
                        # scheduler omits): there is nothing computed for
                        # that group in this step, so nothing to persist.
                        # Skipping is the honest store; inventing empty
                        # blocks would fake coverage.
                        logger.debug(
                            "[dsv41_layerwise] store for request %s skips "
                            "resource %s: engine group %s has no blocks in "
                            "this step's STORE op",
                            request_id,
                            resource.resource_id,
                            resource.engine_group_id,
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

    def _dispatch_spec(self, spec: ResourceTransferSpec) -> None:
        """Send one submitted transfer over the MP request wire (S0).

        Stores extract the real per-block bytes from the registered
        caches; retrieves fetch committed bytes and write them back. The
        settled result is folded into the runtime index; a non-ok status
        is recorded first and then raised, so the engine stops on the
        first failed transfer instead of scheduling on top of it. The one
        exception is a store that loses the server-side single-owner
        election (DUPLICATE_OPERATION): the winning sibling worker already
        stored the key, so the loser skips (returns ``False``) instead of
        failing the step.

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
            payload = self._extract_resource_payload(spec)
            future = self.req_client.store_resource(spec, payload)
            result = future.result(timeout=self._mq_timeout)
        else:
            future = self.req_client.retrieve_resource(spec)
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
            logger.warning(
                "[dsv41_layerwise] store op=%s lost the single-owner "
                "election for (request=%s, step=%d, resource=%s); another "
                "worker already stored this key - skipping",
                spec.operation_id,
                spec.request_id,
                spec.step_id,
                spec.resource_id,
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
                len(payload),
            )
        return True

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
        """Write fetched bytes back into the registered caches (S0).

        Raises:
            RuntimeError: Unknown segment name in the payload header.
        """
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

    def _commit_step_snapshot(self, step_id: int) -> None:
        """Commit one step's dispatched stores as a server snapshot (S0).

        The snapshot id is derived from (plan, epoch, step) so a retried
        commit stays idempotent server-side. Boundaries are empty on
        purpose: without the C2 codec the snapshot is cold-only storage
        that lookups refuse to hit (design 4.0/6).

        Raises:
            RuntimeError: If the server refuses the commit.
        """
        ops = self._step_store_ops.get(step_id)
        if not ops:
            self._uncommitted_store_steps.discard(step_id)
            return
        spec = next(iter(ops.values()))[0]
        # The id stays deterministic per worker (plan/epoch/step plus the
        # per-worker owner tag): sibling TP workers commit their own op
        # sets for the same plan/step, and a shared id collided on the
        # server as "reused with different content" (t75).
        commit = LayerwiseSnapshotCommit(
            snapshot_id=(
                f"snap-{self.plan_hash[:8]}-{self.registration_epoch}"
                f"-{step_id}-{self._op_owner.rstrip('-')}"
            ),
            key=spec.key,
            registration_epoch=self.registration_epoch,
            server_generation=self.server_generation,
            plan_hash=self.plan_hash,
            layout_hash=self.layout_hash,
            operation_ids=tuple(sorted(ops)),
            boundaries=(),  # C2 codec unavailable: cold-only, lookup cannot hit
        )
        ok = self.req_client.commit_layerwise_snapshot(commit).result(
            timeout=self._mq_timeout
        )
        if ok is not True:
            raise RuntimeError(
                f"layerwise snapshot commit for step {step_id} was refused"
            )
        self._uncommitted_store_steps.discard(step_id)
        del self._step_store_ops[step_id]
        logger.info(
            "[dsv41_layerwise] store snapshot committed id=%s ops=%d "
            "boundaries=() (cold-only: lookups cannot hit S0 snapshots)",
            commit.snapshot_id,
            len(commit.operation_ids),
        )

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

        Raises:
            RuntimeError: When the resource references an engine group
                outside the registration, or the recorded groups do not
                cover the resource's group index.
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
            raise RuntimeError(
                f"recorded block groups ({len(per_group_blocks)}) do not "
                f"cover engine group {engine_group_id} at index {index}"
            )
        return per_group_blocks[index]

    def _store_covers_resource(
        self,
        resource: "CacheTransferResource",
        per_group_blocks: tuple[tuple[int, ...], ...],
    ) -> bool:
        """Whether a step's STORE op carries blocks for a resource's group.

        The stock scheduler omits groups with zero allocated tokens from
        the STORE op's per-group block list (e.g. the hybrid SWA scratch
        group), so the recorded list can be shorter than the registered
        engine groups. A store for such a resource has no bytes to move
        this step; callers skip instead of raising coverage.

        Args:
            resource: Token-range resource referenced by the store.
            per_group_blocks: Per-group block ids recorded from the step's
                STORE op.

        Returns:
            True when the recorded groups cover the resource's engine
            group.
        """
        if resource.engine_group_id is None:
            return True
        if resource.engine_group_id not in self._engine_group_ids:
            return False
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
        spec = ResourceTransferSpec(
            operation_id=self.next_operation_id(direction),
            request_id=request_id,
            step_id=step_id,
            registration_epoch=self.registration_epoch,
            server_generation=self.server_generation,
            plan_hash=self.plan_hash,
            layout_hash=self.layout_hash,
            resource_id=resource.resource_id,
            key=f"{request_id}:layerwise",
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
