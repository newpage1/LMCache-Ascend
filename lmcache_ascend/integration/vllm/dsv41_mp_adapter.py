# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 layerwise worker/scheduler adapters (LMCache-Ascend side).

The worker adapter owns the layerwise wiring state of one vLLM worker: the
transfer plan, the attention bindings, the per-resource runtime index and
the admission registry handed to the vLLM-Ascend custom op. The scheduler
adapter validates registration acks and builds prefix lookups. The
message-queue transport itself is deliberately *not* wired in this
milestone: the single seam is :meth:`DSV41LayerwiseMPWorkerAdapter.
enqueue_step_transports`, which raises :class:`NotImplementedError`.
"""

# Standard
import hashlib
import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

# Third Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
)
from lmcache.v1.multiprocess.layerwise_protocol import (
    LAYERWISE_PROTOCOL_VERSION,
    LayerwiseLookupRequest,
    LayerwiseLookupResult,
    ResourceTransferSpec,
)

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

# Reserved wire field until the MP server handshake exists; late protocol
# changes stay cheap because every spec already carries it (design 4.3).
_SERVER_GENERATION_PLACEHOLDER = 0


def compute_plan_hash(plan: "LayerwiseTransferPlan") -> str:
    """Hash the canonical msgspec encoding of ``plan`` (stable identity)."""
    # Third Party
    import msgspec

    return hashlib.sha256(msgspec.json.encode(plan)).hexdigest()


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

        The stock registration RPC runs first; only when it returns is the
        local layerwise state committed and the admission registry bound.
        A failing RPC therefore leaves this adapter unregistered and
        retryable instead of half-committed with a live registry.

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
        self._plan = plan
        self._bindings = bindings
        self._by_dispatch = by_dispatch
        self._by_execution = by_execution
        self._engine_group_ids = next_engine_group_ids
        self.registration_epoch = next_epoch
        self.plan_hash = next_plan_hash
        self.layout_hash = next_layout_hash
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
        return f"{direction}:{next(self._op_counter)}"

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
        admission and stores cover them too (T56-03). Boundary-state and
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
        """Hand a step's submitted transfers to the transport layer.

        **Not wired in this milestone.** The MP transport RPC does not
        exist yet; this method is the single explicit seam between
        scheduling (real, above) and the wire, so nothing silently
        pretends to have moved KV bytes.
        """
        raise NotImplementedError(
            "layerwise transport enqueue is not wired in this milestone "
            "(design section 4.3 transport RPC); transfers are scheduled "
            "and submitted but never moved"
        )

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
        for request_id in window.request_ids:
            store_range = self._step_store_ranges.get(
                (window.step_id, request_id)
            )
            if store_range is None:
                # No STORE op for this request: nothing newly computed to
                # persist (e.g. a fully-served replay).
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
        # Transport seam: the stores are really submitted above; moving
        # them is the one unwired step of this milestone.
        self.enqueue_step_transports(window.step_id)
        return tuple(futures)

    def wait_for_pending_stores(self) -> None:
        """Return only when no store transfer is still pending.

        Fail-closed boundary: without the transport RPC a pending store can
        never settle, so its presence is reported as the unwired-transport
        seam instead of pretending success.

        Raises:
            RuntimeError: No runtime registered.
            NotImplementedError: Stores are still pending and the transport
                is not wired in this milestone.
        """
        if self.index is None:
            raise RuntimeError("layerwise runtime is not registered yet")
        if self.index.pending_request_ids("store"):
            raise NotImplementedError(
                "layerwise stores are pending and the transport is not "
                "wired in this milestone; they can never settle"
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
        admission.clear_admission_registry()


class DSV41LayerwiseMPSchedulerAdapter(LMCacheMPSchedulerAdapter):
    """Scheduler-side adapter: registration identity + prefix lookups.

    The message-queue transport is not wired in this milestone; this class
    delivers the minimal real path: recording the expected registration
    identity, validating the worker ack, and building/applying prefix
    lookups against that identity.
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
