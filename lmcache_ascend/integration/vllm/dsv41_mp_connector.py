# SPDX-License-Identifier: Apache-2.0
"""DSV41LayerwiseMPConnector: opt-in layerwise KV transfer for DSV4.1.

Registered under its own name (``DSV41LayerwiseMPConnector``); it never
overrides the stock LMCache connector. The mode is enabled exclusively by
``kv_transfer_config.kv_connector_extra_config["dsv41.layerwise"]`` with a
strict boolean ``true`` — no environment variable, no string coercion
(``"true"`` is rejected so a YAML quoting mistake cannot silently enable
the path).

With the flag absent or ``False`` every method defers to the stock
:class:`LMCacheMPConnector` behavior bit-for-bit. With the flag ``True``:

- ``register_kv_caches`` runs the stock layout/group derivation unchanged,
  then builds the layerwise plan *after* the group derivation and *before*
  any layerwise state exists, binds it to the live attention objects
  explicitly (no ``v41_attn``→``attn`` string rewriting), and only then
  lets the stock registration RPC run;
- the forward path runs over the real wire (S0): retrieves and stores
  are genuinely scheduled from the stock connector-metadata ops,
  submitted into the runtime index and dispatched one request per step
  to the server (:meth:`enqueue_step_transports`); stores commit as
  cold-only snapshots (empty boundaries), so lookups miss fail-closed
  until the C2 boundary codec exists;
- ``shutdown`` drops the layerwise admission registry before running
  the stock shutdown (idempotent).
"""

# Standard
import dataclasses
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

# Third Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

try:  # Third Party
    from msgspec.structs import replace as msgspec_structs_replace
except ImportError:  # pragma: no cover - msgspec ships with lmcache
    msgspec_structs_replace = None

# First Party (path-loaded sibling module)
from lmcache_ascend.integration.vllm.dsv41_prefix_key import (
    dsv41_prefix_key,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _annotate_entry_configs(entry: Any, configs: dict) -> Any:
    """Rebuild a metadata entry with new request_configs.

    Stock entries are plain dataclasses (``LMCacheMPRequestMetadata``),
    so ``dataclasses.replace`` is the primary path; msgspec structs (if
    a future stock version switches) fall back to
    ``msgspec.structs.replace``. Returns ``None`` when neither applies —
    the caller skips that entry (best effort, t111 run11: msgspec-only
    replace silently failed on the dataclass and zero entries were
    annotated).
    """
    if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
        return dataclasses.replace(entry, request_configs=configs)
    try:
        return msgspec_structs_replace(entry, request_configs=configs)
    except Exception:  # noqa: BLE001 - annotation is best effort
        return None

DSV41_LAYERWISE_FLAG = "dsv41.layerwise"


def parse_layerwise_flag(extra_config: Mapping[str, Any] | None) -> bool:
    """Parse the layerwise switch with strict boolean semantics.

    Args:
        extra_config: ``kv_connector_extra_config`` mapping (may be None).

    Returns:
        The flag value; ``False`` when absent.

    Raises:
        ValueError: If the value is present but not a real ``bool`` —
            ``"true"``, ``1``, ``None``, etc. are all rejected so a config
            formatting mistake cannot half-enable the mode.
    """
    if extra_config is None or DSV41_LAYERWISE_FLAG not in extra_config:
        return False
    value = extra_config[DSV41_LAYERWISE_FLAG]
    if not isinstance(value, bool):
        raise ValueError(
            f"{DSV41_LAYERWISE_FLAG} must be a JSON boolean, got "
            f"{type(value).__name__}: {value!r}"
        )
    return value


def _per_group_blocks(op: Any, request_id: str) -> tuple[tuple[int, ...], ...]:
    """Read one op's block ids as a per-engine-group tuple.

    The stock op carries ``block_ids`` indexed by engine KV cache group
    (one inner list per group); the IPC path may flatten a single-group
    nested list to a plain int list, which stock also treats as the
    single-group shape — that form is kept as exactly one group. A
    resource belonging to a later group then fails closed in the
    adapter (the recorded groups do not cover its index), never by
    positional guessing (T54-03).

    Args:
        op: The stock ``LoadStoreOp`` of one RETRIEVE entry.
        request_id: Request the op belongs to (error context only).

    Returns:
        Block ids per engine group, in group order.

    Raises:
        RuntimeError: If ``block_ids`` is missing or a flattened list
            pretends to carry more than one group.
    """
    raw = getattr(op, "block_ids", None)
    if raw is None:
        raise RuntimeError(
            f"connector op for {request_id} has no block_ids: the "
            "vllm-ascend scheduler must attach per-engine-group block "
            "ids to every RETRIEVE/STORE op (list[int] for one group, "
            "list[list[int]] per group)"
        )
    if not raw:
        return ()
    if isinstance(raw[0], int):
        return (tuple(int(block) for block in raw),)
    return tuple(tuple(int(block) for block in group) for group in raw)


def _model_root_from_context(static_forward_context: Mapping[str, Any]) -> str:
    """Derive the plan prefix from a live attention module's own prefix."""
    for module in static_forward_context.values():
        prefix = getattr(module, "prefix", None)
        if getattr(module, "layerwise_execution_layer", None) is None:
            continue
        if not isinstance(prefix, str):
            raise ValueError("DSV4.1 attention module has no string prefix")
        return prefix.rsplit(".layers.", 1)[0]
    raise ValueError(
        "no DSV4.1 attention modules in the static forward context; "
        "the layerwise connector requires a DSV4.1 backbone"
    )


class DSV41LayerwiseMPConnector(LMCacheMPConnector):
    """Stock LMCache MP connector plus opt-in DSV4.1 layerwise execution."""

    def __init__(
        self,
        vllm_config: Any,
        role: Any,
        kv_cache_config: Any = None,
    ) -> None:
        """Construct the connector for one scheduler or worker role.

        Args:
            vllm_config: Engine configuration.
            role: Scheduler or worker role.
            kv_cache_config: Resolved cache groups, if supplied.

        Raises:
            ValueError: If the flag is not a strict boolean, or layerwise
                mode is enabled together with an unsupported companion
                feature (eager prefetch, lazy offload, intermediate-tensor
                dispatch, speculative decoding).
        """
        transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        extra = (
            transfer_config.kv_connector_extra_config
            if transfer_config is not None
            else None
        )
        self._dsv41_layerwise = parse_layerwise_flag(extra)
        if self._dsv41_layerwise:
            if transfer_config is None:
                raise ValueError("dsv41.layerwise requires a kv_transfer_config")
            unsupported = {
                "lmcache.mp.transfer_intermediate_tensors": (
                    "intermediate-tensor dispatch"
                ),
                "lmcache.mp.eager_prefetch": "eager prefetch",
                "lmcache.mp.lazy_offload": "lazy offload",
            }
            for key, label in unsupported.items():
                if transfer_config.get_from_extra_config(key, False):
                    raise ValueError(
                        f"dsv41.layerwise cannot be combined with {label} ({key})"
                    )
            spec_config = getattr(vllm_config, "speculative_config", None)
            spec_tokens = getattr(spec_config, "num_speculative_tokens", 0)
            if spec_tokens:
                raise ValueError(
                    "dsv41.layerwise does not support speculative decoding yet "
                    f"(num_speculative_tokens={spec_tokens})"
                )
        super().__init__(vllm_config, role, kv_cache_config)
        # Scheduler-step counter for the layerwise retrieve scheduling; one
        # increment per start_load_kv call.
        self._layerwise_step = -1

    # ------------------------------------------------------------------
    # Adapter factories (override points added to the stock connector)
    # ------------------------------------------------------------------

    def create_lmcache_mp_worker_adapter(self, **adapter_kwargs: Any) -> Any:
        """Build the worker adapter; layerwise only when the flag is on."""
        if not self._dsv41_layerwise:
            return super().create_lmcache_mp_worker_adapter(**adapter_kwargs)
        # First Party
        from lmcache_ascend.integration.vllm.dsv41_mp_adapter import (
            DSV41LayerwiseMPWorkerAdapter,
        )

        return DSV41LayerwiseMPWorkerAdapter(**adapter_kwargs)

    def create_lmcache_mp_scheduler_adapter(self, **adapter_kwargs: Any) -> Any:
        """Build the scheduler adapter; layerwise only when the flag is on."""
        if not self._dsv41_layerwise:
            return super().create_lmcache_mp_scheduler_adapter(**adapter_kwargs)
        # First Party
        from lmcache_ascend.integration.vllm.dsv41_mp_adapter import (
            DSV41LayerwiseMPSchedulerAdapter,
        )

        return DSV41LayerwiseMPSchedulerAdapter(**adapter_kwargs)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Register caches; layerwise plan built between groups and RPC.

        Off path: stock behavior, untouched. On path: stock layout/group
        derivation first (identical helpers, identical arguments), then the
        plan and explicit bindings, then the layerwise adapter state, then
        the stock registration RPC — plan construction never reorders the
        registered tensor list or rewrites tuple multi-plane entries.
        """
        if not self._dsv41_layerwise:
            super().register_kv_caches(kv_caches)
            return
        # Deferred: these modules pull the full vLLM stack and are absent in
        # CPU-side harnesses; the on path only runs inside the engine.
        # Third Party
        from lmcache.integration.vllm.kv_cache_group_edits import (
            apply_kv_cache_group_edits,
        )
        from lmcache.integration.vllm.kv_cache_groups import (
            create_engine_group_infos_from_vllm,
        )
        from lmcache.integration.vllm.utils import vllm_layout_hints

        # First Party
        from lmcache_ascend.integration.vllm.dsv41_mp_adapter import (
            DSV41LayerwiseMPWorkerAdapter,
        )
        from lmcache_ascend.integration.vllm.layerwise import (
            build_dsv41_layer_bindings,
            build_dsv41_layerwise_plan,
        )
        from vllm_ascend.models.deepseek_v41.topology import build_layer_plan

        kv_cache_config = getattr(self, "_kv_cache_config", None)
        layout_hints = vllm_layout_hints(self._vllm_config)
        kv_caches = apply_kv_cache_group_edits(
            kv_cache_config, kv_caches, layout_hints=layout_hints
        )
        engine_group_infos = tuple(
            create_engine_group_infos_from_vllm(
                kv_cache_config,
                kv_caches,
                layout_hints=layout_hints,
                dcp_size=self._dcp_size,
            )
        )
        text_config = self._vllm_config.model_config.hf_text_config
        topology = build_layer_plan(text_config)
        static_forward_context = (
            self._vllm_config.compilation_config.static_forward_context
        )
        prefix = _model_root_from_context(static_forward_context)
        plan = build_dsv41_layerwise_plan(
            topology,
            tuple(kv_caches.keys()),
            engine_group_infos,
            prefix=prefix,
        )
        bindings = build_dsv41_layer_bindings(
            topology, static_forward_context, plan
        )
        worker = self.worker_adapter
        if not isinstance(worker, DSV41LayerwiseMPWorkerAdapter):
            raise ValueError(
                "layerwise registration requires the DSV41LayerwiseMPWorkerAdapter"
            )
        worker.register_layerwise_kv_caches(
            kv_caches,
            engine_group_infos=engine_group_infos,
            layout_hints=layout_hints,
            plan=plan,
            bindings=bindings,
        )
        if self.dispatcher is not None:
            # Third Party
            from lmcache.integration.vllm.experimental import dispatch

            dispatch(
                self.dispatcher,
                "register",
                kv_caches=kv_caches,
                kv_cache_config=kv_cache_config,
                vllm_config=self._vllm_config,
            )

    # ------------------------------------------------------------------
    # Forward-path hooks
    # ------------------------------------------------------------------

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the layer's transfers resolved (off: stock no-op)."""
        if not self._dsv41_layerwise:
            super().wait_for_layer_load(layer_name)
            return
        self.worker_adapter.wait_for_layer_load(layer_name)

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        """Schedule the step's layerwise retrieves, then dispatch them.

        The per-request transfer geometry is taken from the stock connector
        metadata ops — ``start``/``end``/``skip_first_n_tokens`` and the
        per-engine-group ``block_ids`` — never from positional alignment
        with engine objects (T54-03). Scheduling really submits the
        retrieve specs and opens one admission window per request, then
        :meth:`enqueue_step_transports` moves them over the MP wire (S0).
        """
        if not self._dsv41_layerwise:
            super().start_load_kv(forward_context, **kwargs)
            return
        # t114: run the stock batched retrieve path first. Its
        # RETRIEVE entries route into the worker's bridged
        # ``submit_retrieve_request`` (paged-KV load + finished report);
        # without this call a layerwise hit never satisfies the stock
        # external-hit state machine and the request parks in
        # WAITING_FOR_REMOTE_KVS forever (t111 Blocker 3).
        super().start_load_kv(forward_context, **kwargs)
        worker = self.worker_adapter
        metadata = self._get_connector_metadata()
        requests = self._layerwise_step_view(metadata)
        self._layerwise_step += 1
        worker.schedule_step_retrieves(
            step_id=self._layerwise_step,
            requests=requests,
        )
        worker.enqueue_step_transports(self._layerwise_step)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        """Submit the layer's stores and dispatch them over the wire."""
        if not self._dsv41_layerwise:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return
        self.worker_adapter.submit_layer_store(layer_name)

    def wait_for_save(self) -> None:
        """Commit dispatched stores as server snapshots (S0 flush)."""
        if not self._dsv41_layerwise:
            super().wait_for_save()
            return
        self.worker_adapter.wait_for_pending_stores()

    def shutdown(self) -> None:
        """Drop the layerwise admission state, then run the stock shutdown.

        Idempotent: ``shutdown_layerwise`` clears the process-level
        admission registry (a no-op when already cleared) before the
        stock connector shutdown runs (T54-05). The off path runs the
        stock shutdown unchanged.
        """
        if self._dsv41_layerwise:
            worker = getattr(self, "worker_adapter", None)
            if worker is not None and hasattr(worker, "shutdown_layerwise"):
                worker.shutdown_layerwise()
        super().shutdown()

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        """Stock metadata plus the t106 cross-request prefix key.

        The stock entries are built unchanged (``super()``), then every
        RETRIEVE/STORE entry whose request tracker holds prompt tokens
        is rebuilt with ``request_configs["dsv41.prefix_key"]`` — a
        stable digest over (model, cache salt, packed prompt token ids).
        The worker adapter commits snapshots under that key and the
        scheduler-side layerwise lookup uses the same digest, so a
        second request with the same prefix hits the first one's
        snapshot (t106). Entries without a tracker keep the stock form.
        """
        metadata = super().build_connector_meta(scheduler_output)
        try:
            requests = list(getattr(metadata, "requests", None) or ())
        except TypeError:
            return metadata
        if not requests:
            return metadata
        annotated = False
        for index, entry in enumerate(requests):
            request_id = getattr(entry, "request_id", None)
            if request_id is None:
                continue
            tracker = self._get_request_tracker(request_id)
            packed = tracker.packed_token_ids()
            if not packed:
                continue
            key = dsv41_prefix_key(
                self.scheduler_adapter.model_name,
                tracker.cache_salt,
                packed,
            )
            configs = dict(getattr(entry, "request_configs", None) or {})
            configs["dsv41.prefix_key"] = key
            # t111 run19: when this request's layerwise lookup already
            # HIT, forward the serving registration identity so the
            # worker's retrieves target the exact state that holds the
            # snapshot instead of fanning out over every published one.
            hit = getattr(self.scheduler_adapter, "_dsv41_lookups", {}).get(
                request_id
            )
            if hit and hit.get("hit") is not None:
                result = hit["hit"]
                configs["dsv41.hit_identity"] = {
                    "plan_hash": result.plan_hash,
                    "layout_hash": result.layout_hash,
                    "registration_epoch": getattr(
                        result, "registration_epoch", 0
                    )
                    or 0,
                    "server_generation": int(result.server_generation),
                }
            rebuilt = _annotate_entry_configs(entry, configs)
            if rebuilt is None:
                continue
            requests[index] = rebuilt
            annotated = True
        logger.info(
            "[dsv41_layerwise] build_connector_meta annotated %d/%d "
            "metadata entries with the cross-request prefix key",
            sum(1 for e in requests if "dsv41.prefix_key" in (getattr(e, "request_configs", None) or {})),
            len(requests),
        )
        if annotated:
            try:
                metadata.requests = tuple(requests)
            except AttributeError:
                try:
                    metadata = msgspec_structs_replace(
                        metadata, requests=tuple(requests)
                    )
                except Exception:  # noqa: BLE001 - best effort
                    return metadata
        return metadata

    def _layerwise_step_view(self, metadata: Any) -> dict[str, Any]:
        """Extract the per-request transfer geometry from the stock ops.

        Both directions contribute, keyed by request id, each built
        purely from its ``LoadStoreOp`` — never from positional alignment
        with engine objects (T54-03). A RETRIEVE entry supplies the
        retrieve window (``start``/``end``/``skip_first_n_tokens`` and
        per-engine-group ``block_ids``); a STORE entry supplies the
        step's actual committed store range (``start``/``end``/block
        ids), which is what a layerwise store persists — the RETRIEVE
        ``end`` is only the LMCache hit prefix bound and must never be
        reused as the store bound (T56-02). Requests that appear only as
        STORE entries (cold/no-hit) still get a view with an empty
        retrieve part, so they open an admission window and are stored
        (T56-03).

        Args:
            metadata: Stock connector metadata with one entry per
                scheduled request.

        Returns:
            request id -> :class:`StepRequestView` covering every RETRIEVE
            or STORE entry of the step.

        Raises:
            RuntimeError: When the metadata or an op lacks the fields
                this view needs, or an entry carries an unknown
                direction — every message names the missing field and
                its supplier (the vllm-ascend admission path) so the gap
                is actionable rather than a placeholder.
        """
        requests = getattr(metadata, "requests", None)
        if requests is None:
            raise RuntimeError(
                "dsv41.layerwise start_load_kv needs connector metadata "
                "with one requests entry per scheduled request "
                "(direction/request_id/op): the vllm-ascend admission "
                "path must attach them before scheduling a layerwise step"
            )
        # First Party (path-loaded sibling module)
        from lmcache_ascend.integration.vllm.dsv41_mp_adapter import (
            StepRequestView,
        )

        views: dict[str, StepRequestView] = {}
        stores: dict[str, tuple[int, int, tuple[tuple[int, ...], ...]]] = {}
        for meta in requests:
            direction = getattr(meta, "direction", None)
            request_id = getattr(meta, "request_id", None)
            op = getattr(meta, "op", None)
            if op is None:
                raise RuntimeError(
                    f"connector metadata entry for {request_id} has no "
                    "op: the vllm-ascend admission path must attach the "
                    "per-request LoadStoreOp (start/end/"
                    "skip_first_n_tokens/block_ids)"
                )
            configs = getattr(meta, "request_configs", None) or {}
            prefix_key = configs.get("dsv41.prefix_key")
            if direction == "RETRIEVE":
                views[request_id] = StepRequestView(
                    committed_tokens=int(getattr(op, "end", 0)),
                    blocks=_per_group_blocks(op, request_id),
                    start_token=int(getattr(op, "start", 0)),
                    skip_first_n_tokens=int(
                        getattr(op, "skip_first_n_tokens", 0)
                    ),
                    has_retrieve=True,
                    prefix_key=prefix_key,
                    hit_identity=configs.get("dsv41.hit_identity"),
                )
            elif direction == "STORE":
                stores[request_id] = (
                    int(getattr(op, "start", 0)),
                    int(getattr(op, "end", 0)),
                    _per_group_blocks(op, request_id),
                    prefix_key,
                )
            else:
                raise RuntimeError(
                    f"connector metadata for {request_id} has unknown "
                    f"direction {direction!r}: expected RETRIEVE or "
                    "STORE from the vllm-ascend admission path"
                )
        for request_id, (start, end, blocks, prefix_key) in stores.items():
            base = views.get(request_id) or StepRequestView(
                committed_tokens=0
            )
            views[request_id] = replace(
                base,
                has_store=True,
                store_start_token=start,
                store_end_token=end,
                store_blocks=blocks,
                prefix_key=base.prefix_key or prefix_key,
            )
        return views
