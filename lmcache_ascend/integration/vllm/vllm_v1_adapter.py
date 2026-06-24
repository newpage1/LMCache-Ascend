# SPDX-License-Identifier: Apache-2.0
# Standard
import time
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.request import RequestStatus
import torch

# First Party
from lmcache_ascend.v1 import cost_model
from lmcache_ascend.v1.cost_model import FetchRecomputeCostModel
from lmcache_ascend.v1.kv_format import KVCacheFormat

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent)
        self.store_async = self.config.store_async
        self._wait_for_save_done = True
        self._finished_req_ids_waiting_for_save: set[str] = set()
        self._late_finished_sending: set[str] = set()
        logger.debug("store_async: %s", self.store_async)

        # ---- Fetch-vs-Recompute cost gate -------------------------------
        # A coarse-grained (per-request, all-or-nothing) decision: when the
        # storage backend is slow, pulling N cached tokens may be slower than
        # recomputing them. See cost_model.py for the full rationale.
        self._cost_model = FetchRecomputeCostModel(
            enabled=bool(getattr(self.config, "fetch_recompute_gate_enable", False)),
            ema_alpha=float(getattr(self.config, "cost_ema_alpha", 0.2)),
            min_samples=int(getattr(self.config, "cost_min_samples", 5)),
            log_decisions=bool(getattr(self.config, "cost_gate_log_enable", True)),
        )
        # Configure per-token KV size from model geometry. This is the K+V
        # footprint of one token across all layers, in bytes.
        try:
            model_config = vllm_config.model_config
            num_layers = model_config.get_num_layers(vllm_config.parallel_config)
            num_kv_heads = model_config.get_num_kv_heads(
                vllm_config.parallel_config
            )
            head_size = model_config.get_head_size()
            # Element width from the model dtype (e.g. bfloat16 -> 2 bytes).
            dtype_size = max(
                torch.tensor([], dtype=model_config.dtype).element_size(), 1
            )
            self._cost_model.set_bytes_per_token(
                num_layers, num_kv_heads, head_size, dtype_size
            )
        except Exception:  # pragma: no cover - best-effort init
            logger.exception("CostGate: failed to initialize bytes_per_token; "
                             "gate will stay conservative.")
        # State for step-level prefill (recompute) sampling. We measure the
        # wall-clock span of a prefill step at the next step's start_load_kv,
        # so we only need a (prev_step_time, prev_step_prefill_tokens) tuple.
        self._prefill_sample_time: Optional[float] = None
        self._prefill_sample_tokens: int = 0
        if self._cost_model.enabled:
            logger.info(
                "CostGate enabled (ema_alpha=%.2f, min_samples=%d).",
                self._cost_model.ema_alpha, self._cost_model.min_samples,
            )
            # Publish the cost model so the NPU connector can feed it fetch
            # bandwidth samples. Keyed by the NPU device id this process owns.
            try:
                device_id = int(torch.npu.current_device())
            except Exception:  # pragma: no cover - non-NPU test envs
                device_id = 0
            cost_model.register_cost_model(device_id, self._cost_model)

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.current_layer = 0
        self._wait_for_save_done = False

        # ---- Step-level recompute (prefill) sampling --------------------
        # Measure the wall-clock span of the *previous* step now that the
        # current step is starting. If the previous step was a no-cache
        # prefill (PrefillNoCache), the elapsed time approximates the NPU's
        # recompute throughput for that many tokens.
        now = time.perf_counter()
        if (
            self._cost_model.enabled
            and self._prefill_sample_time is not None
            and self._prefill_sample_tokens > 0
        ):
            elapsed = now - self._prefill_sample_time
            self._cost_model.observe_recompute(
                self._prefill_sample_tokens, elapsed
            )
        self._prefill_sample_time = now
        # Probe the attn state for the upcoming step to decide whether this
        # step qualifies as a recompute sample.
        self._prefill_sample_tokens = self._probe_recompute_tokens(
            forward_context
        )

        super().start_load_kv(forward_context, **kwargs)

    def _probe_recompute_tokens(
        self, forward_context: "ForwardContext"
    ) -> int:
        """Return the number of tokens about to be prefilled *without* a KV
        cache hit, i.e. genuinely recomputed on the NPU this step.

        Returns 0 if the signal is unavailable (e.g. attn_metadata missing,
        decode step, or a cache-hit prefill). We deliberately keep this
        best-effort: a missed sample just delays EMA convergence.
        """
        if not self._cost_model.enabled:
            return 0
        try:
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                return 0
            # forward_context.attn_metadata is a PerLayerAttnMetadata dict
            # (layer_name -> AscendMetadata). Any single entry carries the
            # step-level attn_state and num_actual_tokens; pick the first.
            meta = None
            if isinstance(attn_metadata, dict):
                if not attn_metadata:
                    return 0
                meta = next(iter(attn_metadata.values()))
            else:
                meta = attn_metadata
            if meta is None:
                return 0
            # PrefillNoCache == 0 in AscendAttentionState. Accept both the
            # enum and the raw int to stay robust to upstream changes.
            state = getattr(meta, "attn_state", None)
            state_val = getattr(state, "value", state)
            if state_val != 0:
                return 0
            num_actual_tokens = int(getattr(meta, "num_actual_tokens", 0) or 0)
            return num_actual_tokens
        except Exception:  # pragma: no cover - best-effort sampling
            return 0

    @property
    def cost_model(self) -> FetchRecomputeCostModel:
        """Expose the cost model so the NPU connector can feed fetch samples."""
        return self._cost_model

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """Ascend override: insert the fetch-vs-recompute cost gate.

        Delegates to the upstream lookup to get the raw number of externally
        cached tokens, then — if the cost gate is enabled and warmed up —
        downgrades a fetch to a recompute (returns 0) when pulling those
        tokens from a slow backend would be slower than recomputing them.
        """
        num_hit = super().get_num_new_matched_tokens(request, num_computed_tokens)
        # None  -> async lookup not ready yet (keep waiting).
        # 0     -> upstream already decided not to fetch.
        if num_hit is None or num_hit == 0:
            return num_hit

        if not self._cost_model.enabled:
            return num_hit

        if not self._cost_model.should_fetch(num_hit):
            logger.info(
                "CostGate RECOMPUTE: req %s hit_tokens=%d, fetching would be "
                "slower than recomputing on NPU -> skip fetch.",
                request.request_id, num_hit,
            )
            return 0
        return num_hit

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._prepare_ascend_layerwise_connector()
        self._manager.post_init()

    def _prepare_ascend_layerwise_connector(self) -> None:
        if not getattr(self, "use_layerwise", False) or self.lmcache_engine is None:
            return
        connector = self.lmcache_engine.gpu_connector
        if connector is None:
            return

        kvcaches = list(self.kv_caches.values())
        if not kvcaches:
            return

        connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
        if hasattr(connector, "_lazy_initialize_buffer"):
            connector._lazy_initialize_buffer(connector.kvcaches)

        if getattr(connector, "kv_format", None) == KVCacheFormat.DSA_C8_KV:
            self.lmcache_engine.metadata.kv_dtype = torch.uint8

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            if self.lmcache_engine is not None:
                for request in connector_metadata.requests:
                    self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            return

        if self.use_layerwise:
            assert not self.store_async, (
                "Layerwise storing is not supported with async store"
            )
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            self._replay_finished_stores_after_save()
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        # lmcache-ascend start ---------------------
        ordering_event = torch.npu.Event()
        ordering_event.record()
        # lmcache-ascend end ---------------------

        for request in connector_metadata.requests:
            self.lmcache_engine.lookup_unpin(request.req_id)

            try:
                save_spec = request.save_spec
                if (
                    save_spec is None or not save_spec.can_save
                ) and self.kv_role != "kv_producer":
                    continue

                token_ids = request.token_ids

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # lmcache-ascend start ---------------------
                slot_mapping = slot_mapping.pin_memory()
                with torch.npu.stream(self.lmcache_engine.gpu_connector.store_stream):
                    slot_mapping_npu = slot_mapping.to(
                        device="npu", dtype=torch.long, non_blocking=True
                    )
                # lmcache-ascend end ---------------------

                skip_leading_tokens = save_spec.skip_leading_tokens

                if skip_leading_tokens == len(token_ids):
                    continue
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                is_last_prefill = request.is_last_prefill
                if is_last_prefill:
                    if request.disagg_spec:
                        request.disagg_spec.is_last_prefill = True
                else:
                    if not self.enable_blending:
                        token_len = len(token_ids)
                        aligned_token_len = (
                            token_len
                            // self._lmcache_chunk_size
                            * self._lmcache_chunk_size
                        )
                        token_ids = token_ids[:aligned_token_len]
                        store_mask = store_mask[:aligned_token_len]
                        slot_mapping = slot_mapping[:aligned_token_len]

                self.lmcache_engine.store(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    transfer_spec=request.disagg_spec,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    ordering_event=ordering_event,
                    slot_mapping_npu=slot_mapping_npu,
                )

                if get_pp_group().is_last_rank:
                    save_spec.skip_leading_tokens = len(token_ids)
                    if request.disagg_spec:
                        request.disagg_spec.num_transferred_tokens = len(token_ids)
            except Exception:
                # Do not let one failing request abort the save loop
                logger.exception(
                    "wait_for_save failed for request %s; skipping save",
                    request.req_id,
                )
                continue

        self._wait_for_save_done = True
        self._replay_finished_stores_after_save()

    def _may_register_store_after_wait_for_save(self, request: "Request") -> bool:
        if self.kv_role == "kv_consumer":
            return False
        save_spec = request.save_spec
        if save_spec is None:
            return False
        if not save_spec.can_save and self.kv_role != "kv_producer":
            return False
        return save_spec.skip_leading_tokens != len(request.token_ids)

    def _replay_finished_stores_after_save(self) -> None:
        if not self._finished_req_ids_waiting_for_save or self.lmcache_engine is None:
            return

        finished_sending = self.lmcache_engine.get_finished_stores(
            self._finished_req_ids_waiting_for_save
        )
        if finished_sending:
            self._late_finished_sending |= finished_sending
        self._finished_req_ids_waiting_for_save = set()

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if self.lmcache_engine is None:
            return None, None
        query_req_ids = set(finished_req_ids)
        if not self._wait_for_save_done:
            # NOTE (gingfung): The is a workaround logic for the case
            # where the requests is deferred (i.e. spec_decode or MTP)
            # and the model_runner call get_finished before wait_for_save.
            connector_metadata = self._parent._get_connector_metadata()
            assert isinstance(connector_metadata, LMCacheConnectorMetadata)

            waiting_for_save = {
                request.req_id
                for request in connector_metadata.requests
                if request.req_id in finished_req_ids
                and self._may_register_store_after_wait_for_save(request)
            }
            if waiting_for_save:
                self._finished_req_ids_waiting_for_save |= waiting_for_save
                query_req_ids -= waiting_for_save

        finished_sending = self.lmcache_engine.get_finished_stores(query_req_ids)
        if self._late_finished_sending:
            finished_sending |= self._late_finished_sending
            self._late_finished_sending = set()
        return (
            finished_sending if finished_sending else None,
            None,
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.lmcache_engine is None:
            return

        logger.debug(
            "LMCache-Ascend handling preemptions: req_ids=%s",
            sorted(preempted_req_ids),
        )

        # Lookup pins are request-scoped and normally released in wait_for_save().
        # A preempted request may leave that path before its metadata is replayed.
        for req_id in preempted_req_ids:
            self.lmcache_engine.lookup_unpin(req_id)

        if not self.store_async or self.kv_role == "kv_consumer":
            return

        waited_req_ids = self.lmcache_engine.wait_for_pending_stores(preempted_req_ids)
        if waited_req_ids:
            logger.info(
                "Handled preemptions after draining async stores: req_ids=%s",
                sorted(waited_req_ids),
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        _, return_params = super().request_finished(request, block_ids)

        if (
            request.status == RequestStatus.FINISHED_ABORTED
            and self.lmcache_engine is not None
        ):
            self.lmcache_engine.lookup_unpin(request.request_id)

            if self.store_async and self.kv_role != "kv_consumer":
                try:
                    self.lmcache_engine.wait_for_pending_stores({request.request_id})
                except Exception:
                    logger.warning(
                        "wait_for_pending_stores failed for aborted request %s",
                        request.request_id,
                        exc_info=True,
                    )

        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
