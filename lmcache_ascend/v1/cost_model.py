# SPDX-License-Identifier: Apache-2.0
"""Cost-based decision between fetching KV cache from a (potentially slow)
storage backend and recomputing it on the NPU.

Background
----------
When the storage backend is slow (e.g. remote DDR over a constrained link or
local SSD), the wall-clock time to *pull* ``N`` cached tokens can exceed the
time the NPU would take to simply *recompute* those same tokens via a normal
prefill. In that situation fetching is strictly counter-productive: the request
pays for both a long stall (the NPU sits idle waiting on the load stream) and
gains nothing over a plain ``PrefillNoCache`` path.

This module implements a light-weight, runtime-sampled cost model that answers
one question per request:

    *Given ``N`` hit tokens, is fetching cheaper than recomputing?*

Both cost estimates are built from *measured* quantities (exponentially-smoothed
EMAs of the observed fetch bandwidth and the observed recompute throughput),
so no manual tuning of model FLOPs or backend bandwidth is required. The model
is conservative during cold start (it always favors fetching until enough
samples have been collected), so enabling it is never worse than the current
behaviour.

The model is intentionally coarse-grained (per-request, all-or-nothing). A
finer-grained per-chunk fetch/recompute split (a la KVPR) is left to future
work.
"""
from __future__ import annotations

import threading
from typing import Optional

from lmcache.logging import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Process-wide registry connecting the vLLM connector (which owns the cost
# model) to the NPU connector (which feeds it fetch samples).
#
# These two objects live in different layers: the adapter is created by vLLM's
# connector factory, while the NPU connector is created by LMCache's engine.
# They don't hold references to each other, so a small registry is the least
# invasive bridge. Registration is keyed by the NPU device id (the same process
# always drives exactly one device), so TP > 1 (one process per rank) still
# works correctly.
# ---------------------------------------------------------------------------
_registry_lock = threading.Lock()
_registry: dict[int, "FetchRecomputeCostModel"] = {}


def register_cost_model(device_id: int, model: "FetchRecomputeCostModel") -> None:
    """Register the cost model for ``device_id`` (called by the adapter)."""
    with _registry_lock:
        _registry[int(device_id)] = model


def get_cost_model(device_id: Optional[int] = None) -> Optional["FetchRecomputeCostModel"]:
    """Look up the cost model for ``device_id``.

    If ``device_id`` is None, returns any registered model (single-device
    convenience) or None when none is registered.
    """
    with _registry_lock:
        if device_id is not None:
            return _registry.get(int(device_id))
        for m in _registry.values():
            return m
    return None


class FetchRecomputeCostModel:
    """EMA-based cost model comparing fetch vs. recompute for cached tokens.

    Cost model
    ----------
    For ``N`` hit tokens with ``B`` bytes/token of KV cache::

        T_fetch(N)     = N * B / bw_eff           # bandwidth-bound
        T_recompute(N) = N / recompute_rate       # compute-bound (tokens/sec)

    Decision: fetch iff ``T_fetch(N) <= T_recompute(N)``.

    Both ``bw_eff`` and ``recompute_rate`` are maintained as exponential moving
    averages, updated by :meth:`observe_fetch` and :meth:`observe_recompute`
    respectively. Callers (the NPU connector and the prefill path) feed in
    ``(num_tokens, elapsed_seconds)`` samples as they become available.

    Cold start
    ----------
    Until at least ``min_samples`` samples have been collected for *both*
    signals, :meth:`should_fetch` returns ``True`` (keep current fetch
    behaviour). This guarantees zero regression when the feature is enabled on
    a fresh process.
    """

    def __init__(
        self,
        enabled: bool = False,
        ema_alpha: float = 0.2,
        min_samples: int = 5,
        log_decisions: bool = True,
    ):
        self.enabled = enabled
        # EMA smoothing factor: higher => faster adaptation, noisier estimate.
        self.ema_alpha = float(ema_alpha)
        # Minimum samples before the model is trusted to override fetch.
        self.min_samples = int(min_samples)
        self.log_decisions = bool(log_decisions)

        # Guard: observe_* may be called from the load stream thread while
        # should_fetch is called from the scheduler thread.
        self._lock = threading.Lock()

        # bw_eff: effective bytes/sec of the full fetch path
        #         (backend read + H2D + scatter + stream sync).
        self._bw_eff_ema: Optional[float] = None
        self._bw_samples = 0

        # recompute_rate: tokens/sec of a no-cache prefill on the NPU.
        self._recompute_rate_ema: Optional[float] = None
        self._recompute_samples = 0

        # Bytes of KV cache per token. Lazily set by the connector once it
        # knows the layer/head dimensions. Guarded by None == not yet set,
        # which forces should_fetch to fall back to True.
        self._bytes_per_token: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_bytes_per_token(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype_size: int,
    ) -> None:
        """Set the per-token KV size (K + V across all layers).

        ``dtype_size`` is the element width in bytes (e.g. 2 for bf16). For
        compressed formats (CacheGen / DSA) callers should pass the effective
        on-the-wire width; the model only needs a consistent ratio.
        """
        if num_layers <= 0 or num_kv_heads <= 0 or head_size <= 0:
            return
        bpt = 2 * num_layers * num_kv_heads * head_size * max(int(dtype_size), 1)
        with self._lock:
            self._bytes_per_token = bpt
        logger.debug(
            "CostModel bytes_per_token set: %d B (layers=%d kv_heads=%d "
            "head=%d dtype=%d)",
            bpt, num_layers, num_kv_heads, head_size, dtype_size,
        )

    # ------------------------------------------------------------------ #
    # Sampling (called by the fetch / prefill paths)
    # ------------------------------------------------------------------ #
    def observe_fetch(self, num_tokens: int, elapsed_sec: float) -> None:
        """Record one fetch observation.

        ``elapsed_sec`` should cover the *full* fetch latency the request pays
        (backend read through to load-stream completion), so the derived
        bandwidth reflects what the scheduler actually waits on.
        """
        if num_tokens <= 0 or elapsed_sec <= 0:
            return
        with self._lock:
            if self._bytes_per_token is None:
                # Nothing useful to learn until the size is known.
                return
            bw = num_tokens * self._bytes_per_token / elapsed_sec
            if self._bw_eff_ema is None:
                self._bw_eff_ema = bw
            else:
                self._bw_eff_ema = (
                    self.ema_alpha * bw
                    + (1.0 - self.ema_alpha) * self._bw_eff_ema
                )
            self._bw_samples += 1

    def observe_recompute(self, num_tokens: int, elapsed_sec: float) -> None:
        """Record one no-cache prefill observation."""
        if num_tokens <= 0 or elapsed_sec <= 0:
            return
        with self._lock:
            rate = num_tokens / elapsed_sec
            if self._recompute_rate_ema is None:
                self._recompute_rate_ema = rate
            else:
                self._recompute_rate_ema = (
                    self.ema_alpha * rate
                    + (1.0 - self.ema_alpha) * self._recompute_rate_ema
                )
            self._recompute_samples += 1

    # ------------------------------------------------------------------ #
    # Decision (called by get_num_new_matched_tokens)
    # ------------------------------------------------------------------ #
    def should_fetch(self, num_hit_tokens: int) -> bool:
        """Return ``True`` if fetching ``num_hit_tokens`` is worthwhile.

        Conservative fallbacks (return ``True``, i.e. keep fetching):
          * feature disabled
          * not enough samples (cold start)
          * per-token size unknown
          * non-positive hit count
        """
        if not self.enabled:
            return True
        if num_hit_tokens <= 0:
            return True

        with self._lock:
            bpt = self._bytes_per_token
            bw = self._bw_eff_ema
            bw_n = self._bw_samples
            rate = self._recompute_rate_ema
            rate_n = self._recompute_samples

        if bpt is None:
            return True
        if bw_n < self.min_samples or rate_n < self.min_samples:
            return True
        if bw is None or rate is None or bw <= 0 or rate <= 0:
            return True

        T_fetch = num_hit_tokens * bpt / bw
        T_recompute = num_hit_tokens / rate
        do_fetch = T_fetch <= T_recompute

        if self.log_decisions:
            logger.info(
                "CostGate: hit_tokens=%d T_fetch=%.4fs T_recompute=%.4fs "
                "-> %s (bw_eff_ema=%.3f GB/s @%d samples, "
                "recompute_rate_ema=%.1f tok/s @%d samples, %d B/tok)",
                num_hit_tokens, T_fetch, T_recompute,
                "FETCH" if do_fetch else "RECOMPUTE",
                bw / 1e9, bw_n, rate, rate_n, bpt,
            )
        return do_fetch

    # Convenience for introspection / tests.
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "bw_eff_ema": self._bw_eff_ema,
                "bw_samples": self._bw_samples,
                "recompute_rate_ema": self._recompute_rate_ema,
                "recompute_samples": self._recompute_samples,
                "bytes_per_token": self._bytes_per_token,
                "ready": (
                    self.enabled
                    and self._bytes_per_token is not None
                    and self._bw_samples >= self.min_samples
                    and self._recompute_samples >= self.min_samples
                ),
            }
