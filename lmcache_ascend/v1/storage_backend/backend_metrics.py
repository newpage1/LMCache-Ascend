# SPDX-License-Identifier: Apache-2.0
"""Low-cardinality Prometheus metrics for Ascend cross-node backends."""

# Standard
from typing import Optional
import logging
import threading

# Third Party
from lmcache.observability import PrometheusLogger


_METRIC_LABELS = [
    "backend",
    "backend_role",
    "mode",
    "device",
    "operation",
    "result",
]

_DURATION_BUCKETS = [
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
]

logger = logging.getLogger(__name__)


class AscendBackendMetrics:
    """Records P2P/PD backend metrics through LMCache's Prometheus registry."""

    _registration_lock = threading.RLock()
    _disabled = False

    def __init__(
        self,
        backend: str,
        backend_role: str,
        mode: str,
        device: str,
    ) -> None:
        self.backend = backend
        self.backend_role = backend_role
        self.mode = mode
        self.device = device

    @staticmethod
    def _get_logger() -> Optional[PrometheusLogger]:
        if AscendBackendMetrics._disabled:
            return None

        try:
            prometheus_logger = PrometheusLogger.GetInstanceOrNone()
            if prometheus_logger is None:
                return None

            with AscendBackendMetrics._registration_lock:
                if not hasattr(prometheus_logger, "ascend_backend_operations"):
                    labelnames = list(prometheus_logger.labels.keys()) + _METRIC_LABELS
                    prometheus_logger.ascend_backend_operations = (
                        prometheus_logger._create_counter(
                            name="lmcache:ascend_backend_operations",
                            documentation=(
                                "Ascend cross-node backend operation outcomes"
                            ),
                            labelnames=labelnames,
                        )
                    )
                    prometheus_logger.ascend_backend_chunks = (
                        prometheus_logger._create_counter(
                            name="lmcache:ascend_backend_chunks",
                            documentation=(
                                "Chunks processed by Ascend cross-node backends"
                            ),
                            labelnames=labelnames,
                        )
                    )
                    prometheus_logger.ascend_backend_bytes = (
                        prometheus_logger._create_counter(
                            name="lmcache:ascend_backend_bytes",
                            documentation=(
                                "Bytes processed by Ascend cross-node backends"
                            ),
                            labelnames=labelnames,
                        )
                    )
                    prometheus_logger.ascend_backend_operation_duration_seconds = (
                        prometheus_logger._create_histogram(
                            name="lmcache:ascend_backend_operation_duration_seconds",
                            documentation=(
                                "Ascend cross-node backend operation duration "
                                "in seconds"
                            ),
                            labelnames=labelnames,
                            buckets=_DURATION_BUCKETS,
                        )
                    )
            return prometheus_logger
        except Exception:
            AscendBackendMetrics._disable_after_error()
            return None

    @staticmethod
    def _disable_after_error() -> None:
        with AscendBackendMetrics._registration_lock:
            if not AscendBackendMetrics._disabled:
                logger.warning(
                    "Disabling Ascend backend metrics after an instrumentation error",
                    exc_info=True,
                )
                AscendBackendMetrics._disabled = True

    def _labels(self, logger: PrometheusLogger, operation: str, result: str):
        return {
            **logger.labels,
            "backend": self.backend,
            "backend_role": self.backend_role,
            "mode": self.mode,
            "device": self.device,
            "operation": operation,
            "result": result,
        }

    def record_operation(
        self,
        operation: str,
        result: str,
        duration_seconds: Optional[float] = None,
    ) -> None:
        try:
            prometheus_logger = self._get_logger()
            if prometheus_logger is None:
                return
            labels = self._labels(prometheus_logger, operation, result)
            prometheus_logger.ascend_backend_operations.labels(**labels).inc()
            if duration_seconds is not None:
                prometheus_logger.ascend_backend_operation_duration_seconds.labels(
                    **labels
                ).observe(duration_seconds)
        except Exception:
            self._disable_after_error()

    def record_chunks(self, operation: str, result: str, count: int) -> None:
        if count <= 0:
            return
        try:
            prometheus_logger = self._get_logger()
            if prometheus_logger is None:
                return
            labels = self._labels(prometheus_logger, operation, result)
            prometheus_logger.ascend_backend_chunks.labels(**labels).inc(count)
        except Exception:
            self._disable_after_error()

    def record_bytes(self, operation: str, result: str, count: int) -> None:
        if count <= 0:
            return
        try:
            prometheus_logger = self._get_logger()
            if prometheus_logger is None:
                return
            labels = self._labels(prometheus_logger, operation, result)
            prometheus_logger.ascend_backend_bytes.labels(**labels).inc(count)
        except Exception:
            self._disable_after_error()

    def transfer_completion_callback(
        self, duration_seconds: float, num_bytes: int
    ) -> None:
        self.record_operation("transfer_device", "success", duration_seconds)
        self.record_bytes("transfer", "success", num_bytes)

    def get_transfer_completion_callback(self):
        if self._get_logger() is None:
            return None
        return self.transfer_completion_callback


def lookup_result(num_hits: int, num_requested: int) -> str:
    if num_hits <= 0:
        return "miss"
    if num_hits >= num_requested:
        return "hit"
    return "partial"
