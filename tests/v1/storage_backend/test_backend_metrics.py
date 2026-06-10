# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import patch

# First Party
from lmcache_ascend.v1.storage_backend.backend_metrics import (
    AscendBackendMetrics,
    lookup_result,
)


class _FakeMetric:
    def __init__(self):
        self.labels_seen = []
        self.increments = []
        self.observations = []

    def labels(self, **labels):
        self.labels_seen.append(labels)
        return self

    def inc(self, value=1):
        self.increments.append(value)

    def observe(self, value):
        self.observations.append(value)


class _FakePrometheusLogger:
    def __init__(self):
        self.labels = {"model_name": "model", "worker_id": 1, "role": "worker"}
        self.created = {}
        self.created_labelnames = {}

    def _create_counter(self, name, documentation, labelnames):
        del documentation
        metric = _FakeMetric()
        self.created[name] = metric
        self.created_labelnames[name] = labelnames
        return metric

    def _create_histogram(self, name, documentation, labelnames, buckets):
        del documentation, buckets
        metric = _FakeMetric()
        self.created[name] = metric
        self.created_labelnames[name] = labelnames
        return metric


def test_lookup_result():
    assert lookup_result(0, 4) == "miss"
    assert lookup_result(2, 4) == "partial"
    assert lookup_result(4, 4) == "hit"


def test_records_operations_and_chunks_with_low_cardinality_labels():
    logger = _FakePrometheusLogger()
    metrics = AscendBackendMetrics("p2p", "client", "pull", "npu")

    with patch.object(
        AscendBackendMetrics,
        "_get_logger",
        return_value=logger,
    ):
        logger.ascend_backend_operations = _FakeMetric()
        logger.ascend_backend_chunks = _FakeMetric()
        logger.ascend_backend_operation_duration_seconds = _FakeMetric()

        metrics.record_operation("transfer", "success", 0.25)
        metrics.record_chunks("transfer", "success", 3)

    labels = logger.ascend_backend_operations.labels_seen[0]
    assert labels["backend"] == "p2p"
    assert labels["backend_role"] == "client"
    assert labels["mode"] == "pull"
    assert labels["device"] == "npu"
    assert labels["operation"] == "transfer"
    assert labels["result"] == "success"
    assert logger.ascend_backend_operations.increments == [1]
    assert logger.ascend_backend_chunks.increments == [3]
    assert logger.ascend_backend_operation_duration_seconds.observations == [0.25]


def test_lazily_registers_metrics_on_existing_prometheus_logger():
    logger = _FakePrometheusLogger()

    with patch(
        "lmcache_ascend.v1.storage_backend.backend_metrics."
        "PrometheusLogger.GetInstanceOrNone",
        return_value=logger,
    ):
        assert AscendBackendMetrics._get_logger() is logger
        assert AscendBackendMetrics._get_logger() is logger

    assert set(logger.created) == {
        "lmcache:ascend_backend_operations",
        "lmcache:ascend_backend_chunks",
        "lmcache:ascend_backend_bytes",
        "lmcache:ascend_backend_operation_duration_seconds",
    }
    assert logger.created_labelnames["lmcache:ascend_backend_operations"] == [
        "model_name",
        "worker_id",
        "role",
        "backend",
        "backend_role",
        "mode",
        "device",
        "operation",
        "result",
    ]


def test_records_device_transfer_duration_and_bytes():
    logger = _FakePrometheusLogger()
    metrics = AscendBackendMetrics("pd", "sender", "push", "npu")

    with patch.object(
        AscendBackendMetrics,
        "_get_logger",
        return_value=logger,
    ):
        logger.ascend_backend_operations = _FakeMetric()
        logger.ascend_backend_chunks = _FakeMetric()
        logger.ascend_backend_bytes = _FakeMetric()
        logger.ascend_backend_operation_duration_seconds = _FakeMetric()

        metrics.transfer_completion_callback(0.125, 4096)

    labels = logger.ascend_backend_operations.labels_seen[0]
    assert labels["operation"] == "transfer_device"
    assert labels["result"] == "success"
    assert logger.ascend_backend_operation_duration_seconds.observations == [0.125]
    assert logger.ascend_backend_bytes.increments == [4096]


def test_device_transfer_callback_only_enabled_with_prometheus_logger():
    metrics = AscendBackendMetrics("pd", "sender", "push", "npu")

    with patch.object(AscendBackendMetrics, "_get_logger", return_value=None):
        assert metrics.get_transfer_completion_callback() is None

    with patch.object(
        AscendBackendMetrics,
        "_get_logger",
        return_value=_FakePrometheusLogger(),
    ):
        assert metrics.get_transfer_completion_callback() is not None


def test_metrics_are_noop_without_prometheus_logger():
    metrics = AscendBackendMetrics("pd", "receiver", "push", "npu")
    with patch.object(AscendBackendMetrics, "_get_logger", return_value=None):
        metrics.record_operation("lookup", "hit", 0.1)
        metrics.record_chunks("lookup", "hit", 2)


def test_metrics_fail_open_after_instrumentation_error():
    metrics = AscendBackendMetrics("pd", "receiver", "push", "npu")
    try:
        with patch.object(
            AscendBackendMetrics,
            "_get_logger",
            side_effect=RuntimeError("metrics unavailable"),
        ):
            metrics.record_operation("lookup", "hit", 0.1)
            metrics.record_chunks("lookup", "hit", 2)
        assert AscendBackendMetrics._disabled is True
    finally:
        AscendBackendMetrics._disabled = False
