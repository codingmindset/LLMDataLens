import pytest
from llmdatalens.core.metrics_registry import metrics_registry, register_metric
from llmdatalens.core.enums import MetricField

def test_metric_registration():
    @register_metric("test_metric", field=MetricField.Accuracy, input_keys=["y_true", "y_pred"])
    def test_metric(y_true, y_pred):
        return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    assert "test_metric" in metrics_registry.get_all()
    metric_info = metrics_registry.get("test_metric")
    assert metric_info is not None, "Metric 'test_metric' not found in registry"
    assert metric_info.field == MetricField.Accuracy
    assert metric_info.input_keys == ["y_true", "y_pred"]
    assert metric_info.func([1, 0, 1], [1, 1, 1]) == 2/3

def test_metric_retrieval():
    @register_metric("retrieval_test", field=MetricField.Performance, input_keys=["latencies"])
    def retrieval_test(latencies):
        return sum(latencies) / len(latencies)

    metric_info = metrics_registry.get("retrieval_test")
    assert metric_info is not None
    assert metric_info.field == MetricField.Performance
    assert metric_info.input_keys == ["latencies"]
    assert metric_info.func([1, 2, 3]) == 2

def test_nonexistent_metric_retrieval():
    assert metrics_registry.get("nonexistent_metric") is None

def test_get_all_metrics():
    all_metrics = metrics_registry.get_all()
    assert isinstance(all_metrics, dict)
    assert len(all_metrics) > 0
    assert "test_metric" in all_metrics
    assert "retrieval_test" in all_metrics
