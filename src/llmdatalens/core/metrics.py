from typing import List, Any, Dict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from llmdatalens.core.metrics_registry import register_metric
from llmdatalens.core.enums import MetricField

@register_metric("overall_accuracy", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
def calculate_overall_accuracy(ground_truths: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> float:
    """Calculate the overall accuracy of predictions."""
    all_true = []
    all_pred = []
    for gt, pred in zip(ground_truths, predictions):
        all_true.extend(gt.values())
        all_pred.extend(pred.values())
    return accuracy_score(all_true, all_pred)

@register_metric("field_specific_accuracy", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
def calculate_field_specific_accuracy(ground_truths: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate accuracy for each field in structured data."""
    field_accuracies = {}
    for field in ground_truths[0].keys():
        field_true = [gt[field] for gt in ground_truths]
        field_pred = [pred[field] for pred in predictions]
        field_accuracies[field] = accuracy_score(field_true, field_pred)
    return field_accuracies

@register_metric("average_latency", field=MetricField.Performance, input_keys=["latencies"])
def calculate_average_latency(latencies: List[float]) -> float:
    """Calculate the average latency of predictions."""
    return float(np.mean(latencies))

@register_metric("throughput", field=MetricField.Performance, input_keys=["total_items", "total_time"])
def calculate_throughput(total_items: int, total_time: float) -> float:
    """Calculate the throughput of the system."""
    return total_items / total_time if total_time > 0 else 0

@register_metric("error_rate", field=MetricField.Accuracy, input_keys=["y_true", "y_pred"])
def calculate_error_rate(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculate the error rate of predictions."""
    return 1 - calculate_overall_accuracy(y_true, y_pred)

@register_metric("confidence_score", field=MetricField.Confidence, input_keys=["confidences"])
def calculate_confidence_score(confidences: List[float]) -> float:
    """Calculate the average confidence score of predictions."""
    return float(np.mean(confidences))

@register_metric("f1_score", field=MetricField.Accuracy, input_keys=["y_true", "y_pred"])
def calculate_f1_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculate the F1 score of predictions."""
    return f1_score(y_true, y_pred, average='weighted')

@register_metric("robustness_score", field=MetricField.Robustness, input_keys=["normal_accuracy", "challenging_accuracy"])
def calculate_robustness_score(normal_accuracy: float, challenging_accuracy: float) -> float:
    """Calculate the robustness score based on performance on normal vs challenging inputs."""
    return challenging_accuracy / normal_accuracy if normal_accuracy > 0 else 0

@register_metric("consistency_score", field=MetricField.Consistency, input_keys=["accuracies"])
def calculate_consistency_score(accuracies: List[float]) -> float:
    """Calculate the consistency score based on accuracies."""
    return float(1 - np.std(accuracies))

# Utility functions (not metrics, so not registered)
def start_timer() -> float:
    import time
    return time.time()

def end_timer(start_time: float) -> float:
    import time
    return time.time() - start_time
