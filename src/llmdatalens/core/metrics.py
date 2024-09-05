from typing import List, Any, Dict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from llmdatalens.core.metrics_registry import register_metric
from llmdatalens.core.enums import MetricField

def calculate_overall_accuracy(ground_truths: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> float:
    """Calculate the overall accuracy across all fields."""
    correct = 0
    total = 0
    
    for gt, pred in zip(ground_truths, predictions):
        correct += compare_nested_structures(gt, pred)
        total += count_elements(gt)
    
    return correct / total if total > 0 else 0

def compare_nested_structures(gt: Any, pred: Any) -> int:
    """Recursively compare nested structures and return the number of correct elements."""
    if isinstance(gt, dict) and isinstance(pred, dict):
        return sum(compare_nested_structures(gt.get(k), pred.get(k)) for k in gt)
    elif isinstance(gt, list) and isinstance(pred, list):
        return sum(compare_nested_structures(g, p) for g, p in zip(gt, pred))
    else:
        return 1 if gt == pred else 0

def count_elements(structure: Any) -> int:
    """Recursively count the number of elements in a nested structure."""
    if isinstance(structure, dict):
        return sum(count_elements(v) for v in structure.values())
    elif isinstance(structure, list):
        return sum(count_elements(item) for item in structure)
    else:
        return 1

@register_metric("OverallAccuracy", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
def calculate_overall_accuracy_wrapper(ground_truths: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> float:
    return calculate_overall_accuracy(ground_truths, predictions)

@register_metric("FieldSpecificAccuracy", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
def calculate_field_specific_accuracy(ground_truths: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate accuracy for each field in structured data."""
    field_accuracies = {}
    for field in ground_truths[0].keys():
        field_true = [gt[field] for gt in ground_truths]
        field_pred = [pred[field] for pred in predictions]
        field_accuracies[field] = accuracy_score(field_true, field_pred)
    return field_accuracies

@register_metric("AverageLatency", field=MetricField.Performance, input_keys=["latencies"])
def calculate_average_latency(latencies: List[float]) -> float:
    """Calculate the average latency of predictions."""
    return float(np.mean(latencies))

@register_metric("Throughput", field=MetricField.Performance, input_keys=["total_items", "total_time"])
def calculate_throughput(total_items: int, total_time: float) -> float:
    """Calculate the throughput of the system."""
    return total_items / total_time if total_time > 0 else 0

@register_metric("ErrorRate", field=MetricField.Accuracy, input_keys=["y_true", "y_pred"])
def calculate_error_rate(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculate the error rate of predictions."""
    return 1 - calculate_overall_accuracy(y_true, y_pred)

@register_metric("ConfidenceScore", field=MetricField.Confidence, input_keys=["confidences"])
def calculate_confidence_score(confidences: List[float]) -> float:
    """Calculate the average confidence score of predictions."""
    return float(np.mean(confidences))

@register_metric("F1Score", field=MetricField.Accuracy, input_keys=["y_true", "y_pred"])
def calculate_f1_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculate the F1 score of predictions."""
    return f1_score(y_true, y_pred, average='weighted')

@register_metric("RobustnessScore", field=MetricField.Robustness, input_keys=["normal_accuracy", "challenging_accuracy"])
def calculate_robustness_score(normal_accuracy: float, challenging_accuracy: float) -> float:
    """Calculate the robustness score based on performance on normal vs challenging inputs."""
    return challenging_accuracy / normal_accuracy if normal_accuracy > 0 else 0

@register_metric("ConsistencyScore", field=MetricField.Consistency, input_keys=["accuracies"])
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
