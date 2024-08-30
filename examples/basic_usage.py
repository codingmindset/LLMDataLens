from llmdatalens.evaluators.structured_output_evaluator import StructuredOutputEvaluator
from llmdatalens.core.base_model import LLMOutputData, GroundTruthData
from llmdatalens.core.metrics_registry import register_metric
from llmdatalens.core.enums import MetricField
import random
import time

def generate_sample_data(num_samples: int = 100):
    llm_outputs = []
    ground_truths = []
    for i in range(num_samples):
        llm_output = LLMOutputData(
            raw_output=f"Invoice {i} processed",
            structured_output={
                "invoice_number": f"INV-{i:04d}",
                "total_amount": round(random.uniform(100, 1000), 2),
                "date": f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            }
        )
        ground_truth = GroundTruthData(
            data={
                "invoice_number": f"INV-{i:04d}",
                "total_amount": round(random.uniform(100, 1000), 2),
                "date": f"2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
            }
        )
        llm_outputs.append(llm_output)
        ground_truths.append(ground_truth)
    return llm_outputs, ground_truths

def print_results(result):
    print("Evaluation Results:")
    for metric_name, metric_value in result.metrics.items():
        if isinstance(metric_value, dict):
            print(f"{metric_name}:")
            for field, value in metric_value.items():
                print(f"  {field}: {value}")
        else:
            print(f"{metric_name}: {metric_value}")
    print("\nEvaluation Details:")
    for detail_name, detail_value in result.details.items():
        print(f"{detail_name}: {detail_value}")
    print("\n" + "="*50 + "\n")

# Example 1: Using default metrics
def example_default_metrics():
    print("Example 1: Using Default Metrics")
    evaluator = StructuredOutputEvaluator(metrics=["overall_accuracy", "average_latency"])
    llm_outputs, ground_truths = generate_sample_data()

    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        evaluator.add_llm_output(llm_output, latency=random.uniform(0.1, 0.5), confidence=random.uniform(0.8, 1.0))
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()
    print_results(result)

# Example 2: Cherry-picking specific metrics
def example_specific_metrics():
    print("Example 2: Cherry-picking Specific Metrics")
    evaluator = StructuredOutputEvaluator(metrics=["overall_accuracy", "average_latency"])
    llm_outputs, ground_truths = generate_sample_data()

    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        evaluator.add_llm_output(llm_output, latency=random.uniform(0.1, 0.5), confidence=random.uniform(0.8, 1.0))
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()
    print_results(result)

# Example 3: Using a custom metric
def example_custom_metric():
    print("Example 3: Using a Custom Metric")

    @register_metric("custom_precision", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
    def calculate_precision(ground_truths, predictions):
        """Calculate the precision of predictions."""
        true_positives = 0
        predicted_positives = 0
        for gt, pred in zip(ground_truths, predictions):
            for gt_val, pred_val in zip(gt.values(), pred.values()):
                if gt_val == pred_val == 1:
                    true_positives += 1
                if pred_val == 1:
                    predicted_positives += 1
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    evaluator = StructuredOutputEvaluator(metrics=["overall_accuracy", "custom_precision"])
    llm_outputs, ground_truths = generate_sample_data()

    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        evaluator.add_llm_output(llm_output, latency=random.uniform(0.1, 0.5), confidence=random.uniform(0.8, 1.0))
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()
    print_results(result)

if __name__ == "__main__":
    example_default_metrics()
    example_specific_metrics()
    example_custom_metric()
