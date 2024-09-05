from llmdatalens.evaluators.structured_output_evaluator import StructuredOutputEvaluator
from llmdatalens.core.base_model import LLMOutputData, GroundTruthData
from llmdatalens.core.metrics_registry import register_metric, MetricNames
from llmdatalens.core.enums import MetricField
import random
import time
from datetime import datetime

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

def print_experiment_history(evaluator):
    experiment_id = evaluator.experiment_id
    experiment = evaluator.experiment_manager.get_experiment(experiment_id)
    
    print(f"Experiment: {experiment.name}")
    print(f"Description: {experiment.description}")
    print(f"Created at: {experiment.created_at}")
    print(f"Number of runs: {len(experiment.runs)}")
    
    print("\nPrompt History:")
    for prompt_hash, prompt in experiment.prompts.items():
        print(f"Version: {prompt.version}")
        print(f"Created at: {prompt.created_at}")
        print(f"Modified at: {prompt.modified_at}")
        print(f"Text: {prompt.text}")
        print()
    
    print("Model History:")
    for model_name, model in experiment.models.items():
        print(f"Model: {model_name}")
        for version, version_info in model.versions.items():
            print(f"  Version: {version}")
            print(f"  First used: {version_info.first_used}")
            print(f"  Last used: {version_info.last_used}")
            print(f"  Run count: {version_info.run_count}")
        print()

    print("=" * 50 + "\n")

# Example 1: Using default metrics with experiment tracking
def example_default_metrics():
    print("Example 1: Using Default Metrics with Experiment Tracking")
    evaluator = StructuredOutputEvaluator(experiment_name="Default Metrics Experiment")
    llm_outputs, ground_truths = generate_sample_data(num_samples=10)

    for i, (llm_output, ground_truth) in enumerate(zip(llm_outputs, ground_truths)):
        llm_output.metadata["model_info"] = {"name": "GPT-3.5", "version": "1.0"}
        llm_output.metadata["prompt_info"] = {"text": f"Extract invoice details from text (Run {i+1}):"}
        evaluator.add_llm_output(llm_output, latency=random.uniform(0.1, 0.5), confidence=random.uniform(0.8, 1.0))
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()
    print_results(result)
    print_experiment_history(evaluator)

# Example 2: Cherry-picking specific metrics and updating prompts
def example_specific_metrics():
    print("Example 2: Cherry-picking Specific Metrics and Updating Prompts")
    evaluator = StructuredOutputEvaluator(
        metrics=[MetricNames.FieldSpecificAccuracy, MetricNames.AverageLatency, MetricNames.ConfidenceScore],
        experiment_name="Specific Metrics Experiment"
    )
    llm_outputs, ground_truths = generate_sample_data(num_samples=10)

    prompts = [
        "Extract invoice details from the following text:",
        "Parse the invoice information from this text:",
        "Identify the invoice number, total amount, and date from the given text:"
    ]

    for i, (llm_output, ground_truth) in enumerate(zip(llm_outputs, ground_truths)):
        prompt = prompts[i % len(prompts)]
        llm_output.metadata["model_info"] = {"name": "GPT-4", "version": "1.0"}
        llm_output.metadata["prompt_info"] = {"text": prompt}
        evaluator.add_llm_output(llm_output, latency=random.uniform(0.1, 0.5), confidence=random.uniform(0.8, 1.0))
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()
    print_results(result)
    print_experiment_history(evaluator)

# Example 3: Using a custom metric and multiple model versions
def example_custom_metric():
    print("Example 3: Using a Custom Metric and Multiple Model Versions")

    @register_metric("CustomPrecision", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
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

    evaluator = StructuredOutputEvaluator(
        metrics=[MetricNames.OverallAccuracy, MetricNames.CustomPrecision],
        experiment_name="Custom Metric and Multiple Models Experiment"
    )
    llm_outputs, ground_truths = generate_sample_data(num_samples=15)

    models = [
        {"name": "GPT-3.5", "version": "1.0"},
        {"name": "GPT-3.5", "version": "1.1"},
        {"name": "GPT-4", "version": "1.0"}
    ]

    for i, (llm_output, ground_truth) in enumerate(zip(llm_outputs, ground_truths)):
        model = models[i % len(models)]
        llm_output.metadata["model_info"] = model
        llm_output.metadata["prompt_info"] = {"text": f"Extract invoice details (Model: {model['name']} v{model['version']})"}
        evaluator.add_llm_output(llm_output, latency=random.uniform(0.1, 0.5), confidence=random.uniform(0.8, 1.0))
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()
    print_results(result)
    print_experiment_history(evaluator)

if __name__ == "__main__":
    example_default_metrics()
    example_specific_metrics()
    example_custom_metric()
