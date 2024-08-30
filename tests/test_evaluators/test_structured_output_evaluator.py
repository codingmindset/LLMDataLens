import pytest
from llmdatalens.evaluators.structured_output_evaluator import StructuredOutputEvaluator
from llmdatalens.core.base_model import LLMOutputData, GroundTruthData
from llmdatalens.core.metrics_registry import register_metric
from llmdatalens.core.enums import MetricField

def generate_sample_data(num_samples: int = 10):
    llm_outputs = []
    ground_truths = []
    for i in range(num_samples):
        llm_output = LLMOutputData(
            raw_output=f"Invoice {i} processed",
            structured_output={
                "invoice_number": f"INV-{i:04d}",
                "total_amount": 100.0 if i % 2 == 0 else 200.0,
                "date": "2023-01-01"
            }
        )
        ground_truth = GroundTruthData(
            data={
                "invoice_number": f"INV-{i:04d}",
                "total_amount": 100.0,
                "date": "2023-01-01"
            }
        )
        llm_outputs.append(llm_output)
        ground_truths.append(ground_truth)
    return llm_outputs, ground_truths

def test_structured_output_evaluator_initialization():
    evaluator = StructuredOutputEvaluator()
    assert evaluator.metrics == []  # Assuming default metrics are empty
    assert evaluator.llm_outputs == []
    assert evaluator.ground_truths == []

def test_add_llm_output_and_ground_truth():
    evaluator = StructuredOutputEvaluator()
    llm_output = LLMOutputData(raw_output="Test", structured_output={"key": "value"})
    ground_truth = GroundTruthData(data={"key": "value"})

    evaluator.add_llm_output(llm_output, latency=0.1, confidence=0.9)
    evaluator.add_ground_truth(ground_truth)

    assert len(evaluator.llm_outputs) == 1
    assert len(evaluator.ground_truths) == 1
    assert evaluator.llm_outputs[0].metadata is not None
    assert evaluator.llm_outputs[0].metadata.get("latency") == 0.1
    assert evaluator.llm_outputs[0].metadata.get("confidence") == 0.9

def test_evaluate_with_default_metrics():
    evaluator = StructuredOutputEvaluator(metrics=["overall_accuracy", "average_latency"])
    llm_outputs, ground_truths = generate_sample_data()

    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        evaluator.add_llm_output(llm_output, latency=0.1, confidence=1.0)
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()

    assert "overall_accuracy" in result.metrics
    assert "average_latency" in result.metrics
    assert abs(result.metrics["overall_accuracy"] - 0.8333) < 0.0001  # Allow for small floating-point differences
    assert result.metrics["average_latency"] == 0.1

def test_evaluate_with_custom_metric():
    @register_metric("custom_metric", field=MetricField.Accuracy, input_keys=["ground_truths", "predictions"])
    def custom_metric(ground_truths, predictions):
        total_fields = 0
        correct_fields = 0
        for gt, pred in zip(ground_truths, predictions):
            for key in gt:
                total_fields += 1
                if gt[key] == pred[key]:
                    correct_fields += 1
        return correct_fields / total_fields if total_fields > 0 else 0

    evaluator = StructuredOutputEvaluator(metrics=["custom_metric"])
    llm_outputs, ground_truths = generate_sample_data()

    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        evaluator.add_llm_output(llm_output)
        evaluator.add_ground_truth(ground_truth)

    result = evaluator.evaluate()

    assert "custom_metric" in result.metrics
    assert abs(result.metrics["custom_metric"] - 0.8333) < 0.0001  # Allow for small floating-point differences

def test_evaluator_reset():
    evaluator = StructuredOutputEvaluator()
    llm_outputs, ground_truths = generate_sample_data(5)

    for llm_output, ground_truth in zip(llm_outputs, ground_truths):
        evaluator.add_llm_output(llm_output)
        evaluator.add_ground_truth(ground_truth)

    assert len(evaluator.llm_outputs) == 5
    assert len(evaluator.ground_truths) == 5

    evaluator.reset()

    assert len(evaluator.llm_outputs) == 0
    assert len(evaluator.ground_truths) == 0

def test_mismatched_data_raises_error():
    evaluator = StructuredOutputEvaluator()
    llm_outputs, ground_truths = generate_sample_data(5)

    for llm_output in llm_outputs:
        evaluator.add_llm_output(llm_output)

    for ground_truth in ground_truths[:4]:  # Add one less ground truth
        evaluator.add_ground_truth(ground_truth)

    with pytest.raises(ValueError):
        evaluator.evaluate()
