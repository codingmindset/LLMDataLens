import pytest
from pydantic import ValidationError
from llmdatalens.core.base_model import LLMOutputData, GroundTruthData, EvaluationResult

def test_llm_output_data():
    llm_output = LLMOutputData(
        raw_output="Test output",
        structured_output={"key": "value"},
        metadata={"confidence": 0.9}
    )
    assert llm_output.raw_output == "Test output"
    assert llm_output.structured_output == {"key": "value"}
    assert llm_output.metadata == {"confidence": 0.9}

def test_ground_truth_data():
    ground_truth = GroundTruthData(
        data={"key": "value"},
        metadata={"source": "human_annotator"}
    )
    assert ground_truth.data == {"key": "value"}
    assert ground_truth.metadata == {"source": "human_annotator"}

def test_evaluation_result():
    result = EvaluationResult(
        metrics={"accuracy": 0.9, "f1_score": 0.85},
        details={"total_items": 100}
    )
    assert result.metrics == {"accuracy": 0.9, "f1_score": 0.85}
    assert result.details == {"total_items": 100}

def test_invalid_llm_output_data():
    with pytest.raises((ValidationError, ValueError)):
        LLMOutputData(raw_output="", structured_output={"key": "value"})  # Empty raw_output
    with pytest.raises((ValidationError, ValueError)):
        LLMOutputData(raw_output="test", structured_output={})  # Empty structured_output

def test_invalid_ground_truth_data():
    with pytest.raises((ValidationError, ValueError)):
        GroundTruthData(data={})  # Empty data dictionary
