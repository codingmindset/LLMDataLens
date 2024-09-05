import pytest
import json
from llmdatalens.evaluators.field_evaluators import (
    NumberFieldEvaluator, StringFieldEvaluator, EnumFieldEvaluator, ArrayFieldEvaluator
)
from llmdatalens.evaluators.llm_evaluator import LLMEvaluator
from unittest.mock import Mock

def test_number_field_evaluator():
    evaluator = NumberFieldEvaluator(field_name="test", field_schema={"type": "number"})
    result = evaluator.evaluate(10, 10)
    assert result["correct"] == True
    assert result["predicted"] == 10
    assert result["ground_truth"] == 10

    result = evaluator.evaluate(10, 10.1)
    assert result["correct"] == False

    # Adjust this test based on the actual tolerance in your implementation
    result = evaluator.evaluate(1e6, 1e6 + 1e-5)
    assert result["correct"] == False

def test_string_field_evaluator():
    evaluator = StringFieldEvaluator(field_name="test", field_schema={"type": "string"})
    result = evaluator.evaluate("hello", "hello")
    assert result["correct"] == True
    assert result["predicted"] == "hello"
    assert result["ground_truth"] == "hello"

    result = evaluator.evaluate("hello", "world")
    assert result["correct"] == False

def test_enum_field_evaluator():
    evaluator = EnumFieldEvaluator(field_name="test", field_schema={"type": "string", "enum": ["red", "green", "blue"]})
    result = evaluator.evaluate("red", "red")
    assert result["correct"] == True
    assert result["predicted"] == "red"
    assert result["ground_truth"] == "red"

    result = evaluator.evaluate("yellow", "red")
    assert result["correct"] == False

def test_array_field_evaluator():
    evaluator = ArrayFieldEvaluator(
        field_name="test",
        field_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"}
                }
            }
        }
    )
    predicted = [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}, {"name": "item3", "value": 30}]
    ground_truth = [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}, {"name": "item3", "value": 30}]
    result = evaluator.evaluate(predicted, ground_truth)
    assert result["correct"] == True
    assert result["predicted"] == predicted
    assert result["ground_truth"] == ground_truth

    predicted = [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}]
    ground_truth = [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}, {"name": "item3", "value": 30}]
    result = evaluator.evaluate(predicted, ground_truth)
    assert result["correct"] == False

def test_number_field_evaluator_with_custom_tolerances():
    evaluator = NumberFieldEvaluator(
        field_name="test",
        field_schema={"type": "number"},
        relative_tolerance=1e-2,
        absolute_tolerance=1e-1
    )
    result = evaluator.evaluate(100, 101)
    assert result["correct"] == False  # Adjust based on your actual implementation

    result = evaluator.evaluate(100, 100.05)
    assert result["correct"] == True

def test_array_field_evaluator_with_nested_objects():
    evaluator = ArrayFieldEvaluator(
        field_name="test",
        field_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"}
                }
            }
        }
    )
    predicted = [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}]
    ground_truth = [{"name": "item1", "value": 10}, {"name": "item2", "value": 20}, {"name": "item3", "value": 30}]
    result = evaluator.evaluate(predicted, ground_truth)
    assert result["correct"] == False