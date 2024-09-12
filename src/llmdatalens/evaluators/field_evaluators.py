from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from .llm_evaluator import LLMEvaluator
import json
import math

class FieldEvaluator(BaseModel):
    field_name: str
    field_schema: Dict[str, Any]

    def evaluate(self, predicted_value: Any, ground_truth: Any) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

class NumberFieldEvaluator(FieldEvaluator):
    relative_tolerance: float = Field(default=1e-6)
    absolute_tolerance: float = Field(default=1e-9)

    def evaluate(self, predicted_value: Any, ground_truth: Any) -> Dict[str, Any]:
        if not isinstance(predicted_value, (int, float)) or not isinstance(ground_truth, (int, float)):
            return {
                "correct": False,
                "predicted": predicted_value,
                "ground_truth": ground_truth,
                "error": "Type mismatch"
            }

        relative_error = abs(predicted_value - ground_truth) / max(abs(ground_truth), 1e-9)
        absolute_error = abs(predicted_value - ground_truth)

        is_correct = (relative_error <= self.relative_tolerance) and (absolute_error <= self.absolute_tolerance)

        return {
            "correct": is_correct,
            "predicted": predicted_value,
            "ground_truth": ground_truth,
            "details": {
                "relative_error": relative_error,
                "absolute_error": absolute_error,
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance
            }
        }

class EnumFieldEvaluator(FieldEvaluator):
    def evaluate(self, predicted_value: Any, ground_truth: Any) -> Dict[str, Any]:
        enum_values = self.field_schema.get("enum", [])
        is_correct = predicted_value == ground_truth and predicted_value in enum_values

        return {
            "correct": is_correct,
            "predicted": predicted_value,
            "ground_truth": ground_truth,
            "valid_options": enum_values
        }

class StringFieldEvaluator(FieldEvaluator):
    llm_evaluator: LLMEvaluator = Field(default_factory=LLMEvaluator)
    use_llm: bool = Field(default=False)

    def evaluate(self, predicted_value: Any, ground_truth: Any) -> Dict[str, Any]:
        if not isinstance(predicted_value, str) or not isinstance(ground_truth, str):
            return {
                "correct": False,
                "predicted": predicted_value,
                "ground_truth": ground_truth,
                "error": "Type mismatch"
            }

        if not self.use_llm:
            # For simple string fields, use exact matching
            is_correct = predicted_value == ground_truth
            return {
                "correct": is_correct,
                "predicted": predicted_value,
                "ground_truth": ground_truth,
                "details": {"match_type": "exact"}
            }

        # For complex string fields, use LLM evaluation
        llm_evaluation = self.llm_evaluator.evaluate_relevancy(ground_truth, predicted_value)

        if "error" in llm_evaluation:
            return {
                "correct": False,
                "predicted": predicted_value,
                "ground_truth": ground_truth,
                "error": llm_evaluation["error"]
            }

        relevancy_score = llm_evaluation.get("relevancy_score", 0)
        is_correct = relevancy_score >= 0.8  # We can adjust this threshold as needed

        return {
            "correct": is_correct,
            "predicted": predicted_value,
            "ground_truth": ground_truth,
            "details": llm_evaluation
        }

class ArrayFieldEvaluator(FieldEvaluator):
    def evaluate(self, predicted_value: Any, ground_truth: Any) -> Dict[str, Any]:
        if not isinstance(predicted_value, list) or not isinstance(ground_truth, list):
            return {
                "correct": False,
                "predicted": predicted_value,
                "ground_truth": ground_truth,
                "error": "Type mismatch"
            }

        item_results = []
        correct_items = 0
        total_items = max(len(predicted_value), len(ground_truth))

        for i in range(total_items):
            if i < len(predicted_value) and i < len(ground_truth):
                predicted_item = predicted_value[i]
                ground_truth_item = ground_truth[i]
                
                # Compare each field in the item
                item_correct = True
                for key in set(predicted_item.keys()) | set(ground_truth_item.keys()):
                    if predicted_item.get(key) != ground_truth_item.get(key):
                        item_correct = False
                        break
                
                if item_correct:
                    correct_items += 1
                
                item_results.append({
                    "correct": item_correct,
                    "predicted": predicted_item,
                    "ground_truth": ground_truth_item
                })
            else:
                item_results.append({
                    "correct": False,
                    "predicted": predicted_value[i] if i < len(predicted_value) else None,
                    "ground_truth": ground_truth[i] if i < len(ground_truth) else None,
                    "error": "Missing item in prediction or ground truth"
                })

        array_accuracy = correct_items / total_items if total_items > 0 else 0

        return {
            "correct": array_accuracy == 1,  # Only correct if all items are correct
            "predicted": predicted_value,
            "ground_truth": ground_truth,
            "details": {
                "item_results": item_results,
                "array_accuracy": array_accuracy,
                "correct_items": correct_items,
                "total_items": total_items
            }
        }

def create_field_evaluator(field_name: str, field_schema: Dict[str, Any]) -> FieldEvaluator:
    field_type = field_schema.get("type", "string")
    
    if field_type == "number":
        # Custom tolerances for specific fields
        if field_name == "total":
            return NumberFieldEvaluator(field_name=field_name, field_schema=field_schema, relative_tolerance=1e-4, absolute_tolerance=0.01)
        else:
            return NumberFieldEvaluator(field_name=field_name, field_schema=field_schema)
    elif "enum" in field_schema:
        return EnumFieldEvaluator(field_name=field_name, field_schema=field_schema)
    elif field_type == "array":
        return ArrayFieldEvaluator(field_name=field_name, field_schema=field_schema)
    else:
        use_llm = field_name in ["customer_name", "description"]
        return StringFieldEvaluator(field_name=field_name, field_schema=field_schema, use_llm=use_llm)

__all__ = ['create_field_evaluator', 'StringFieldEvaluator', 'NumberFieldEvaluator', 'EnumFieldEvaluator', 'ArrayFieldEvaluator']
