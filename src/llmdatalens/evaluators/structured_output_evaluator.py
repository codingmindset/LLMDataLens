from typing import Dict, Any, List, Optional
from pydantic import Field
from llmdatalens.core.base_model import LLMEvaluator
from llmdatalens.core.metrics import start_timer, end_timer
from llmdatalens.core.metrics_registry import metrics_registry
from llmdatalens.core.enums import MetricField
from llmdatalens.experiment.models import (
    LLMStructuredOutput,
    GroundTruth,
    EvaluationResult,
    Run,
    Metadata,
    Prompt,
    FunctionSchema,
    FieldResult  # Add this import
)
from llmdatalens.experiment.experiment_manager import ExperimentManager
from .field_evaluators import create_field_evaluator, StringFieldEvaluator

class StructuredOutputEvaluator(LLMEvaluator):
    llm_outputs: List[LLMStructuredOutput] = Field(default_factory=list)
    ground_truths: List[GroundTruth] = Field(default_factory=list)
    experiment_manager: ExperimentManager = Field(default_factory=ExperimentManager)
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_version: Optional[str] = None
    openai_api_key: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.experiment_name and self.experiment_version:
            self.experiment_id = self.experiment_manager.create_or_load_experiment(
                self.experiment_name, 
                self.experiment_version
            )
        if 'openai_api_key' in data:
            self.openai_api_key = data['openai_api_key']

    def evaluate(self) -> EvaluationResult:
        self._validate_data()
        evaluation_results = []

        for llm_output, ground_truth in zip(self.llm_outputs, self.ground_truths):
            start_time = start_timer()
            result = self._evaluate_single_output(llm_output, ground_truth)
            end_time = end_timer(start_time)

            evaluation_results.append(result)

            run = Run(
                llm_output=llm_output,
                ground_truth=ground_truth,
                evaluation_result=result
            )
            self.experiment_manager.add_run(self.experiment_id, run)

        overall_result = self._aggregate_results(evaluation_results)
        return overall_result

    def _evaluate_single_output(self, llm_output: LLMStructuredOutput, ground_truth: GroundTruth) -> EvaluationResult:
        function_schema = llm_output.metadata.prompt.function_call
        predicted_output = llm_output.structured_output
        gt_output = ground_truth.data  # Make sure this is the correct golden_data

        field_results = {}
        total_correct = 0
        total_fields = 0

        for field_name, field_schema in function_schema.parameters["properties"].items():
            field_evaluator = create_field_evaluator(field_name, field_schema)
            if isinstance(field_evaluator, StringFieldEvaluator):
                field_evaluator.llm_evaluator.api_key = self.openai_api_key
            evaluation = field_evaluator.evaluate(
                predicted_output.get(field_name),
                gt_output.get(field_name)
            )
            
            field_results[field_name] = FieldResult(
                correct=evaluation.get("correct", False),
                predicted=evaluation.get("predicted"),
                ground_truth=evaluation.get("ground_truth"),
                details={k: v for k, v in evaluation.items() if k not in ["correct", "predicted", "ground_truth"]}
            )
            
            if evaluation.get("correct", False):
                total_correct += 1
            total_fields += 1

        overall_accuracy = total_correct / total_fields if total_fields > 0 else 0

        return EvaluationResult(
            overall_accuracy=overall_accuracy,
            field_results=field_results
        )

    def _aggregate_results(self, evaluation_results: List[EvaluationResult]) -> EvaluationResult:
        total_accuracy = sum(result.overall_accuracy for result in evaluation_results)
        average_accuracy = total_accuracy / len(evaluation_results) if evaluation_results else 0

        aggregated_field_results = {}
        for result in evaluation_results:
            for field_name, field_result in result.field_results.items():
                if field_name not in aggregated_field_results:
                    aggregated_field_results[field_name] = []
                aggregated_field_results[field_name].append(field_result)

        final_field_results = {
            field_name: FieldResult(
                correct=all(fr.correct for fr in field_results),
                predicted=[fr.predicted for fr in field_results],
                ground_truth=[fr.ground_truth for fr in field_results],
                details={"individual_results": [fr.details for fr in field_results]}
            )
            for field_name, field_results in aggregated_field_results.items()
        }

        return EvaluationResult(
            overall_accuracy=average_accuracy,
            field_results=final_field_results,
            details={"num_evaluations": len(evaluation_results)}
        )

    def _validate_data(self):
        if len(self.llm_outputs) != len(self.ground_truths):
            raise ValueError("Number of LLM outputs and ground truths must match")

    def add_llm_output(self, output: LLMStructuredOutput):
        """Add an LLM structured output to the evaluator."""
        self.llm_outputs.append(output)

    def add_ground_truth(self, ground_truth: GroundTruth):
        """Add a ground truth to the evaluator."""
        self.ground_truths.append(ground_truth)

    def _process_data(self) -> Dict[str, Any]:
        ground_truths = [gt.data for gt in self.ground_truths]
        predictions = [llm.structured_output for llm in self.llm_outputs]
        latencies = []
        confidences = []
        total_time = 0

        for llm_output in self.llm_outputs:
            latency = llm_output.metadata.latency or 0
            latencies.append(latency)
            total_time += latency
            confidences.append(llm_output.metadata.confidence or 1.0)

        return {
            "ground_truths": ground_truths,
            "predictions": predictions,
            "latencies": latencies,
            "confidences": confidences,
            "total_time": total_time,
            "total_items": len(self.llm_outputs)
        }

    def _calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        metric_results = {}
        for metric_name in self.metrics:
            metric_info = metrics_registry.get(metric_name)
            if metric_info is not None:
                input_data = {key: data[key] for key in metric_info.input_keys if key in data}
                metric_results[metric_name] = metric_info.func(**input_data)
            else:
                print(f"Warning: Metric '{metric_name}' not found in registry.")
        return metric_results

    def _create_evaluation_result(self, metric_results: Dict[str, Any], data: Dict[str, Any]) -> EvaluationResult:
        return EvaluationResult(
            metrics=metric_results,
            details={
                "total_items": data["total_items"],
                "total_time": data["total_time"],
            }
        )

    def _get_model_info(self) -> Dict[str, Any]:
        if self.llm_outputs:
            return {
                "name": self.llm_outputs[0].metadata.model_name,
                "version": self.llm_outputs[0].metadata.model_version
            }
        return {}

    def _get_prompt_info(self) -> Dict[str, Any]:
        if self.llm_outputs:
            prompt = self.llm_outputs[0].metadata.prompt
            if prompt:
                return {
                    "system": prompt.system,
                    "user": prompt.user,
                    "function_call": prompt.function_call.dict() if prompt.function_call else None
                }
        return {}
