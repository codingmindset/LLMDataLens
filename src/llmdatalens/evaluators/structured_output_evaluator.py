from typing import Dict, Any, List
from llmdatalens.core.base_model import LLMEvaluator, EvaluationResult
from llmdatalens.core.metrics import start_timer, end_timer
from llmdatalens.core.metrics_registry import metrics_registry
from llmdatalens.core.enums import MetricField

class StructuredOutputEvaluator(LLMEvaluator):
    def evaluate(self) -> EvaluationResult:
        self._validate_data()
        processed_data = self._process_data()
        metric_results = self._calculate_metrics(processed_data)
        return self._create_evaluation_result(metric_results, processed_data)

    def _validate_data(self):
        if len(self.llm_outputs) != len(self.ground_truths):
            raise ValueError("Number of LLM outputs and ground truths must match")

    def _process_data(self) -> Dict[str, Any]:
        ground_truths = [gt.data for gt in self.ground_truths]
        predictions = [llm.structured_output for llm in self.llm_outputs]
        latencies = []
        confidences = []
        total_time = 0

        for llm_output in self.llm_outputs:
            latency = llm_output.metadata.get('latency', 0)
            latencies.append(latency)
            total_time += latency
            confidences.append(llm_output.metadata.get('confidence', 1.0))

        return {
            "ground_truths": ground_truths,
            "predictions": predictions,
            "latencies": latencies,
            "confidences": confidences,
            "total_time": total_time,
            "total_items": len(self.llm_outputs)
        }

    def _process_single_item(self, llm_output, ground_truth):
        start_time = start_timer()

        pred = llm_output.structured_output
        truth = ground_truth.data

        latency = llm_output.metadata.get('latency')
        if latency is None:
            latency = end_timer(start_time)

        confidence = llm_output.metadata.get('confidence', 1.0)

        return pred, truth, latency, confidence

    def _calculate_accuracy(self, pred: Dict[str, Any], truth: Dict[str, Any]) -> float:
        return sum(pred[k] == truth[k] for k in truth) / len(truth)

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
