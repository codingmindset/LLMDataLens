from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from llmdatalens.core.metrics_registry import metrics_registry
from llmdatalens.core.enums import MetricField

class LLMOutputData(BaseModel):
    """Base model for LLM output data."""
    raw_output: str = Field(..., min_length=1)
    structured_output: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('raw_output')
    @classmethod
    def raw_output_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('raw_output must not be empty')
        return v

    @field_validator('structured_output')
    @classmethod
    def check_structured_output_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("structured_output must not be empty")
        return v

class GroundTruthData(BaseModel):
    """Base model for ground truth data."""
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('data')
    @classmethod
    def check_data_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("data must not be empty")
        return v

class EvaluationResult(BaseModel):
    """Base model for evaluation results."""
    metrics: Dict[str, Any]
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MetricConfig(BaseModel):
    """Configuration for a metric."""
    name: str
    field: MetricField
    description: str

class LLMEvaluator(BaseModel):
    """Base class for LLM evaluators."""
    metrics: List[str] = Field(default_factory=list)
    llm_outputs: List[LLMOutputData] = Field(default_factory=list)
    ground_truths: List[GroundTruthData] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_metric(self, metric_name: str):
        """Add a metric to the evaluator."""
        if metrics_registry.get(metric_name):
            self.metrics.append(metric_name)
        else:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

    def remove_metric(self, metric_name: str):
        """Remove a metric from the evaluator."""
        self.metrics = [m for m in self.metrics if m != metric_name]

    def add_llm_output(self, output: LLMOutputData, latency: Optional[float] = None, confidence: Optional[float] = None):
        """Add an LLM output to the evaluator with optional latency and confidence."""
        if latency is not None or confidence is not None:
            output.metadata = output.metadata or {}
            if latency is not None:
                output.metadata['latency'] = latency
            if confidence is not None:
                output.metadata['confidence'] = confidence
        self.llm_outputs.append(output)

    def add_ground_truth(self, ground_truth: GroundTruthData):
        """Add a ground truth to the evaluator."""
        self.ground_truths.append(ground_truth)

    def evaluate(self) -> EvaluationResult:
        """Evaluate the LLM outputs against the ground truths."""
        raise NotImplementedError("Subclasses must implement evaluate method")

    def reset(self):
        """Reset the evaluator, clearing all data."""
        self.llm_outputs.clear()
        self.ground_truths.clear()

    def get_metric_info(self) -> Dict[str, MetricConfig]:
        """
        Get information about the metrics used by this evaluator.

        :return: A dictionary with metric names as keys and their configurations as values
        """
        metric_info = {}
        for metric_name in self.metrics:
            metric = metrics_registry.get(metric_name)
            if metric:
                metric_info[metric_name] = MetricConfig(
                    name=metric_name,
                    field=metric.field,
                    description=metric.description
                )
            else:
                # Handle the case where a metric is not found in the registry
                metric_info[metric_name] = MetricConfig(
                    name=metric_name,
                    field=MetricField.Other,
                    description="Metric not found in registry"
                )
        return metric_info
