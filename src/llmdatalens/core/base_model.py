from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class LLMEvaluator(BaseModel):
    """Base class for LLM evaluators."""
    metrics: List[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_metric(self, metric_name: str):
        """Add a metric to the evaluator."""
        from llmdatalens.core.metrics_registry import metrics_registry
        if metrics_registry.get(metric_name):
            self.metrics.append(metric_name)
        else:
            raise ValueError(f"Metric '{metric_name}' not found in registry")

    def remove_metric(self, metric_name: str):
        """Remove a metric from the evaluator."""
        self.metrics = [m for m in self.metrics if m != metric_name]

    def evaluate(self):
        """Evaluate the LLM outputs against the ground truths."""
        raise NotImplementedError("Subclasses must implement evaluate method")

    def reset(self):
        """Reset the evaluator, clearing all data."""
        raise NotImplementedError("Subclasses must implement reset method")

class MetricConfig(BaseModel):
    """Configuration for a metric."""
    name: str
    field: str
    description: str

class BaseEvaluationResult(BaseModel):
    """Base model for evaluation results."""
    metrics: Dict[str, Any]
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __str__(self):
        metrics_str = ", ".join(f"{k}: {v}" for k, v in self.metrics.items())
        return f"EvaluationResult(metrics={{{metrics_str}}})"

    def __repr__(self):
        return self.__str__()

class LLMOutputData(BaseModel):
    raw_output: str
    structured_output: Dict[str, Any]
    metadata: Dict[str, Any]

class GroundTruthData(BaseModel):
    data: Dict[str, Any]
    metadata: Dict[str, Any] = {}
