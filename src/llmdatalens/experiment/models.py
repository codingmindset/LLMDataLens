from pydantic import BaseModel, Field, ConfigDict, validator, field_validator
from typing import Dict, Any, List, Union, Optional, Literal
from datetime import datetime
from uuid import uuid4

class FunctionSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Prompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    system: Optional[str] = None
    user: Optional[str] = None
    function_call: Optional[FunctionSchema] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)

class Metadata(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    prompt: Optional[Prompt] = None
    latency: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(protected_namespaces=())

class LLMTextOutput(BaseModel):
    """Model for LLM natural text output."""
    output_type: Literal["text"] = "text"
    raw_output: str = Field(min_length=1)
    metadata: Metadata

class LLMStructuredOutput(BaseModel):
    """Model for LLM structured output data."""
    output_type: Literal["structured"] = "structured"
    structured_output: Union[Dict[str, Any], BaseModel] = Field(...)
    metadata: Metadata

class GroundTruth(BaseModel):
    """Model for ground truth data."""
    data: Union[Dict[str, Any], BaseModel] = Field(...)

class FieldResult(BaseModel):
    correct: bool
    predicted: Any
    ground_truth: Any
    details: Optional[Dict[str, Any]] = None

class EvaluationResult(BaseModel):
    field_accuracy: float
    field_results: Dict[str, FieldResult]
    details: Optional[Dict[str, Any]] = None

class Run(BaseModel):
    """Model for a single run in an experiment."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    llm_output: Union[LLMTextOutput, LLMStructuredOutput]
    ground_truth: Optional[GroundTruth] = None
    evaluation_result: Optional[EvaluationResult] = None

    model_config = ConfigDict(protected_namespaces=())

    @field_validator('llm_output')
    def check_llm_output_type(cls, v):
        if not isinstance(v, (LLMTextOutput, LLMStructuredOutput)):
            raise ValueError("llm_output must be either LLMTextOutput or LLMStructuredOutput")
        return v

class ModelVersion(BaseModel):
    version: str
    first_used: datetime = Field(default_factory=datetime.now)
    last_used: datetime = Field(default_factory=datetime.now)
    run_count: int = 0

class Model(BaseModel):
    name: str
    versions: Dict[str, ModelVersion] = Field(default_factory=dict)

class Experiment(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    version: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    runs: List[Run] = Field(default_factory=list)
    prompts: Dict[str, Prompt] = Field(default_factory=dict)
    models: Dict[str, Model] = Field(default_factory=dict)

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})