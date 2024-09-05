import pytest
from datetime import datetime
from llmdatalens.experiment.models import (
    FunctionSchema, Prompt, Metadata, LLMTextOutput, LLMStructuredOutput,
    GroundTruth, FieldResult, EvaluationResult, Run, ModelVersion, Model, Experiment
)

def test_function_schema():
    schema = FunctionSchema(
        name="test_function",
        description="A test function",
        parameters={"param1": {"type": "string"}}
    )
    assert schema.name == "test_function"
    assert schema.description == "A test function"
    assert schema.parameters == {"param1": {"type": "string"}}

def test_prompt():
    prompt = Prompt(
        system="You are a helpful assistant",
        user="What is the capital of France?",
        function_call=FunctionSchema(name="get_capital", parameters={})
    )
    assert prompt.system == "You are a helpful assistant"
    assert prompt.user == "What is the capital of France?"
    assert isinstance(prompt.function_call, FunctionSchema)
    assert isinstance(prompt.created_at, datetime)
    assert isinstance(prompt.modified_at, datetime)

def test_metadata():
    metadata = Metadata(
        model_name="GPT-3",
        model_version="1.0",
        prompt=Prompt(user="Test prompt"),
        latency=0.5,
        confidence=0.9
    )
    assert metadata.model_name == "GPT-3"
    assert metadata.model_version == "1.0"
    assert isinstance(metadata.prompt, Prompt)
    assert metadata.latency == 0.5
    assert metadata.confidence == 0.9

def test_llm_text_output():
    output = LLMTextOutput(
        raw_output="This is a test output",
        metadata=Metadata(model_name="GPT-3")
    )
    assert output.output_type == "text"
    assert output.raw_output == "This is a test output"
    assert isinstance(output.metadata, Metadata)

def test_llm_structured_output():
    output = LLMStructuredOutput(
        structured_output={"key": "value"},
        metadata=Metadata(model_name="GPT-3")
    )
    assert output.output_type == "structured"
    assert output.structured_output == {"key": "value"}
    assert isinstance(output.metadata, Metadata)

def test_ground_truth():
    truth = GroundTruth(data={"key": "value"})
    assert truth.data == {"key": "value"}

def test_field_result():
    result = FieldResult(
        correct=True,
        predicted="value",
        ground_truth="value",
        details={"score": 1.0}
    )
    assert result.correct == True
    assert result.predicted == "value"
    assert result.ground_truth == "value"
    assert result.details == {"score": 1.0}

def test_evaluation_result():
    result = EvaluationResult(
        field_accuracy=0.8,
        field_results={
            "field1": FieldResult(correct=True, predicted="value", ground_truth="value"),
            "field2": FieldResult(correct=False, predicted="wrong", ground_truth="right")
        }
    )
    assert result.field_accuracy == 0.8
    assert len(result.field_results) == 2
    assert result.field_results["field1"].correct == True
    assert result.field_results["field2"].correct == False

def test_run():
    run = Run(
        llm_output=LLMStructuredOutput(
            structured_output={"key": "value"},
            metadata=Metadata(model_name="GPT-3")
        ),
        ground_truth=GroundTruth(data={"key": "value"}),
        evaluation_result=EvaluationResult(
            field_accuracy=1.0,
            field_results={"key": FieldResult(correct=True, predicted="value", ground_truth="value")}
        )
    )
    assert isinstance(run.llm_output, LLMStructuredOutput)
    assert isinstance(run.ground_truth, GroundTruth)
    assert isinstance(run.evaluation_result, EvaluationResult)

def test_model_version():
    version = ModelVersion(version="1.0")
    assert version.version == "1.0"
    assert isinstance(version.first_used, datetime)
    assert isinstance(version.last_used, datetime)
    assert version.run_count == 0

def test_model():
    model = Model(
        name="GPT-3",
        versions={"1.0": ModelVersion(version="1.0")}
    )
    assert model.name == "GPT-3"
    assert "1.0" in model.versions
    assert isinstance(model.versions["1.0"], ModelVersion)

def test_experiment():
    experiment = Experiment(
        name="Test Experiment",
        version="1.0",
        description="A test experiment",
        runs=[Run(
            llm_output=LLMStructuredOutput(
                structured_output={"key": "value"},
                metadata=Metadata(model_name="GPT-3")
            )
        )],
        prompts={"prompt1": Prompt(user="Test prompt")},
        models={"GPT-3": Model(name="GPT-3")}
    )
    assert experiment.name == "Test Experiment"
    assert experiment.version == "1.0"
    assert experiment.description == "A test experiment"
    assert len(experiment.runs) == 1
    assert "prompt1" in experiment.prompts
    assert "GPT-3" in experiment.models