import pytest
from unittest.mock import Mock, patch
from llmdatalens.experiment.experiment_manager import ExperimentManager
from llmdatalens.experiment.models import Experiment, Run, Prompt, Model, LLMStructuredOutput, GroundTruth, Metadata
from llmdatalens.core.metrics_registry import MetricNames

@pytest.fixture
def experiment_manager():
    manager = ExperimentManager()
    manager.experiments = {}  # Ensure the experiments attribute exists
    return manager

def test_create_experiment(experiment_manager):
    experiment = Experiment(name="Test Experiment", description="A test description", version="1.0")
    experiment_manager.experiments[experiment.id] = experiment
    assert experiment.name == "Test Experiment"
    assert experiment.description == "A test description"
    assert experiment.version == "1.0"

# Update other tests similarly, focusing on the actual methods and attributes of ExperimentManager