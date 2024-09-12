import json
import os
from typing import Dict, Any, List, Union, Optional
from datetime import datetime
import hashlib
from .models import Experiment, Run, Prompt, Model, ModelVersion, LLMStructuredOutput, LLMTextOutput, GroundTruth, EvaluationResult, Metadata

class ExperimentManager:
    def __init__(self, storage_path: str = "experiments"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    def create_or_load_experiment(self, name: str, version: str, description: str = "") -> str:
        existing_experiment = self._find_existing_experiment(name, version)
        if existing_experiment:
            return existing_experiment.id
        
        experiment = Experiment(name=name, version=version, description=description)
        self._save_experiment(experiment)
        return experiment.id

    def _find_existing_experiment(self, name: str, version: str) -> Optional[Experiment]:
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                try:
                    experiment = self._load_experiment(filename[:-5])  # Remove .json extension
                    if experiment.name == name and experiment.version == version:
                        return experiment
                except Exception as e:
                    print(f"Error loading experiment {filename}: {str(e)}")
        return None

    def add_run(self, experiment_id: str, run: Run) -> str:
        experiment = self._load_experiment(experiment_id)

        # Handle prompt versioning
        if run.llm_output.metadata.prompt:
            prompt = self._get_or_create_prompt(experiment, run.llm_output.metadata.prompt)
            run.llm_output.metadata.additional_info["prompt_id"] = prompt.id
            run.llm_output.metadata.additional_info["prompt_version"] = prompt.version

        # Handle model versioning
        self._update_model_info(experiment, run.llm_output.metadata.model_name, run.llm_output.metadata.model_version)

        experiment.runs.append(run)
        self._save_experiment(experiment)
        return run.id

    def _get_or_create_prompt(self, experiment: Experiment, prompt: Prompt) -> Prompt:
        prompt_hash = self._hash_prompt(prompt)
        
        if prompt_hash not in experiment.prompts:
            prompt.version = f"1.0.{len(experiment.prompts)}"
            experiment.prompts[prompt_hash] = prompt
        else:
            existing_prompt = experiment.prompts[prompt_hash]
            existing_prompt.modified_at = datetime.now()
            prompt = existing_prompt
        
        return prompt

    @staticmethod
    def _hash_prompt(prompt: Prompt) -> str:
        prompt_dict = prompt.model_dump(exclude={'id', 'created_at', 'modified_at', 'version'})
        return hashlib.md5(json.dumps(prompt_dict, sort_keys=True).encode()).hexdigest()

    def _update_model_info(self, experiment: Experiment, model_name: str, model_version: str):
        if model_name not in experiment.models:
            experiment.models[model_name] = Model(name=model_name)
        
        model = experiment.models[model_name]
        if model_version and model_version not in model.versions:
            model.versions[model_version] = ModelVersion(version=model_version)
        
        version_info = model.versions.get(model_version) if model_version else None
        if version_info:
            version_info.last_used = datetime.now()
            version_info.run_count += 1

    def get_experiment(self, experiment_id: str) -> Experiment:
        return self._load_experiment(experiment_id)

    def get_all_experiments(self) -> List[Experiment]:
        experiments = []
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                experiments.append(self._load_experiment(filename[:-5]))  # Remove .json extension
        return experiments

    def get_prompt_history(self, experiment_id: str) -> List[Prompt]:
        experiment = self._load_experiment(experiment_id)
        return list(experiment.prompts.values())

    def get_model_history(self, experiment_id: str) -> Dict[str, Model]:
        experiment = self._load_experiment(experiment_id)
        return experiment.models

    def _save_experiment(self, experiment: Experiment):
        filename = f"{experiment.id}.json"
        with open(os.path.join(self.storage_path, filename), "w") as f:
            json.dump(experiment.model_dump(), f, indent=2, default=self._json_serializer)

    @staticmethod
    def _json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    def _load_experiment(self, experiment_id: str) -> Experiment:
        filename = f"{experiment_id}.json"
        with open(os.path.join(self.storage_path, filename), "r") as f:
            data = json.load(f)
        
        # Check if the data needs migration
        if "version" not in data:
            data = self._migrate_experiment_data(data)
        
        return Experiment.model_validate(data)

    def _migrate_experiment_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["version"] = data.get("version", "1.0.0")  # Set a default version if not present
        
        # Migrate runs
        for run in data.get("runs", []):
            if "evaluation_result" in run:
                eval_result = run["evaluation_result"]
                if "metrics" in eval_result and "overall_accuracy" not in eval_result:
                    # Migrate old format to new format
                    eval_result["overall_accuracy"] = eval_result["metrics"].get("OverallAccuracy", 0.0)
                    eval_result["field_results"] = {}
                    eval_result["details"] = eval_result.get("details", {})
                    eval_result.pop("metrics", None)

        return data
