# LLMDataLens

[![PyPI version](https://badge.fury.io/py/llm-data-lens.svg)](https://badge.fury.io/py/llm-data-lens)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-data-lens.svg)](https://pypi.org/project/llm-data-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/llmdatalens/badge/?version=latest)](https://llmdatalens.readthedocs.io/en/latest/?badge=latest)

LLMDataLens is a powerful and flexible framework for evaluating LLM-based applications with structured output. It provides a comprehensive suite of tools for assessing the performance of language models across various metrics, with a focus on experiment tracking and reproducibility.

## 🌟 Features

- **Structured Output Evaluation**: Assess LLM outputs against ground truth data with precision.
- **Customizable Metrics**: Easily define and use custom metrics for comprehensive performance assessment.
- **Experiment Tracking**: Built-in experiment management for reproducibility and comparison.
- **Prompt Versioning**: Keep track of prompt evolution and its impact on model performance.
- **Model Version Tracking**: Monitor performance across different model versions.
- **Flexible Integration**: Seamlessly integrate with existing LLM pipelines and workflows.
- **Extensible Architecture**: Add custom metrics, evaluators, and experiment trackers with ease.

## 🚀 Quick Start

### Installation

Install LLMDataLens directly from PyPI:

```bash
pip install llm-data-lens
```

For development or to get the latest version from the repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/codingmindset/LLMDataLens.git
   cd llmdatalens
   ```

2. Install the package using Poetry:
   ```bash
   poetry install
   ```

### Basic Usage

Here's a simple example to get you started:

```python
from llmdatalens.evaluators import StructuredOutputEvaluator
from llmdatalens.core import LLMOutputData, GroundTruthData
from llmdatalens.core.metrics_registry import MetricNames

# Create an evaluator with specific metrics
evaluator = StructuredOutputEvaluator(
    metrics=[MetricNames.OverallAccuracy, MetricNames.AverageLatency],
    experiment_name="Invoice Processing Experiment"
)

# Add LLM output and ground truth data
llm_output = LLMOutputData(
    raw_output="Processed invoice: $100",
    structured_output={"invoice_amount": 100},
    metadata={
        "model_info": {"name": "GPT-3.5", "version": "1.0"},
        "prompt_info": {"text": "Extract invoice amount:"}
    }
)
ground_truth = GroundTruthData(
    data={"invoice_amount": 100}
)

evaluator.add_llm_output(llm_output, latency=0.5, confidence=0.9)
evaluator.add_ground_truth(ground_truth)

# Evaluate
result = evaluator.evaluate()

# Print results
print(result.metrics)

# Access experiment data
experiment = evaluator.experiment_manager.get_experiment(evaluator.experiment_id)
print(f"Experiment: {experiment.name}")
print(f"Number of runs: {len(experiment.runs)}")
print(f"Prompts used: {len(experiment.prompts)}")
print(f"Models used: {list(experiment.models.keys())}")
```

## 📊 Advanced Features

### Custom Metrics

Create and register custom metrics easily:

```python
from llmdatalens.core.metrics_registry import register_metric
from llmdatalens.core.enums import MetricField

@register_metric("CustomF1Score", field=MetricField.Accuracy, input_keys=["y_true", "y_pred"])
def calculate_custom_f1_score(y_true, y_pred):
    """ This description will be shown in the metrics registry """
    # (Your custom F1 score calculation here
    pass
```

### Experiment Tracking

Track experiments, prompts, and model versions:

```python
# Get prompt history
prompt_history = evaluator.experiment_manager.get_prompt_history(evaluator.experiment_id)

# Get model history
model_history = evaluator.experiment_manager.get_model_history(evaluator.experiment_id)

# Compare runs
for run in experiment.runs:
    print(f"Run {run.id}: {run.metrics}")
```


For more detailed examples, check the `examples/` directory in the repository. (More examples will be added soon!)

## 📘 Documentation
(Comming soon!)


## 🛠️ Project Structure

```
llmdatalens/
├── src/
│   └── llmdatalens/
│       ├── core/
│       │   ├── base_model.py
│       │   ├── enums.py
│       │   └── metrics_registry.py
│       ├── evaluators/
│       │   └── structured_output_evaluator.py
│       └── experiment/
│           ├── experiment_manager.py
│           └── models.py
├── tests/
│   ├── test_core/
│   ├── test_evaluators/
│   └── test_experiment/
├── examples/
├── docs/
├── pyproject.toml
└── README.md
```

## 🤝 Contributing

We welcome contributions to LLMDataLens! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

LLMDataLens is released under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

## 📬 Contact

If you have any questions, suggestions, or just want to say hi, feel free to reach out:

- **Email**: [elvin@codingmindset.io](mailto:elvin@codingmindset.io)
- **X**: [@codingmindset](https://x.com/codingmindsetio)
- **GitHub Issues**: For bug reports and feature requests

## 🙏 Acknowledgements

- Thanks to all our contributors and users for their valuable feedback and support.
- Special thanks to the open-source community for the amazing tools and libraries that made this project possible.

---

Built with ❤️ by [Coding Mindset](https://codingmindset.io)

---

## Citing LLMDataLens

If you use LLMDataLens in your research, please cite it as follows:

```bibtex
@software{llmdatalens,
  title = {LLMDataLens: A Framework for Evaluating LLM-based Applications},
  author = {Elvin Gomez},
  year = {2024},
  url = {https://github.com/codingmindset/LLMDataLens.git},
}
```
