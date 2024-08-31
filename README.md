# LLMDataLens

[![PyPI version](https://badge.fury.io/py/llm-data-lens.svg)](https://badge.fury.io/py/llm-data-lens)
[![Python Versions](https://img.shields.io/pypi/pyversions/llm-data-lens.svg)](https://pypi.org/project/llm-data-lens/)

LLMDataLens is a powerful framework for evaluating LLM-based applications with structured output. It provides a flexible and extensible way to assess the performance of language models across various metrics.

## Features

- Evaluate LLM outputs against ground truth data
- Customizable metrics for comprehensive performance assessment
- Support for structured output evaluation
- Easy integration with existing LLM pipelines
- Extensible architecture for adding custom metrics and evaluators

## Installation

You can install LLMDataLens directly from PyPI:

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

## Usage

Here's a basic example of how to use LLMDataLens:

```python
from llmdatalens.evaluators import StructuredOutputEvaluator
from llmdatalens.core import LLMOutputData, GroundTruthData

# Create an evaluator with default metrics
evaluator = StructuredOutputEvaluator()

# Add LLM output and ground truth data
llm_output = LLMOutputData(
    raw_output="Processed invoice: $100",
    structured_output={"invoice_amount": 100}
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
```

For more detailed examples, check the `examples/` directory in the repository.

## Documentation
(Comming soon!)


## Project Structure

```
llmdatalens/
├── src/
│   └── llmdatalens/
│       ├── core/
│       │   ├── base_model.py
│       │   ├── enums.py
│       │   └── metrics_registry.py
│       └── evaluators/
│           └── structured_output_evaluator.py
├── tests/
│   ├── test_core/
│   └── test_evaluators/
├── examples/
├── pyproject.toml
└── README.md
```

## Contributing

We welcome contributions to LLMDataLens! Here are some ways you can contribute:

1. Report bugs or suggest features by opening an issue
2. Improve documentation
3. Submit pull requests with bug fixes or new features

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

LLMDataLens is released under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

## Contact

If you have any questions or feedback, please open an issue on GitHub or contact the maintainers at [elvin@codingmindset.io](mailto:elvin@codingmindset.io).

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
