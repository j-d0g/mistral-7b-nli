# Evaluation Components

This directory contains the scripts and utilities for evaluating NLI models.

## Key Components

- `sample_model.py`: Core implementation for model inference and evaluation
- `parse_predictions.py`: Utilities for extracting predictions from model outputs

## Usage

For detailed documentation on the evaluation process, including quickstart guides and technical details, please refer to the [EVALUATION.md](../EVALUATION.md) document in the project root.

Basic usage:

```bash
# From the project root:
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv
```

See the main [EVALUATION.md](../EVALUATION.md) for more options and detailed explanations. 