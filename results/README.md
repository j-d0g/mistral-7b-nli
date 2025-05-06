# Results Directory

This directory stores the outputs of model evaluation runs.

## Content

After running evaluations, this directory will contain:

- **JSON files** (`[model_name]-[dataset_name]-[timestamp].json`): Detailed output with full model responses
- **CSV files** (`[model_name]-[dataset_name]-[timestamp].csv`): Simplified output with predictions
- **Checkpoint files** (`checkpoint_[model_name]-[dataset_name]-[timestamp].json`): Saved progress during evaluation

## Usage

To run an evaluation and generate results:

```bash
# Basic evaluation
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv

# Specify GPU to use
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv --gpu 1
```

For detailed documentation on the evaluation process and interpreting results, please refer to the [EVALUATION.md](../EVALUATION.md) document in the project root.