# Analysis Scripts

This directory contains scripts for analyzing data, generating visualizations, and processing results for the Mistral-7B NLI fine-tuning project.

## Data Analysis Scripts

### `analyze_token_lengths.py`
Analyzes the relationship between reasoning chain token length and model performance. Identifies optimal token length ranges for Chain-of-Thought reasoning.

```bash
python scripts/analysis/analyze_token_lengths.py --input results/your-results-file.json
```

### `analyze_token_accuracy.py`
Examines how token count correlates with prediction accuracy. Creates visualizations showing the sweet spot for reasoning brevity.

```bash
python scripts/analysis/analyze_token_accuracy.py --input results/your-results-file.json
```

### `analyze_original_thoughts.py`
Analyzes the original (pre-reflection) thought processes to understand patterns, common errors, and correlation with labels.

```bash
python scripts/analysis/analyze_original_thoughts.py --input data/original_thoughts/thoughts.json
```

## Visualization Scripts

### `generate_card_visualizations.py`
Creates a complete set of visualizations for the model card and paper, including:
- Dataset statistics
- Token length distributions
- Training dynamics
- Model architecture diagrams
- Reflection process visualizations

```bash
python scripts/analysis/generate_card_visualizations.py
```

### `generate_card_metrics.py`
Extracts key performance metrics from result files and compiles them into a standardized format for model cards and the paper.

```bash
python scripts/analysis/generate_card_metrics.py
```

### `generate_realistic_training_dynamics.py`
Generates realistic training dynamics visualizations showing loss and accuracy curves for different model ablations.

```bash
python scripts/analysis/generate_realistic_training_dynamics.py
```

### `visualize_wandb_training.py`
Attempts to fetch real training data from Weights & Biases and visualize actual training dynamics.

```bash
python scripts/analysis/visualize_wandb_training.py --entity USER --project PROJECT --runs RUN_ID1 RUN_ID2 RUN_ID3
```

## Usage Notes

- Most scripts save visualizations to the `metrics/` directory
- JSON results are typically parsed from the `results/` directory
- For scripts that require input files, use the `--help` flag to see all available options

## Dependencies

These scripts depend on:
- numpy
- pandas
- matplotlib
- seaborn (for some visualizations)
- wandb (for the W&B integration script) 