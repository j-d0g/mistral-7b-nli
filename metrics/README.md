# Metrics and Visualizations

This directory contains the metrics and visualization data for the Mistral-7B NLI project. The PNG image files are excluded from version control but can be regenerated using the analysis scripts.

## Visualization Files

When generated, this directory will contain the following visualization files:

- `data_pipeline.png`: Visualization of the data processing pipeline
- `dataset_banner.png`: Banner image for the dataset card
- `dataset_statistics.png`: Statistics about the dataset
- `model_architecture.png`: Architecture diagram of the model
- `model_performance.png`: Performance metrics of the model
- `original_token_vs_accuracy.png`: Analysis of token length vs accuracy for original thoughts
- `reasoning_benefits.png`: Visualization showing benefits of reasoned outputs
- `reflection_process.png`: Diagram of the reflection process
- `token_accuracy_comparison_combined.png`: Combined comparison of token accuracy
- `token_lengths.png`: Distribution of token lengths
- `token_vs_accuracy.png`: Analysis of token length vs accuracy
- `training_dynamics.png`: Visualization of training dynamics

## Regenerating Visualizations

To regenerate all visualization files, run from the project root:

```bash
./run_metrics.sh
```

Or to regenerate specific visualizations:

```bash
python scripts/analysis/generate_card_visualizations.py
```

## Metrics Data

JSON files in this directory contain the raw metrics data used to generate visualizations:

- `card_metrics.json`: Metrics for the model and dataset cards
- `original_token_vs_accuracy.json`: Data on token length vs accuracy
- `finetuned_token_vs_accuracy.json`: Data on token length vs accuracy for the finetuned model

See the `scripts/analysis/README.md` file for more information on how these metrics are generated. 