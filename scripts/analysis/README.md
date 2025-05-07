# Analysis Scripts

This directory contains all the scripts needed to analyze data and generate visualizations for the Mistral-7B NLI project.

## Visualization Regeneration

All visualizations can be regenerated from the processed data. The PNG files are excluded from version control to keep the repository size small.

### How to Regenerate All Visualizations

Run the following command from the project root:

```bash
./run_metrics.sh
```

This will execute all necessary scripts to analyze the data and regenerate all visualization files.

### Individual Scripts

- `generate_card_visualizations.py`: Generates all visualizations for model and dataset cards
- `generate_card_metrics.py`: Computes metrics for model and dataset cards
- `analyze_token_lengths.py`: Analyzes token lengths of the thoughts and generates visualizations
- `analyze_token_accuracy.py`: Analyzes the relationship between token length and accuracy
- `analyze_original_thoughts.py`: Analyzes original thoughts dataset
- `generate_realistic_training_dynamics.py`: Generates training dynamics visualizations
- `visualize_wandb_training.py`: Visualizes training metrics from Weights & Biases

### Example: Regenerating Card Visualizations Only

```bash
python scripts/analysis/generate_card_visualizations.py
```

### Output Locations

- Model and dataset card visualizations: `metrics/`
- Training analysis visualizations: `metrics/train_analysis/`

## Dependencies

All scripts require the dependencies listed in the project's `requirements.txt` file. The main visualization libraries used are:
- matplotlib
- seaborn
- pandas
- numpy 