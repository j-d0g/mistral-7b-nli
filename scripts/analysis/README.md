# Analysis Scripts

This directory contains the scripts that compute the numbers and draw the figures used in the documentation. All of them read the public data and result files (`python3 data/download_data.py`, `python3 results/download_results.py`) and write to `metrics/`.

### Scripts

- `compute_test_metrics.py`: accuracy, macro precision/recall/F1 and MCC for each result file, plus a check that no evaluated pair appears in `train.csv`
- `analyze_finetuned_results.py`: fine-tuned model accuracy by estimated rationale length (`metrics/token_vs_accuracy.png`, `metrics/finetuned_token_vs_accuracy.json`)
- `analyze_original_thoughts.py`: agreement of the `open-mistral-7b` rationales with the gold label by estimated length (`metrics/original_token_vs_accuracy.png` and `.json`)
- `analyze_token_accuracy.py`: same analysis as `analyze_finetuned_results.py` with a configurable results file; also writes the result into `card_metrics.json`
- `analyze_token_lengths.py`: split sizes, label balance and estimated token statistics for `card_metrics.json`
- `generate_card_metrics.py`: older metrics generator kept for reference
- `generate_card_visualizations.py`: diagrams and the `card_metrics.json`-based figures (`dataset_statistics.png`, `token_lengths.png`, `model_performance.png`)
- `plot_training_dynamics.py`: validation loss and token accuracy from the public `trainer_state.json` logs (`metrics/training_dynamics.png`)
- `visualize_wandb_training.py`: the same curves from Weights & Biases, if you have access to the runs

Token counts in these scripts are estimated as characters / 4, not with the Mistral tokenizer.

### Output Locations

- `metrics/`

## Dependencies

All scripts require the dependencies listed in the project's `requirements.txt` file. The main visualization libraries used are:
- matplotlib
- seaborn
- pandas
- numpy 