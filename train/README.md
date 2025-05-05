# Training System Components

This directory contains the Python code and configuration files for fine-tuning the Mistral-7B NLI model using QLoRA.

## Key Components

*   **`train_sft.py`**: The main Python script that orchestrates the Supervised Fine-Tuning (SFT) process using the `trl` library. It handles loading data, configuring the model (QLoRA), setting up the trainer, and running the training loop.
*   **`config_loader.py`**: A utility script responsible for loading training configurations. It merges defaults from `configs/default.py`, specific experiment settings from `configs/*.py`, and command-line overrides.
*   **`configs/`**: A directory containing Python-based configuration files:
    *   `default.py`: Base configuration with default parameters for all training runs.
    *   `ablation*.py`: Example configurations used for different experiments/ablations (e.g., `ablation2.py` for the primary reflected data strategy).
    *   `minimal_test.py`: A minimal config for quick testing.
    *   `distributed.py`: Example config for multi-GPU setup (check compatibility/implementation details).

## Dataset and Data Pipeline

This training system expects fine-tuning data in JSONL format as specified in the configuration files (see `configs/default.py`). The default paths are:

- `data/finetune/train_ft.jsonl` - Training data
- `data/finetune/dev_ft.jsonl` - Validation data

For details on the dataset structure, data generation pipeline, and how to download pre-generated datasets, please refer to the [Data README](../data/README.md).

## How to Run Training

Training is **always** initiated via the `run_training.sh` script in the project root directory. This script handles Docker setup and passes arguments to `train_sft.py`.

**Basic Usage Examples (from project root):**

```bash
# Run with default configuration
./run_training.sh

# Use a specific configuration file (e.g., the primary one)
./run_training.sh --config train/configs/ablation0.py

# Specify GPU (e.g., GPU ID 1)
./run_training.sh --config train/configs/ablation0.py --gpu 1

# Run with default distributed set-up
./run_training.sh --all-gpus --config train/configs/distributed.py

# Run custom ablation overriding default distributed set-up
./run_training.sh --all-gpus --config train/configs/ablation0_dist.py
```

---

**For comprehensive details on the training process, including:**
*   Detailed setup instructions (Docker, environment)
*   In-depth explanation of the configuration system
*   Breakdown of the `train_sft.py` script logic
*   Rationale behind QLoRA and hyperparameter choices
*   Weights & Biases integration guide
*   Extensive troubleshooting tips

**Please refer to the main [TRAINING.md](../TRAINING.md) document in the project root.**

*(Note: A more verbose, potentially older guide ([FINETUNE_GUIDE.md](./FINETUNE_GUIDE.md)) also exists in this directory and may contain supplementary details, but `TRAINING.md` is intended as the primary, consolidated reference.)* 