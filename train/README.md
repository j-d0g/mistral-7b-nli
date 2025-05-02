# Training System for Mistral-7B NLI

This directory contains the training scripts for fine-tuning Mistral-7B models on NLI tasks.

## Key Components

- `train_sft.py`: The main training script that handles the actual fine-tuning process and config loading
- `config_loader.py`: Utility for loading Python configuration files
- `configs/`: Directory containing configuration files for different training runs:
  - `default.py`: Base configuration with default parameters for all runs
  - `initial_test_run.py`: Configuration for the initial test run
  - `ablation1.py`: Configuration for Ablation 1 (standard training)
  - `ablation2.py`: Configuration for Ablation 2 (mixed data optimization)

## Docker-Based Training Workflow

The training system uses Docker to ensure a consistent environment:

1. Training is initiated through `../train.sh`, which:
   - Mounts the necessary volumes (code, data, models, cache)
   - Runs the Python training script within the Docker container
   - Passes all command-line arguments directly to the script

2. The training workflow:
   - Loads the specified configuration file (or default.py if none specified)
   - Processes command-line overrides (these take precedence over config values)
   - Performs the actual fine-tuning using PEFT/QLoRA
   - Saves checkpoints and logs metrics (optionally to W&B)

## Usage

The training system is designed to be used via the `../train.sh` script:

```bash
# Run with default configuration
./train.sh

# Use a specific configuration file
./train.sh --config train/configs/ablation1.py

# Override specific parameters
./train.sh --config train/configs/ablation1.py --batch_size 8 --no_wandb

# Specify GPU to use
./train.sh --config train/configs/ablation1.py --gpu_id 1
```

## Configuration System

The system uses a three-tier approach to configuration:

1. **Default Configuration**: `configs/default.py` contains sensible defaults
2. **Custom Configurations**: Specific configs can override defaults
3. **Command-line Overrides**: CLI arguments take highest precedence

Each configuration file can import and extend the default configuration:

```python
# Import and extend defaults
from train.configs.default import *

# Override specific parameters
output_dir = "models/mistral-7b-nli-cot-ablation1"
batch_size = 16
grad_accumulation_steps = 2  # Effective batch size: 32
learning_rate = 2e-4
```

## Error Handling

The system includes robust error handling:

1. If a specified config file fails to load, it falls back to the default config
2. If the default config fails to load, it uses minimal hardcoded defaults
3. Command-line arguments always take precedence

## Environment Setup

Before running training, make sure:

1. The Docker image is built:
   ```bash
   docker build -t mistral-nli-ft .
   ```

2. The `.env` file exists with any needed environment variables (e.g., WANDB_API_KEY)

3. The necessary data files exist in the expected locations 