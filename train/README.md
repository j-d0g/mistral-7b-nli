# Training Components

This directory contains the scripts and configuration files for fine-tuning the Mistral-7B model on NLI tasks.

## Key Components

- `train_sft.py`: Main training implementation using PEFT and QLoRA
- `config_loader.py`: Configuration loading utility
- `configs/`: Directory containing various training configurations
  - `default.py`: Base configuration with default parameters
  - `sample_test.py`: Minimal configuration for testing
  - `ablation0.py`, `ablation1.py`, etc.: Configurations for different experiments

## Usage

For detailed documentation on the training process, including quickstart guides and technical details, please refer to the [TRAINING.md](../TRAINING.md) document in the project root.

Basic usage:

```bash
# From the project root:
./run_training.sh --config train/configs/default.py
```

See the main [TRAINING.md](../TRAINING.md) for more options and detailed explanations. 