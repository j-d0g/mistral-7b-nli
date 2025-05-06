# Training Configurations

This directory contains configuration files for fine-tuning the Mistral-7B model on NLI data with Chain-of-Thought reasoning.

## Configuration Structure

The configurations follow three main experimental tracks, each with a base version and an optimized "best" version:

### Track 1: Small Batch Training (Ablation 0)

- **`ablation0.py`**: Tests frequent stochastic updates with small effective batch size (16) for 1 epoch.
- **`ablation0_best.py`**: Optimized version that extends training to 2 epochs for better results.

### Track 2: Medium Batch with Warmup Variations (Ablation 1)

- **`ablation1.py`**: Tests medium batch size (32) with higher warmup ratio (0.05) and no gradient checkpointing.
- **`ablation1_best.py`**: Optimized version that adds gradient checkpointing and uses a lower warmup ratio (0.03). This is the overall best configuration.

### Track 3: Large Model Capacity (Ablation 2)

- **`ablation2.py`**: Tests increased model capacity (rank 32/alpha 64) with larger batch size (64) and stability measures including gradient clipping, higher warmup ratio, and lower learning rate.
- **`ablation2_best.py`**: Further optimized version with even lower learning rate (5e-5), extended training (5 epochs), and modified warmup ratio.

### Special Configurations

- **`distributed.py`**: Adapts the best configuration for multi-GPU distributed training.
- **`quick_test.py`**: A consolidated configuration for quick testing with minimal resources.

## Usage

Use these configurations with the `run_training.sh` script:

```bash
# Run the best overall configuration
./run_training.sh --config train/configs/ablation1_best.py

# Run a specific ablation
./run_training.sh --config train/configs/ablation2_best.py

# Run distributed training on multiple GPUs
./run_training.sh --gpus 0,1 --config train/configs/distributed.py
```

You can also override specific parameters:

```bash
./run_training.sh --config train/configs/ablation1_best.py --learning_rate 1e-4 --batch_size 32
```

## Epochs and Learning Rate Dynamics

An important insight discovered during experimentation is that changing the number of epochs fundamentally transforms the entire learning process because:

1. When using `warmup_ratio`, increasing epochs reduces the relative portion of training spent warming up
2. Learning rate decay (e.g., cosine) is stretched or compressed by changing epoch count
3. Adding epochs to a previously successful configuration can destabilize training by altering the warmup/decay balance

For more details on training evolution and parameter rationale, see the [BLOG.md](../../BLOG.md) document. 