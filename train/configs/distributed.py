"""
Multi-GPU Distributed Training Configuration

This configuration adapts the best-performing setup (ablation1_best) for distributed training
across multiple GPUs. It maintains the core parameters that gave the best results
while adjusting batch size appropriately for multi-GPU scenarios.

Key characteristics:
- Based on the best configuration (ablation1_best)
- Optimized for multiple GPUs (particularly 2x NVIDIA RTX 4090)
- Per-GPU batch size of 8 for effective total batch size of 16 per GPU
- Gradient checkpointing enabled for memory efficiency
- Same LoRA parameters as the best config (rank 16, alpha 32)
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the best config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation1_best import *

# Model and data paths
output_dir = "models/nlistral-distributed"

# Training parameters - adjusted for multiple GPUs
batch_size = 8  # Per-GPU batch size
num_epochs = 3  # Extended slightly for distributed setup

# Evaluation and logging
eval_steps = 250  # Less frequent evaluation with multiple GPUs

# Wandb
wandb_project = "nlistral-distributed"
wandb_run_name = "distributed"

# Distributed training settings - explicitly enabled
distributed_training = True 