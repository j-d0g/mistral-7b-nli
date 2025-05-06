"""
Ablation 1: Small Effective Batch Size Configuration

This configuration tests the hypothesis that frequent stochastic updates
with a smaller effective batch size (16) might help escape local minima.

Key characteristics:
- Effective batch size of 16 (batch_size=8, grad_accumulation_steps=2)
- LoRA rank 16, alpha 32
- No gradient checkpointing
- Single epoch training
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the best config as base
sys.path.insert(0, os.path.dirname(current_dir))
from configs.best import *

# Training parameters - reduced batch size
batch_size = 8
grad_accumulation_steps = 2  # Effective batch size: 16
num_epochs = 1
learning_rate = 2e-4
warmup_ratio = 0.03

# Output directory
output_dir = "models/mistral-thinking-ablation1-small-batch"

# Disable gradient checkpointing (testing without memory optimization)
gradient_checkpointing = False

# Wandb
wandb_run_name = "ablation1_small_batch" 