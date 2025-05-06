"""
Ablation 2: Extended Training of Small Batch Configuration

This configuration extends Ablation 1 (small batch size) to train for more epochs,
demonstrating the impact of epoch count on training dynamics including warmup ratio
and learning rate decay effects.

Key characteristics:
- Same as Ablation 1 (effective batch size of 16)
- Extended to 2 epochs (instead of 1)
- Maintains same LoRA parameters (rank 16, alpha 32)
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the small batch config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation1_small_batch import *

# Main change: Extended training duration
num_epochs = 2

# Output directory
output_dir = "models/mistral-thinking-ablation2-extended"

# Wandb
wandb_run_name = "ablation2_extended" 