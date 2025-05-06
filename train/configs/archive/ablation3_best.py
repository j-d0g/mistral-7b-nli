"""
Ablation 0 Best: Tuned Small Batch Configuration

This configuration represents the tuned version of Ablation 0, extending the
training duration to achieve better results while maintaining the small
effective batch size approach.

Key characteristics:
- Same as Ablation 0 (effective batch size 16, LoRA rank 16/alpha 32)
- Extended to 2 epochs for more thorough training
- Maintains the careful warmup ratio of 0.03
"""

# Import the base ablation0 configuration
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation0 import *

# Main change: Extended training duration
num_epochs = 2

# Output directory
output_dir = "models/mistral-thinking-ablation0-best"

# Wandb
wandb_run_name = "ablation0_best" 