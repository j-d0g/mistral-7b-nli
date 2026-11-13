"""
Ablation 0 Best: Small Batch, Two Epochs

Same as Ablation 0 (effective batch size 16, LoRA rank 16/alpha 32, no
gradient checkpointing) but trained for 2 epochs instead of 1.

Note: this file was empty in the repository from May 2025 (commit 78153f2)
until the documentation clean-up. Its values are the ones documented in
train/configs/README.md and MODEL_CARD.md, and match the training log of the
run behind the published ablation 0 result (`mistral-thinking-abl0-ext`:
batch size 8, 2 epochs, peak learning rate 2e-4).
"""

# Import the base ablation0 configuration
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation0 import *

# Two epochs instead of one
num_epochs = 2

# Output directory
output_dir = "models/nlistral-ablation0-best"

# Wandb
wandb_run_name = "ablation0_best"
