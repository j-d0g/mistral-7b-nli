"""
Configuration for Ablation 0: Longer Epochs on Merged Training Data
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the default config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.default import *

# Model and data paths
output_dir = "models/mistral-thinking-default-final"
data_dir = "data/finetune/train_ft_final.jsonl" # Train on final dataset that merges train and test set we allocated.

num_epochs = 5

# Wandb
use_wandb = True
wandb_project = "mistral_thinking_nli"
wandb_name = "default-final"