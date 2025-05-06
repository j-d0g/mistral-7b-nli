"""
Ablation 2 Best: Refined Large Model Configuration

This configuration further refines the ablation2 setup with an even lower
learning rate, extended training duration, and a more moderate warmup ratio.
It represents the most refined version of the large model approach.

Key characteristics:
- Based on ablation2 (large model capacity, stability measures)
- Ultra-low learning rate (5e-5) for very stable training
- Extended to 5 epochs for thorough learning
- Moderate warmup ratio (0.05) balancing stability and training efficiency
- Final dataset version
"""

# Import the ablation2 config as base
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation2 import *

# Further refinements to learning rate and training duration
learning_rate = 5e-5  # Even lower learning rate for maximum stability
num_epochs = 5  # Extended training duration 
warmup_ratio = 0.05  # More moderate warmup ratio

# Output directory
output_dir = "models/mistral-thinking-ablation2-best"

# Specify final dataset version (if applicable)
# If path to final dataset is different, uncomment and modify these:
# train_data = "data/finetune/final_train_ft.jsonl"
# eval_data = "data/finetune/final_dev_ft.jsonl"

# Wandb
wandb_run_name = "ablation2_best" 