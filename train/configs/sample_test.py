"""
Sample test configuration for quick testing with small datasets
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the default config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.default import *

# Model and data paths - use sample data
train_data = "data/finetune/sample_ft.jsonl"
eval_data = "data/finetune/sample_ft.jsonl"  # Same file for simplicity
output_dir = "models/mistral-thinking-sample-test"

# Training parameters - minimal settings to run quickly
batch_size = 2
grad_accumulation_steps = 1
num_epochs = 1
max_seq_length = 256  # Reduced sequence length
logging_steps = 5
eval_steps = 10
save_steps = 20
save_total_limit = 1

# LoRA parameters - smaller to reduce memory usage
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

# Wandb - disabled for test runs
use_wandb = False 