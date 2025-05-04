"""
Configuration for Ablation 1: Standard training run
Based on the parameters from training_runs.md
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
default_path = os.path.join(current_dir, 'default.py')

# Import the default config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.default import *


# Model and data paths
output_dir = "models/mistral-thinking-abl2"

# Training parameters
num_epochs = 2
batch_size = 16
grad_accumulation_steps = 1
learning_rate = 2e-4
warmup_ratio = 0.03
max_seq_length = 512

# LoRA parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Wandb
use_wandb = True
wandb_run_id = None
wandb_project = "mistral_thinking_nli"
wandb_name = "ablation2"

# Other settings
use_packing = False
gradient_checkpointing = False
resume_from_checkpoint = None