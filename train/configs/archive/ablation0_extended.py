"""
Configuration for Ablation 0: Longer Epochs
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the default config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation0 import *

# Model and data paths
output_dir = "models/mistral-thinking-abl0-ext"

# Training parameters
batch_size = 16
grad_accumulation_steps = 1  # Effective batch size: 16
learning_rate = 2e-4
num_epochs = 5
warmup_steps = 100

# Wandb
use_wandb = True
wandb_run_id = None
wandb_project = "mistral_thinking_nli"
wandb_name = "ablation0-ext"

# Other settings
use_packing = False
gradient_checkpointing = False
resume_from_checkpoint = None
gpu_id = 0