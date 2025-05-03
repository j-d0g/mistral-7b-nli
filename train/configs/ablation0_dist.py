"""
Configuration for Ablation 0: Standard Training
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the default config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.default import *

# Model and data paths
output_dir = "models/mistral-thinking-ablation0-dist"

# Training parameters
batch_size = 16
grad_accumulation_steps = 1  # Effective batch size: 16
learning_rate = 2e-4
warmup_ratio = 0.03
num_epochs = 5

# LoRA parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Wandb
use_wandb = True
wandb_run_id = None
wandb_project = "mistral_thinking_nli"
wandb_name = "ablation0_dist"

# Other settings
use_packing = False
gradient_checkpointing = True
resume_from_checkpoint = None

# Enable distributed training
distributed_training = True