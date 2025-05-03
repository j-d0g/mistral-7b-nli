"""
Configuration for Ablation 3: Advanced model
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
output_dir = "models/mistral-thinking-abl3"

# Training parameters
num_epochs = 2
batch_size = 16
grad_accumulation_steps = 4  # Effective batch size: 64
learning_rate = 5e-5
warmup_ratio = 0.05  # 5% warmup
max_seq_length = 512
max_grad_norm = 1.0  # Enable gradient clipping at 1.0

# LoRA parameters
lora_r = 32 # Double rank to handle the increased batch size
lora_alpha = 64 # Double alpha to maintain the same scaling
lora_dropout = 0.05

# Wandb
use_wandb = True
wandb_run_id = None
wandb_project = "mistral_thinking_nli"
wandb_name = "ablation3"

# Other settings
use_packing = False
gradient_checkpointing = False
resume_from_checkpoint = None