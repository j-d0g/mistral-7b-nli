"""
Configuration for Ablation 1: Standard training run
Based on the parameters from training_runs.md
"""

# Import and extend defaults
from train.configs.default import *

# Model and data paths
output_dir = "models/mistral-thinking-abl2"

# Training parameters
batch_size = 16
grad_accumulation_steps = 1  # Effective batch size: 32
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
gradient_checkpointing = True
resume_from_checkpoint = None