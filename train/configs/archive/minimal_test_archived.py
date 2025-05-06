"""
Minimal test configuration with very small batch size and sequence length
Used to verify GPU configuration
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the default config
sys.path.insert(0, os.path.dirname(current_dir))
from configs.default import *

# Model and data paths
output_dir = "models/mistral-7b-nli-minimal-test"

# Training parameters - very minimal settings to avoid OOM
batch_size = 1 
grad_accumulation_steps = 1
learning_rate = 2e-4
warmup_ratio = 0.03
num_epochs = 1
max_seq_length = 256  # Reduced sequence length
logging_steps = 1
eval_steps = 5
save_steps = 10
save_total_limit = 1

# LoRA parameters - smaller parameters
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

# Wandb
use_wandb = False  # Disable wandb

# Other settings
use_packing = False
gradient_checkpointing = True
resume_from_checkpoint = None