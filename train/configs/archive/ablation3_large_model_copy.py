"""
Ablation 3: Large Model with Stability Measures

This configuration tests the hypothesis that the complex and potentially contradictory 
reasoning in the reflected dataset requires increased model capacity and training stability.

Key characteristics:
- Large effective batch size of 64 (batch_size=16, grad_accumulation_steps=4)
- Doubled LoRA capacity (rank 32, alpha 64)
- Stability measures: gradient clipping, increased warmup ratio
- Lower learning rate (1e-4) appropriate for larger batch size
- Extended training (10 epochs)
"""

# Import with a relative path that works both inside and outside Docker
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the best config as base
sys.path.insert(0, os.path.dirname(current_dir))
from configs.best import *

# Training parameters - larger batch and stability measures
num_epochs = 10
batch_size = 16
grad_accumulation_steps = 4  # Effective batch size: 64
learning_rate = 1e-4  # Lower learning rate for stability
warmup_ratio = 0.10  # 10% warmup for more careful initialization
max_grad_norm = 1.0  # Enable gradient clipping for stability

# LoRA parameters - doubled capacity
lora_r = 32  # Double rank to handle the increased complexity
lora_alpha = 64  # Double alpha to maintain the same scaling

# Output directory
output_dir = "models/mistral-thinking-ablation3-large-model"

# Wandb
wandb_run_name = "ablation3_large_model" 