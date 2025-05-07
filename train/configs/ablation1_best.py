"""
Ablation 1 Best: Tuned Medium Batch Configuration

This configuration represents the tuned version of Ablation 1, optimizing the
parameters for better performance while maintaining the medium batch size.
This is the best overall configuration found through extensive experimentation.

Key characteristics:
- Same as Ablation 1 (effective batch size 32, LoRA rank 16/alpha 32)
- Adds gradient checkpointing for memory efficiency and performance
- Uses a carefully tuned lower warmup ratio (0.03) instead of the higher 0.05
"""

# Import the base ablation1 configuration
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))
from configs.ablation1 import *

# Key optimizations
warmup_ratio = 0.03  # Lower, carefully tuned warmup ratio
gradient_checkpointing = True  # Enable gradient checkpointing for performance

# Output directory
output_dir = "models/nlistral-ablation1-best"

# Wandb
wandb_run_name = "ablation1_best" 