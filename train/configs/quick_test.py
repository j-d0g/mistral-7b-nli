"""
Quick Test Configuration

A minimal configuration for rapidly verifying that the training pipeline works.
This uses a small amount of data, minimal iterations, and reduced model parameters
to complete very quickly (typically under a minute).

Use this for:
- Verifying Docker setup is working
- Testing code changes to the training pipeline
- Debugging training infrastructure issues

For more substantial testing with sample data but quicker than full training,
set the sample_data parameter to True.
"""

# Import base references but don't inherit (to prevent picking up unintended parameters)
import sys
import os

# Model and data paths - will use sample data if sample_data=True
model_id = "mistralai/Mistral-7B-v0.3"
sample_data = False  # Toggle this to True to use sample data instead of minimal data
if sample_data:
    train_data = "data/finetune/sample_ft.jsonl"
    eval_data = "data/finetune/sample_ft.jsonl"
    output_dir = "models/nlistral-sample-test"
else:
    # Ultra-minimal data - just use the first 10 examples
    train_data = "data/finetune/train_ft.jsonl"
    eval_data = "data/finetune/dev_ft.jsonl"
    output_dir = "models/nlistral-quick-test"
seed = 42

# LoRA parameters - reduced for speed
lora_r = 8  # Reduced rank for faster training
lora_alpha = 16
lora_dropout = 0.05

# Training parameters - absolute minimal for a quick test
num_epochs = 1
max_seq_length = 512
batch_size = 4
grad_accumulation_steps = 1
learning_rate = 2e-4
lr_scheduler = "constant"  # Simplified scheduler for testing
warmup_ratio = 0.0
weight_decay = 0.01
max_grad_norm = 1.0  # Enable gradient clipping for stability

# Evaluation and logging - very frequent for testing
logging_steps = 5
eval_steps = 10
save_steps = 10
save_total_limit = 1

# Minimal data handling - keep only a few examples
max_train_samples = 10 if not sample_data else 100  # Use 10 examples for minimal, 100 for sample
max_eval_samples = 10 if not sample_data else 50    # Use 10 examples for minimal, 50 for sample

# Wandb
wandb_project = "nlistral-quick-test"
wandb_run_name = "quick_test"
use_wandb = False  # Disable wandb for quick testing

# Other settings
use_packing = False
gradient_checkpointing = False  # Speed over memory efficiency for testing
resume_from_checkpoint = None
gpu_id = 0

# Distributed training settings
distributed_training = False 