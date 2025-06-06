"""
Default configuration for Mistral-7B NLI fine-tuning.
"""

# Model and data paths
model_id = "mistralai/Mistral-7B-v0.3"
train_data = "data/finetune/train_ft.jsonl"
eval_data = "data/finetune/dev_ft.jsonl"
output_dir = "models/mistral-thinking-default-epochs2"
seed = 42

# LoRA parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Training parameters
num_epochs = 5
max_seq_length = 512
batch_size = 16
grad_accumulation_steps = 2
learning_rate = 2e-4
lr_scheduler = "cosine"
warmup_ratio = 0.03
weight_decay = 0.01
max_grad_norm = None  # No gradient clipping

# Evaluation and logging
logging_steps = 25
eval_steps = 250
save_steps = 250
save_total_limit = 2

# Wandb
wandb_project = "mistral_thinking_nli"  # Default project name
wandb_run_name = "mistral-thinking-default"  # Will be generated if not specified
wandb_run_id = "knyrgpal"

# Other settings
use_packing = False
gradient_checkpointing = True
use_wandb = True
resume_from_checkpoint = "models/mistral-thinking-default-epochs2/checkpoint-2224"
gpu_id = 0 

# Distributed training settings
distributed_training = False 