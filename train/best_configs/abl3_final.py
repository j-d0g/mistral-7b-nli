"""
Default configuration for Mistral-7B NLI fine-tuning.
"""

# Model and data paths
model_id = "mistralai/Mistral-7B-v0.3"
train_data = "data/finetune/train_ft_final.jsonl"
eval_data = "data/finetune/dev_ft.jsonl"
output_dir = "models/mistral-thinking-abl3-best"
seed = 42

# LoRA parameters
lora_r = 32
lora_alpha = 64
lora_dropout = 0.05

# Training parameters
num_epochs = 5
max_seq_length = 512
batch_size = 16
grad_accumulation_steps = 4
learning_rate = 5e-5
lr_scheduler = "cosine"
warmup_ratio = 0.05
weight_decay = 0.01
max_grad_norm = 1.0  # No gradient clipping

# Evaluation and logging
logging_steps = 25
eval_steps = 250
save_steps = 250
save_total_limit = 2

# Wandb
wandb_project = "mistral_thinking_nli"  # Default project name
wandb_run_name = None  # Will be generated if not specified
wandb_run_id = None

# Other settings
use_packing = False
gradient_checkpointing = True
use_wandb = True
resume_from_checkpoint = None

# Distributed training settings
distributed_training = False 