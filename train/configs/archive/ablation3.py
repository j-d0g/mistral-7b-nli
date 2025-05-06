"""
Ablation 0 (1 Epoch): Small Batch Size Configuration

This configuration tests frequent stochastic updates with a small effective batch size (16).
It represents the original ablation0 with a single epoch training duration.

Key characteristics:
- Effective batch size of 16 (batch_size=8, grad_accumulation_steps=2)
- LoRA rank 16, alpha 32
- No gradient checkpointing
- Single epoch training
"""

# Model and data paths
model_id = "mistralai/Mistral-7B-v0.3"
train_data = "data/finetune/train_ft.jsonl"
eval_data = "data/finetune/dev_ft.jsonl"
output_dir = "models/mistral-thinking-ablation0-1epoch"
seed = 42

# LoRA parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Training parameters
num_epochs = 1
max_seq_length = 512
batch_size = 8
grad_accumulation_steps = 2  # Effective batch size: 16
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
wandb_project = "mistral_thinking_nli"
wandb_run_name = "ablation0_1epoch"
wandb_run_id = None

# Other settings
use_packing = False
gradient_checkpointing = False
use_wandb = True
resume_from_checkpoint = None
gpu_id = 0

# Distributed training settings
distributed_training = False 