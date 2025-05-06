"""
Best Performing Configuration for Mistral-7B NLI Fine-tuning

This configuration represents the optimal balance of parameters found through
extensive experimentation. It's an adjusted version of ablation1 with gradient 
checkpointing enabled and a carefully tuned lower warmup ratio (0.03).

Key characteristics:
- Effective batch size of 32 (batch_size=16, grad_accumulation_steps=2)
- LoRA rank 16, alpha 32
- Gradient checkpointing enabled (critical for performance)
- Careful warmup ratio of 0.03 (lower than ablation1's 0.05)
- 2 epochs of training
"""

# Model and data paths
model_id = "mistralai/Mistral-7B-v0.3"
train_data = "data/finetune/train_ft.jsonl"
eval_data = "data/finetune/dev_ft.jsonl"
output_dir = "models/mistral-thinking-best"
seed = 42

# LoRA parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Training parameters
num_epochs = 2
max_seq_length = 512
batch_size = 16
grad_accumulation_steps = 2  # Effective batch size: 32
learning_rate = 2e-4
lr_scheduler = "cosine"
warmup_ratio = 0.03  # Carefully tuned lower warmup ratio
weight_decay = 0.01
max_grad_norm = None  # No gradient clipping

# Evaluation and logging
logging_steps = 25
eval_steps = 250
save_steps = 250
save_total_limit = 2

# Wandb
wandb_project = "mistral_thinking_nli"
wandb_run_name = "best_config"
wandb_run_id = None

# Other settings
use_packing = False
gradient_checkpointing = True  # Critical performance optimization
use_wandb = True
resume_from_checkpoint = None
gpu_id = 0 

# Distributed training settings
distributed_training = False 