"""
Ablation 2: Large Model Capacity with Stability Measures

This configuration tests increased model capacity and training stability measures
to handle the complexity of the augmented data with potentially contradictory reasoning.

Key characteristics:
- Large effective batch size of 64 (batch_size=16, grad_accumulation_steps=4)
- Doubled LoRA capacity (rank 32, alpha 64)
- Stability measures: gradient clipping at 1.0, increased warmup ratio (0.10)
- Lower learning rate (1e-4) appropriate for larger batch size
- 3 epochs of training
"""

# Model and data paths
model_id = "mistralai/Mistral-7B-v0.3"
train_data = "data/finetune/train_ft.jsonl"
eval_data = "data/finetune/dev_ft.jsonl"
output_dir = "models/nlistral-ablation2"
seed = 42

# LoRA parameters - doubled capacity
lora_r = 32  # Double rank for complex reasoning
lora_alpha = 64  # Double alpha to maintain the same scaling
lora_dropout = 0.05

# Training parameters
num_epochs = 3
max_seq_length = 512
batch_size = 16
grad_accumulation_steps = 4  # Effective batch size: 64
learning_rate = 1e-4  # Lower learning rate for stability
lr_scheduler = "cosine"
warmup_ratio = 0.10  # 10% warmup for better initialization
weight_decay = 0.01
max_grad_norm = 1.0  # Enable gradient clipping for stability

# Evaluation and logging
logging_steps = 25
eval_steps = 250
save_steps = 250
save_total_limit = 2

# Wandb
wandb_project = "nlistral-ablation2"
wandb_run_name = "ablation2"
wandb_run_id = None

# Other settings
use_packing = False
gradient_checkpointing = True  # Memory efficiency
use_wandb = True
resume_from_checkpoint = None
gpu_id = 0

# Distributed training settings
distributed_training = False