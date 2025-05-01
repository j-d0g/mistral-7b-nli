#!/usr/bin/env python3
import logging
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-v0.3"
TRAIN_DATA = "data/finetune/train_ft.jsonl"
EVAL_DATA = "data/finetune/dev_ft.jsonl"
OUTPUT_DIR = "models/mistral-7b-nli-cot"
SEED = 42

# Optimized QLoRA Config for 4090 GPUs
LORA_R = 16             # LoRA attention dimension
LORA_ALPHA = 32         # LoRA alpha parameter
LORA_DROPOUT = 0.05     # Dropout probability for LoRA layers
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Target modules for LoRA

# Optimized Training Config for 9-hour constraint
NUM_EPOCHS = 1                  # Single epoch is sufficient for this task
MAX_SEQ_LENGTH = 2048           # Maximum sequence length (adjust based on your data)
BATCH_SIZE_PER_DEVICE = 1       # Reduced to 1 to avoid sequence length mismatches
GRAD_ACCUMULATION_STEPS = 16    # Increased to maintain effective batch size 16
LEARNING_RATE = 2e-4            # Slightly higher LR for faster convergence
LR_SCHEDULER = "cosine"         # Cosine scheduler with warmup
WARMUP_RATIO = 0.03             # Warm up for first 3% of training steps
WEIGHT_DECAY = 0.01             # L2 regularization
OPTIMIZER = "paged_adamw_8bit"  # Memory-efficient optimizer
LOGGING_STEPS = 25              # Log every 25 steps
EVAL_STEPS = 250                # Evaluate every 250 steps
SAVE_STEPS = 250                # Save checkpoint every 250 steps (matching eval)
SAVE_TOTAL_LIMIT = 2            # Keep only 2 checkpoints to save disk space
USE_PACKING = False             # Disable packing for more stable training
GRADIENT_CHECKPOINTING = True   # Enable gradient checkpointing for memory efficiency
USE_WANDB = True                # Enable Weights & Biases logging

def main():
    set_seed(SEED)
    logger.info(f"Starting SFT run with output dir: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Datasets
    logger.info("Loading datasets...")
    try:
        dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "eval": EVAL_DATA})
        logger.info(f"Datasets loaded: {dataset}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # 3. Define QLoRA Config (BitsAndBytes)
    logger.info("Defining BitsAndBytes quantization config...")
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load Base Model
    logger.info(f"Loading base model {MODEL_ID} with QLoRA config...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map={"": 0},  # Force loading onto the first GPU
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        logger.info(f"Model loaded onto device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Check CUDA setup, GPU memory, and model ID.")
        return

    model.config.use_cache = False  # Necessary for gradient checkpointing

    # 5. Prepare model for K-bit training & Gradient Checkpointing
    logger.info("Preparing model for k-bit training with gradient checkpointing...")
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRADIENT_CHECKPOINTING)
        # Verify if gradient checkpointing is enabled
        if hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
            logger.info("Gradient checkpointing successfully enabled via prepare_model_for_kbit_training.")
        elif GRADIENT_CHECKPOINTING:
             logger.warning("prepare_model_for_kbit_training finished, but model.is_gradient_checkpointing is not True. Check configuration.")
        else:
             logger.info("Gradient checkpointing is disabled by configuration.")

    except Exception as e:
        logger.error(f"Error during prepare_model_for_kbit_training: {e}")
        return

    # Ensure lm_head is compatible - crucial for preventing dtype errors
    if hasattr(model, 'lm_head'):
        if model.lm_head.weight.dtype != compute_dtype:
            logger.warning(f"lm_head dtype ({model.lm_head.weight.dtype}) doesn't match compute_dtype ({compute_dtype}). Casting...")
            # More thorough casting of the lm_head
            model.lm_head = model.lm_head.to(compute_dtype)
            # Force cast the weight specifically 
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.to(compute_dtype))
            if model.lm_head.bias is not None:
                model.lm_head.bias = nn.Parameter(model.lm_head.bias.to(compute_dtype))
            logger.info(f"After casting, lm_head dtype is now: {model.lm_head.weight.dtype}")
        else:
             logger.info(f"lm_head dtype ({model.lm_head.weight.dtype}) already matches compute_dtype ({compute_dtype}).")

    # 6. Define PEFT Config (LoRA)
    logger.info("Defining LoRA config...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 7. Define Training Arguments (using SFTConfig)
    logger.info("Defining SFT Training Arguments...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE, 
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=USE_PACKING,
        bf16=True,  # Enable BF16 with autocast
        fp16=False, # Disable FP16
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",  # Evaluate at regular intervals
        eval_steps=EVAL_STEPS,        # Evaluate every N steps
        save_strategy="steps",        # Save at regular intervals
        save_steps=SAVE_STEPS,        # Save every N steps
        save_total_limit=SAVE_TOTAL_LIMIT,  # Limit the total number of checkpoints
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_loss",  # Use eval loss to determine best model
        greater_is_better=False,      # Lower loss is better
        report_to="wandb" if USE_WANDB else "none",
        seed=SEED,
        remove_unused_columns=False,  # Don't remove unused columns to prevent errors
    )

    # 8. Initialize Trainer with autocast for BF16 precision
    logger.info("Initializing SFTTrainer with autocast for BF16 precision...")
    
    # Monkey patch the forward method of the model to use autocast
    # This is to solve the dtype mismatch issue with bfloat16
    logger.info("Applying autocast patch to model forward method to handle dtype conversions...")
    original_forward = model.forward
    
    def forward_with_autocast(*args, **kwargs):
        with autocast(dtype=compute_dtype):
            return original_forward(*args, **kwargs)
    
    # Replace the forward method with our patched version
    model.forward = forward_with_autocast
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # Using processing_class instead of tokenizer (TRL 0.12.0)
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        peft_config=peft_config,
    )

    # 9. Start Training
    logger.info("Starting training...")
    try:
        train_result = trainer.train()

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # 10. Save Final Model (Adapter)
        logger.info(f"Saving final adapter model to {OUTPUT_DIR}")
        trainer.save_model()

        logger.info("Training finished successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 