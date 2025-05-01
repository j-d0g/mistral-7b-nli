#!/usr/bin/env python3
import logging
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast  # Add autocast import

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
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
OUTPUT_DIR = "models/mistral-7b-nli-cot-simple"
SEED = 42

# QLoRA Config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Config
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 1024 # Smaller sequence length for initial testing
BATCH_SIZE_PER_DEVICE = 1 # Minimal batch size
GRAD_ACCUMULATION_STEPS = 8 # Effective batch size = 1 * 8 * num_gpus = 8 (assuming 1 GPU targeted)
LEARNING_RATE = 5e-5 # Common LR for LoRA
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
OPTIMIZER = "paged_adamw_8bit" # Try 8-bit optimizer for memory saving
LOGGING_STEPS = 10
EVAL_STEPS = 50 # Evaluate reasonably often
SAVE_STEPS = 50 # Save reasonably often
SAVE_TOTAL_LIMIT = 1 # Keep only the last checkpoint
USE_PACKING = False # Disable packing initially
GRADIENT_CHECKPOINTING = True # Essential for memory


def main():
    set_seed(SEED)
    logger.info(f"Starting simplified SFT run with output dir: {OUTPUT_DIR}")
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
    compute_dtype = torch.bfloat16  # Switch back to bfloat16 since we'll use autocast
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
            device_map={"": 0},  # Force loading onto the first available GPU
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None, # Use Flash Attn 2 if available
            torch_dtype=compute_dtype, # Important for consistency
            trust_remote_code=True,
        )
        logger.info(f"Model loaded onto device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Check CUDA setup, GPU memory, and model ID.")
        return

    model.config.use_cache = False # Necessary for gradient checkpointing

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
        per_device_eval_batch_size=BATCH_SIZE_PER_DEVICE, # Keep eval batch size same for simplicity
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_seq_length=MAX_SEQ_LENGTH, # This goes in SFTConfig, not SFTTrainer
        packing=USE_PACKING, # This goes in SFTConfig, not SFTTrainer
        bf16=True, # Enable BF16 again since we're using autocast
        fp16=False, # Disable FP16
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=False, # Keep it simple for now
        report_to="none", # Disable wandb/tensorboard for simplicity
        seed=SEED,
    )

    # 8. Initialize Trainer - Using simplified pattern with SFTConfig
    logger.info("Initializing SFTTrainer with minimal parameters and SFTConfig...")
    
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
        processing_class=tokenizer, # Using processing_class instead of tokenizer (TRL 0.12.0)
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        peft_config=peft_config,
        # Remove other parameters that should be in SFTConfig
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
        trainer.save_model() # Saves the adapter config and weights

        # Optionally save the full model if needed (requires more disk space)
        # logger.info("Merging adapter weights and saving full model...")
        # merged_model = model.merge_and_unload()
        # merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))
        # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_merged_model"))

        logger.info("Training finished successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True) # Log traceback

if __name__ == "__main__":
    main() 