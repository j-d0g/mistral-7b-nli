#!/usr/bin/env python3
import logging
import os
import torch
import torch.nn as nn
import argparse
from torch.cuda.amp import autocast

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Default configuration (can be overridden by command line args)
DEFAULT_CONFIG = {
    "model_id": "mistralai/Mistral-7B-v0.3",
    "train_data": "data/finetune/train_ft.jsonl",
    "eval_data": "data/finetune/dev_ft.jsonl",
    "output_dir": "models/mistral-7b-nli-cot",
    "seed": 42,
    
    # LoRA parameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Training parameters
    "num_epochs": 3,  # Allow up to 3 epochs
    "max_seq_length": 512,
    "batch_size": 16,
    "grad_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": None,  # No gradient clipping by default
    
    # Evaluation and logging
    "logging_steps": 25,
    "eval_steps": 250,
    "save_steps": 250,
    "save_total_limit": 2,
    
    # Other settings
    "use_packing": False,
    "gradient_checkpointing": True,
    "use_wandb": True,
    "resume_from_checkpoint": None,  # Default to not resuming
    "wandb_run_id": None  # For resuming the same wandb run
}

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B for NLI with Chain-of-Thought")
    
    # Model and data paths
    parser.add_argument("--model_id", type=str, default=DEFAULT_CONFIG["model_id"], help="HuggingFace model ID")
    parser.add_argument("--train_data", type=str, default=DEFAULT_CONFIG["train_data"], help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=DEFAULT_CONFIG["eval_data"], help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], help="Directory to save model")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"], help="Random seed")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=DEFAULT_CONFIG["lora_r"], help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_CONFIG["lora_alpha"], help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_CONFIG["lora_dropout"], help="LoRA dropout")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"], help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"], help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Per-device batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, default=DEFAULT_CONFIG["grad_accumulation_steps"], help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default=DEFAULT_CONFIG["lr_scheduler"], help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_CONFIG["warmup_ratio"], help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_CONFIG["weight_decay"], help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"], help="Maximum gradient norm for clipping")
    
    # Evaluation and logging
    parser.add_argument("--logging_steps", type=int, default=DEFAULT_CONFIG["logging_steps"], help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_CONFIG["eval_steps"], help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"], help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=DEFAULT_CONFIG["save_total_limit"], help="Save total limit")
    
    # Other settings
    parser.add_argument("--use_packing", action="store_true", default=DEFAULT_CONFIG["use_packing"], help="Enable packing")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--resume_from_checkpoint", type=str, default=DEFAULT_CONFIG["resume_from_checkpoint"], help="Path to checkpoint to resume from, or 'latest'")
    parser.add_argument("--wandb_run_id", type=str, default=DEFAULT_CONFIG["wandb_run_id"], help="Wandb run ID to resume")
    
    args = parser.parse_args()
    
    # Handle inverse flags
    args.gradient_checkpointing = not args.no_gradient_checkpointing
    args.use_wandb = not args.no_wandb
    
    return args

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    logger.info(f"Starting SFT run with output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log current CUDA device and GPU info
    device_id = torch.cuda.current_device()
    gpu_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(device_id)
    logger.info(f"Current CUDA device: {device_id}")
    logger.info(f"Available CUDA devices: {gpu_count}")
    logger.info(f"CUDA device name: {device_name}")
    
    # 1. Load Datasets
    logger.info("Loading datasets...")
    try:
        dataset = load_dataset("json", data_files={"train": args.train_data, "eval": args.eval_data})
        logger.info(f"Datasets loaded: {dataset}")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # --- FIX: Add new pad token ---
    if tokenizer.pad_token is None:
        logger.info("Adding new pad token '[PAD]'")
        # Add the token. `special_tokens_map.json` in the save dir will be updated.
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        # Keep the pad_token_id consistent if the tokenizer already assigned one after adding.
        # Otherwise, use the one from the added token.
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        logger.info(f"Set pad_token to {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    else:
         logger.info(f"Using existing pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    # --- END FIX ---

    tokenizer.padding_side = 'right' # Ensure padding side is set AFTER potentially adding the token
    # Note: No longer setting pad_token = eos_token here

    # Helper function for dataset processing to ensure consistent lengths
    def preprocess_function(examples):
        # Don't return tensors from the tokenizer in preprocessing
        return tokenizer(
            examples['text'],
            padding=False,  # We'll handle padding in the data collator
            truncation=True,
            max_length=args.max_seq_length
        )
        
    # Apply preprocessing to datasets
    logger.info("Preprocessing datasets to ensure consistent tensor shapes...")
    try:
        processed_dataset = {
            'train': dataset['train'].map(
                preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=['text']  # Remove the original text
            ),
            'eval': dataset['eval'].map(
                preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=['text']  # Remove the original text
            )
        }
        logger.info(f"Preprocessing complete.")
    except Exception as e:
        logger.error(f"Failed to preprocess datasets: {e}")
        processed_dataset = dataset  # Fallback to original dataset
    
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
    logger.info(f"Loading base model {args.model_id} with QLoRA config...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map={"": 0},  # Use device 0 inside the container
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        logger.info(f"Model loaded onto device: {model.device}")

        # --- FIX: Resize embeddings ---
        logger.info(f"Resizing token embeddings to match tokenizer size ({len(tokenizer)})")
        model.resize_token_embeddings(len(tokenizer))
        # Check if the pad token embedding needs initialization (optional but good practice)
        # if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        #     logger.error("Embedding size mismatch after resize!")
        # Update vocab_size in config
        model.config.vocab_size = len(tokenizer)
        # Ensure the new pad token ID is used if resizing happened
        if hasattr(tokenizer, 'pad_token_id'):
             model.config.pad_token_id = tokenizer.pad_token_id
             logger.info(f"Set model config pad_token_id to: {model.config.pad_token_id}")
        # --- END FIX ---

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Check CUDA setup, GPU memory, and model ID.")
        return

    model.config.use_cache = False  # Necessary for gradient checkpointing

    # 5. Prepare model for K-bit training & Gradient Checkpointing
    logger.info("Preparing model for k-bit training with gradient checkpointing...")
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        # Verify if gradient checkpointing is enabled
        if hasattr(model, 'is_gradient_checkpointing') and model.is_gradient_checkpointing:
            logger.info("Gradient checkpointing successfully enabled via prepare_model_for_kbit_training.")
        elif args.gradient_checkpointing:
             logger.warning("prepare_model_for_kbit_training finished, but model.is_gradient_checkpointing is not True. Check configuration.")
        else:
             logger.info("Gradient checkpointing is disabled by configuration.")

    except Exception as e:
        logger.error(f"Error during prepare_model_for_kbit_training: {e}")
        return

    # Ensure lm_head is compatible - crucial for preventing dtype errors
    if hasattr(model, 'lm_head'):
        if model.lm_head.out_features != len(tokenizer):
            logger.warning(f"lm_head output features ({model.lm_head.out_features}) mismatch tokenizer size ({len(tokenizer)}). Resizing lm_head...")
            model.lm_head = nn.Linear(model.config.hidden_size, len(tokenizer), bias=False)
            # Consider initializing new lm_head weights if needed
            logger.info(f"lm_head resized to output features: {model.lm_head.out_features}")

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
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 7. Define Training Arguments (using SFTConfig)
    logger.info("Defining SFT Training Arguments...")
    
    # Configure wandb resume if needed
    if args.use_wandb and args.wandb_run_id:
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        logger.info(f"Configured to resume wandb run with ID: {args.wandb_run_id}")
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.grad_accumulation_steps,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,  # Add gradient clipping
        max_seq_length=args.max_seq_length,
        packing=args.use_packing,
        bf16=True,  # Enable BF16 with autocast
        fp16=False, # Disable FP16
        logging_steps=args.logging_steps,
        eval_strategy="steps",  # Evaluate at regular intervals
        eval_steps=args.eval_steps,        # Evaluate every N steps
        save_strategy="steps",        # Save at regular intervals
        save_steps=args.save_steps,        # Save every N steps
        save_total_limit=args.save_total_limit,  # Limit the total number of checkpoints
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_loss",  # Use eval loss to determine best model
        greater_is_better=False,      # Lower loss is better
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,
        remove_unused_columns=False,  # Don't remove unused columns to prevent errors
        resume_from_checkpoint=args.resume_from_checkpoint,
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
    
    # Create a data collator that will properly pad sequences
    logger.info("Creating data collator for properly aligning sequences...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )
    
    # Add early stopping callback
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # Using processing_class instead of tokenizer (TRL 0.12.0)
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['eval'],
        peft_config=peft_config,
        data_collator=data_collator,  # Add the data collator
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop after 3 evals with no improvement
    )

    # 9. Start Training
    logger.info("Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # 10. Save Final Model (Adapter)
        logger.info(f"Saving final adapter model to {args.output_dir}")
        trainer.save_model()

        logger.info("Training finished successfully!")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)

if __name__ == "__main__":
    main() 