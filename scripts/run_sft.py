#!/usr/bin/env python3
import argparse
import logging
import os
import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Supervised Fine-Tuning (SFT) with QLoRA.")

    # Model and Tokenizer arguments
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.3", help="Base model ID.")

    # Dataset arguments
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Path to the training JSONL file.")
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="Path to the evaluation JSONL file.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--packing", action="store_true", default=True, help="Use packing for dataset processing.")

    # QLoRA arguments
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Modules to target for LoRA.")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and final model.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Maximum number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Train batch size per device.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Evaluation batch size per device.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for LR scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing.")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer (e.g., paged_adamw_8bit, paged_adamw_32bit, adamw_torch).")
    parser.add_argument("--bf16", action="store_true", default=torch.cuda.is_bf16_supported(), help="Use bfloat16 precision.")
    parser.add_argument("--fp16", action="store_true", default=not torch.cuda.is_bf16_supported(), help="Use float16 precision if bf16 not available.")
    parser.add_argument("--use_flash_attn_2", action="store_true", default=False, help="Enable Flash Attention 2.")

    # Evaluation and Saving arguments
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy (steps or epoch).")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps (if evaluation_strategy='steps').")
    parser.add_argument("--logging_steps", type=int, default=25, help="Log every N steps.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy (steps or epoch).")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps (if save_strategy='steps').")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True, help="Load the best model checkpoint at the end of training.")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to determine the best model.")
    parser.add_argument("--greater_is_better", action="store_true", default=False, help="Set to True if metric_for_best_model should be maximized.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience for early stopping.")

    # W&B arguments
    parser.add_argument("--report_to", type=str, default="none", help="Report results to (e.g., 'wandb', 'tensorboard', 'none').")
    parser.add_argument("--wandb_project", type=str, default="mistral-nli-sft", help="Weights & Biases project name.")

    args = parser.parse_args()

    if args.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = args.wandb_project

    logger.info(f"Starting SFT with arguments: {args}")

    # 1. Load Datasets
    logger.info("Loading datasets...")
    dataset = load_dataset("json", data_files={"train": args.train_dataset_path, "eval": args.eval_dataset_path})
    logger.info(f"Datasets loaded: {dataset}")

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer for {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = 'right' # Ensure right padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # 3. Define QLoRA Config
    logger.info("Defining BitsAndBytes quantization config...")
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Load Base Model
    logger.info(f"Loading base model {args.model_name_or_path} with QLoRA config...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distribute across GPUs
        attn_implementation="flash_attention_2" if args.use_flash_attn_2 else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False # Disable cache for training
    # Potentially resize token embeddings if pad token was added, although SFTTrainer might handle this

    # 5. Define PEFT Config
    logger.info("Defining LoRA config...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Define Training Arguments
    logger.info("Defining Training Arguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
        push_to_hub=False, # Disable hub pushing by default
        # Add any other necessary arguments
    )

    # 7. Initialize Trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience > 0 else [],
        # Add compute_metrics function here if needed
    )

    # 8. Start Training
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 9. Save Final Model
    logger.info(f"Saving final adapter model to {args.output_dir}")
    trainer.save_model() # Saves the adapter config and weights

    logger.info("Training finished successfully!")


if __name__ == "__main__":
    main() 