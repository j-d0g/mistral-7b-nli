#!/usr/bin/env python3
"""
Training script for Mistral-7B NLI with Chain-of-Thought.
Supports both direct command-line arguments and config files.
"""

import logging
import os
import sys
import torch
import torch.nn as nn
import argparse
from torch.cuda.amp import autocast
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments with support for config files."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral-7B for NLI with Chain-of-Thought",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file option (with default)
    parser.add_argument(
        "--config", type=str, default="train/configs/default.py",
        help="Path to Python configuration file"
    )
    
    # Model and data paths
    parser.add_argument("--model_id", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--train_data", type=str, help="Path to training data (JSONL format)")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data (JSONL format)")
    parser.add_argument("--output_dir", type=str, help="Directory to save the fine-tuned model")
    
    # Training parameters
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm for clipping")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout")
    
    # Evaluation and logging
    parser.add_argument("--logging_steps", type=int, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, help="Save steps")
    parser.add_argument("--save_total_limit", type=int, help="Save total limit")
    
    # Other settings
    parser.add_argument("--use_packing", action="store_true", help="Enable packing")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from, or 'latest'")
    parser.add_argument("--wandb_run_id", type=str, help="Wandb run ID to resume")
    
    # Distributed training parameters
    parser.add_argument("--distributed_training", type=bool, default=False, help="Enable distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    return parser.parse_args()

def load_and_merge_config(args):
    """
    Load configuration from file if specified and merge with CLI arguments.
    CLI arguments take precedence over config file values.
    """
    # Start with all CLI arguments
    config = {k: v for k, v in vars(args).items() if v is not None}
    
    # Import config_loader
    import os.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_loader_path = os.path.join(current_dir, 'config_loader.py')
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_loader", config_loader_path)
    config_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_loader)
    
    # If config file specified and different from default, load it
    if args.config:
        try:
            logger.info(f"Loading configuration from: {args.config}")
            file_config = config_loader.load_config(args.config)
            
            # Add config file values for keys not explicitly set via CLI
            for key, value in file_config.items():
                if key not in config or config[key] is None:
                    config[key] = value
                    
            logger.info(f"Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            logger.error("Falling back to default configuration")
            
            # Try to load the default config as fallback
            try:
                default_config_path = os.path.join(current_dir, 'configs', 'default.py')
                logger.info(f"Loading default configuration from: {default_config_path}")
                default_config = config_loader.load_config(default_config_path)
                
                # Add default config values for keys not explicitly set via CLI
                for key, value in default_config.items():
                    if key not in config or config[key] is None:
                        config[key] = value
                        
                logger.info(f"Default configuration loaded successfully")
            except Exception as ex:
                logger.error(f"Failed to load default config: {str(ex)}")
                logger.error("Using minimal hardcoded defaults as last resort")
                
                # Absolute minimal defaults as last resort fallback
                minimal_defaults = {
                    'model_id': "mistralai/Mistral-7B-v0.3",
                    'output_dir': "models/nlistral-7b-qlora",
                    'seed': 42,
                    'batch_size': 16,
                    'grad_accumulation_steps': 2
                }
                
                for key, value in minimal_defaults.items():
                    if key not in config:
                        config[key] = value
    else:
        # No config specified, load the default config
        try:
            default_config_path = os.path.join(current_dir, 'configs', 'default.py')
            logger.info(f"Loading default configuration from: {default_config_path}")
            default_config = config_loader.load_config(default_config_path)
            
            # Add default config values for keys not explicitly set via CLI
            for key, value in default_config.items():
                if key not in config or config[key] is None:
                    config[key] = value
                    
            logger.info(f"Default configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load default config: {str(e)}")
            logger.error("Using minimal hardcoded defaults as last resort")
            
            # Absolute minimal defaults as last resort fallback
            minimal_defaults = {
                'model_id': "mistralai/Mistral-7B-v0.3",
                'output_dir': "models/mistral-7b-nli-cot",
                'seed': 42,
                'batch_size': 16,
                'grad_accumulation_steps': 2
            }
            
            for key, value in minimal_defaults.items():
                if key not in config:
                    config[key] = value
    
    # Process special flags
    if 'no_gradient_checkpointing' in config:
        config['gradient_checkpointing'] = not config['no_gradient_checkpointing']
        del config['no_gradient_checkpointing']
        
    if 'no_wandb' in config:
        config['use_wandb'] = not config['no_wandb']
        del config['no_wandb']
    
    return config

def main():
    """Main entry point for training."""
    # Parse command-line arguments and load config
    args = parse_args()
    config = load_and_merge_config(args)
    
    # Distributed training setup
    distributed_training = config.get('distributed_training', False)
    # Use the environment variable set by torchrun
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    if distributed_training:
        if local_rank != -1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            is_master = local_rank == 0
        else:
            is_master = True
    else:
        is_master = True
    
    # Create output directory
    if is_master:
        os.makedirs(config['output_dir'], exist_ok=True)
    
    # Set random seed
    set_seed(config['seed'] + local_rank if local_rank != -1 else config['seed'])
    
    # Log key configuration (minimal) - only on master node
    if is_master:
        logger.info(f"=== Training Configuration ===")
        logger.info(f"Model: {config['model_id']}")
        logger.info(f"Output directory: {config['output_dir']}")
        effective_batch = config['batch_size'] * config['grad_accumulation_steps']
        if distributed_training:
            world_size = dist.get_world_size()
            effective_batch *= world_size
            logger.info(f"Distributed training enabled with {world_size} GPUs")
        
        logger.info(f"Effective batch size: {effective_batch} (batch: {config['batch_size']}, grad_accum: {config['grad_accumulation_steps']})")
        logger.info(f"Training epochs: {config['num_epochs']}")
        logger.info(f"Learning rate: {config['learning_rate']}")
    
    # Set the GPU ID to use if not using distributed training
    if not distributed_training:
        # Log device info based on what PyTorch actually sees
        if torch.cuda.is_available():
            try:
                device_id = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_id)
                if is_master:
                    logger.info(f"PyTorch using GPU: {device_name} (CUDA Device ID {device_id} within container)")
            except Exception as e:
                 if is_master:
                    logger.warning(f"Could not get CUDA device info: {e}")
        elif is_master:
             logger.warning("CUDA not available according to PyTorch.")
    else:
        # When using distributed training, torch.distributed.launch handles GPU assignment
        device_id = local_rank
        device_name = torch.cuda.get_device_name(device_id)
        if is_master:
            logger.info(f"Master process using GPU: {device_name}")
    
    # Load datasets
    if is_master:
        logger.info("Loading datasets...")
    try:
        dataset = load_dataset("json", data_files={"train": config['train_data'], "eval": config['eval_data']})
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return

    # Load tokenizer
    if is_master:
        logger.info(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], trust_remote_code=True)
    
    # Add pad token if needed
    if tokenizer.pad_token is None:
        if is_master:
            logger.info("Adding [PAD] token to tokenizer")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    tokenizer.padding_side = 'right'

    # Preprocess function
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=config['max_seq_length']
        )
        
    # Process datasets
    if is_master:
        logger.info("Preprocessing datasets...")
    try:
        processed_dataset = {
            'train': dataset['train'].map(
                preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=['text']
            ),
            'eval': dataset['eval'].map(
                preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=['text']
            )
        }
    except Exception as e:
        logger.error(f"Failed to preprocess datasets: {e}")
        processed_dataset = dataset
    
    # BitsAndBytes config
    logger.info("Configuring model quantization...")
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    logger.info("Loading quantized model...")
    try:
        # Handle device mapping correctly for distributed training
        if distributed_training:
            device_map = {"": torch.cuda.current_device()}
            logger.info(f"Using device map: {device_map} for GPU {torch.cuda.current_device()}")
        else:
            # For single GPU training
            device_map = {"": 0}
        
        model = AutoModelForCausalLM.from_pretrained(
            config['model_id'],
            quantization_config=bnb_config,
            device_map=device_map,
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
        logger.info(f"Model loaded successfully on {model.device}")

        # Resize token embeddings
        logger.info(f"Resizing token embeddings to match tokenizer")
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model.config.use_cache = False

    # Prepare for k-bit training
    logger.info("Preparing model for training...")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config['gradient_checkpointing'])

    # LoRA config
    logger.info("Setting up LoRA...")
    peft_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Configure wandb if needed
    if config['use_wandb']:
        if config.get('wandb_run_id'):
            os.environ["WANDB_RESUME"] = "allow"
            os.environ["WANDB_RUN_ID"] = config['wandb_run_id']
        
        # Set WandB project and run name if provided in config
        if config.get('wandb_project'):
            os.environ["WANDB_PROJECT"] = config['wandb_project']
        
        if config.get('wandb_name') or config.get('wandb_run_name'):
            # Support both wandb_name and wandb_run_name for compatibility
            run_name = config.get('wandb_name') or config.get('wandb_run_name')
            os.environ["WANDB_NAME"] = run_name
    
    # Training args
    logger.info("Configuring training parameters...")
    training_args = SFTConfig(
        output_dir=config['output_dir'],
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'], 
        gradient_accumulation_steps=config['grad_accumulation_steps'],
        optim="paged_adamw_8bit",
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config.get('lr_scheduler', 'cosine'),
        warmup_ratio=config['warmup_ratio'],
        weight_decay=config['weight_decay'],
        max_grad_norm=config['max_grad_norm'],
        max_seq_length=config['max_seq_length'],
        packing=config.get('use_packing', False),
        bf16=True,
        fp16=False,
        logging_steps=config['logging_steps'],
        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        save_strategy="steps",
        save_steps=config['save_steps'],
        save_total_limit=config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if config['use_wandb'] else "none",
        run_name=config.get('wandb_name') or config.get('wandb_run_name') or config['output_dir'],
        seed=config['seed'],
        remove_unused_columns=False,
        resume_from_checkpoint=config.get('resume_from_checkpoint'),
        local_rank=local_rank if distributed_training else -1,
        ddp_find_unused_parameters=False if distributed_training else None,
    )

    # Apply autocast to forward method
    if 'gradient_checkpointing' in config and config['gradient_checkpointing']:
        logger.info("Applying autocast to model forward")
        original_forward = model.forward
        
        def forward_with_autocast(*args, **kwargs):
            with autocast(dtype=compute_dtype):
                return original_forward(*args, **kwargs)
        
        model.forward = forward_with_autocast
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['eval'],
        peft_config=peft_config,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start training
    logger.info("Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=config.get('resume_from_checkpoint'))

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Save final model
        logger.info(f"Saving model...")
        trainer.save_model()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 