# Mistral 7B Fine-Tuning Guide with QLoRA for Chain-of-Thought NLI

This guide provides detailed instructions for fine-tuning Mistral-7B-v0.3 on Chain-of-Thought Natural Language Inference (NLI) tasks using Quantized Low-Rank Adaptation (QLoRA). It includes solutions to common issues, recommended hyperparameters, and a complete workflow from setup to evaluation.

## 1. Setup and Environment

### Docker Setup

We use a Docker-based approach for reproducibility and ease of deployment. The environment is based on PyTorch with CUDA support.

**Dockerfile**:
```dockerfile
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.py
```

**requirements.txt**:
```
# Core ML Libraries
torch --index-url https://download.pytorch.org/whl/cu121
transformers
peft
trl
datasets
accelerate
bitsandbytes
sentencepiece
flash-attn
protobuf

# Optional - For logging (uncomment for WandB)
# wandb
```

### Building and Running the Docker Container

```bash
# Build the Docker image
docker build -t mistral-nli-ft .

# Create cache directory for HuggingFace
mkdir -p hf_cache

# Run the fine-tuning script with default configuration
./train.sh

# Run with a specific configuration file
./train.sh --config train/configs/ablation1.py

# Override specific parameters
./train.sh --config train/configs/ablation1.py --batch_size 8 --learning_rate 1e-4

# Specify GPU to use
./train.sh --gpu_id 1
```

**Note**: Create a `.env` file with your HuggingFace API token and WandB API key if using WandB:
```
HF_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here
```

## 2. Configuration System

The training system uses a Python-based configuration approach:

1. **Default Configuration**: `train/configs/default.py` contains sensible defaults
2. **Custom Configurations**: Create specific config files for different experiments
3. **Command-line Overrides**: CLI arguments always take highest precedence

Example configuration file:
```python
"""
Ablation Study Configuration - Higher learning rate
"""

# Import default configuration values
from train.configs.default import *

# Override specific parameters
output_dir = "models/mistral-7b-nli-cot-ablation1"
batch_size = 8
grad_accumulation_steps = 4  # Effective batch size: 32
learning_rate = 3e-4  # Higher learning rate
num_epochs = 2  # More epochs
```

## 3. Common Issues and Solutions

### Dtype Mismatch Error with BF16 Precision

**Problem**: When using bfloat16 precision with QLoRA and gradient checkpointing, you may encounter the following error:
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

**Solution**: Use the `torch.cuda.amp.autocast()` context manager to handle dtype conversions properly. This can be implemented by monkey-patching the model's forward method:

```python
from torch.cuda.amp import autocast

# Original forward method
original_forward = model.forward

# Create wrapped forward with autocast
def forward_with_autocast(*args, **kwargs):
    with autocast(dtype=torch.bfloat16):
        return original_forward(*args, **kwargs)

# Replace the model's forward method
model.forward = forward_with_autocast
```

### QLoRA Parameter Efficient Fine-Tuning

To enable memory-efficient training of large models on consumer hardware:

1. Use 4-bit quantization with NF4 format
2. Apply gradient checkpointing to reduce memory usage
3. Apply LoRA to specific attention layers (q_proj, k_proj, v_proj, o_proj)
4. Force model loading onto a single GPU with appropriate device mapping

## 4. Training Script With Optimized Parameters

The script below provides an optimized training setup for fine-tuning Mistral-7B on NLI tasks within a 9-hour timeframe:

```python
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
BATCH_SIZE_PER_DEVICE = 2       # Increased from 1 for faster training
GRAD_ACCUMULATION_STEPS = 8     # Effective batch size = 2 * 8 = 16
LEARNING_RATE = 2e-4            # Slightly higher LR for faster convergence
LR_SCHEDULER = "cosine"         # Cosine scheduler with warmup
WARMUP_RATIO = 0.03             # Warm up for first 3% of training steps
WEIGHT_DECAY = 0.01             # L2 regularization
OPTIMIZER = "paged_adamw_8bit"  # Memory-efficient optimizer
LOGGING_STEPS = 25              # Log every 25 steps
EVAL_STEPS = 250                # Evaluate every 250 steps
SAVE_STEPS = 500                # Save checkpoint every 500 steps
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
        eval_strategy="steps",  # Use eval_strategy instead of evaluation_strategy
        eval_steps=EVAL_STEPS,  # Evaluate every N steps
        save_strategy="steps",  # Save at regular intervals 
        save_steps=SAVE_STEPS,  # Save every N steps
        save_total_limit=SAVE_TOTAL_LIMIT,  # Limit the total number of checkpoints
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="eval_loss",  # Use eval loss to determine best model
        greater_is_better=False,  # Lower loss is better
        report_to="wandb" if USE_WANDB else "none",
        seed=SEED,
        remove_unused_columns=False,  # Prevent tensor size mismatch errors
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
```

Save this as `scripts/train_sft.py` and make it executable (`chmod +x scripts/train_sft.py`).

## 4. Hyperparameter Optimization for 9-Hour Training

For a training budget of 9 hours on 2x RTX 4090 GPUs, these hyperparameters provide a good balance:

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| Batch Size | 2 | Small batch size to fit in GPU memory with 7B model |
| Gradient Accumulation | 8 | Creates effective batch size of 16 |
| Learning Rate | 2e-4 | Slightly higher than default for faster convergence |
| LoRA Rank (r) | 16 | Good balance between parameters and expressiveness |
| LoRA Alpha | 32 | 2x the rank for stable training |
| Epochs | 1 | Single epoch should be sufficient for this task |
| Max Sequence Length | 2048 | Accommodates most examples without truncation |
| Precision | BF16 with autocast | Better numerical stability than FP16 |

These parameters will allow for completing the training in approximately 7-8 hours on 2x RTX 4090s, leaving buffer time for potential issues.

## 5. Weights & Biases Integration

### Setup WandB

1. Install WandB:
   - Uncomment `wandb` in `requirements.txt` and rebuild the Docker image, or
   - Run `pip install wandb` in the Docker container

2. Set up WandB credentials using your API key:
   - Add `WANDB_API_KEY=your_api_key_here` to your `.env` file

3. Make sure `USE_WANDB=True` in your training script and `report_to="wandb"` in the SFTConfig

### Metrics to Track

The SFTTrainer will automatically log these metrics to WandB:
- Training loss
- Learning rate
- Training throughput (samples/second)
- GPU memory usage
- Evaluation loss (if evaluation data is provided)

### Custom Metrics (Optional)

If you want to track additional metrics specific to NLI tasks (accuracy, precision, recall), you can implement a custom `compute_metrics` function and pass it to the trainer.

## 6. Config-Based Training System

We've implemented a configuration-based training system for easier experimentation:

### Configuration Files

Our training system uses Python configuration files located in `train/configs/`:

- `default.py`: Base configuration with defaults for all runs
- `initial_test_run.py`: Configuration for the initial test run
- `ablation1.py`: Configuration for Ablation 1 (standard training)
- `ablation2.py`: Configuration for Ablation 2 (mixed data optimization)

Each configuration file extends the default configuration by importing it and overriding specific parameters:

```python
# Import and extend defaults
from train.configs.default import *

# Override specific parameters
output_dir = "models/mistral-7b-nli-cot-ablation1"
batch_size = 16
grad_accumulation_steps = 2  # Effective batch size: 32
learning_rate = 2e-4
warmup_ratio = 0.05  # 5% warmup

# Wandb settings
use_wandb = True
wandb_project = "mistral7b_cot"
wandb_name = "ablation1"

# Hardware selection
gpu_id = 1
```

### Running Training with Configurations

Instead of directly running the Python script, use our wrapper `train.sh` which handles Docker orchestration and config loading:

```bash
# Run with default configuration
./train.sh

# Use a specific configuration file
./train.sh --config train/configs/ablation1.py

# Use ablation2 configuration on GPU 1
./train.sh --config train/configs/ablation2.py --gpu 1

# Override specific parameters
./train.sh --config train/configs/ablation1.py --batch_size 8 --no_wandb
```

### Benefits of the Configuration System

- **Declarative Configuration**: Parameters defined in clean Python files instead of bash scripts
- **Intuitive Overrides**: Command-line parameters take precedence over config values
- **Reduced Duplication**: Default values exist in only one place
- **Self-Documenting Code**: Config files include comments and explanations
- **Simplified Experimentation**: Create new configurations by copying and editing files
- **Better Type Handling**: Proper typing of parameters (integers, floats, booleans)

## 7. Quick Reference: Running the Complete Pipeline

```bash
# Step 1: Build Docker image
docker build -t mistral-nli-ft .

# Step 2: Create directories
mkdir -p models/mistral-7b-nli-cot

# Step 3: Run training with a specific configuration
./train.sh --config train/configs/ablation2.py --gpu 1

# Step 4: Run inference/evaluation
./evaluate/run_inference.sh --model models/mistral-7b-nli-cot-ablation2 --data data/finetune/test_ft.jsonl --gpu 1
```

## 8. Troubleshooting Common Issues

1. **OOM Errors**: If you encounter Out-of-Memory errors:
   - Reduce batch size
   - Reduce sequence length
   - Increase gradient accumulation steps
   - Ensure gradient checkpointing is enabled

2. **Slow Training**: To speed up training:
   - Enable Flash Attention 2 (should be automatic with recent PyTorch)
   - Use packing=True only if your sequences are short
   - Use 8-bit Adam optimizer (paged_adamw_8bit)
   - Consider disabling evaluation during training

3. **Poor Convergence**:
   - Increase learning rate slightly (3e-4 to 5e-4)
   - Increase LoRA rank (r=32, alpha=64)
   - Check for data quality issues

4. **Library Compatibility Issues**:
   - Make sure all libraries are up-to-date
   - Use the `autocast` wrapper for forward pass as demonstrated
   - Consider pinning library versions in requirements.txt
   - With TRL 0.12.0, use `processing_class` instead of `tokenizer` in SFTTrainer
   - Use `eval_strategy` instead of `evaluation_strategy` in SFTConfig for SFTTrainer
   - Add `remove_unused_columns=False` to SFTConfig to prevent tensor size mismatch errors

5. **Tensor Size Mismatch Errors**:
   - When encountering errors like `The size of tensor a (X) must match the size of tensor b (Y)`:
     - Reduce batch size to 1 and increase gradient accumulation steps
     - Set `remove_unused_columns=False` in training arguments
     - Avoid using very different sequence lengths in the same batch
     - Disable packing by setting `packing=False`

Happy training! 