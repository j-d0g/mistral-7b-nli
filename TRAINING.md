# Training the Mistral-7B NLI Model

This document provides instructions for training the Mistral-7B NLI model using QLoRA fine-tuning, with both quick start guides and in-depth technical explanations.

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Training Process Overview](#training-process-overview)
  - [Running Training Commands](#running-training-commands)
  - [Training with Sample Data](#training-with-sample-data)
  - [Common Issues](#common-issues)
- [Deep Dive: Technical Details](#deep-dive-technical-details)
  - [Training Philosophy & Setup](#training-philosophy--setup)
  - [Configuration System](#configuration-system) 
  - [Training Script Details](#training-script-details)
  - [Fine-Tuning Strategy & Hyperparameters](#fine-tuning-strategy--hyperparameters)
  - [Advanced Troubleshooting](#advanced-troubleshooting)

---

# Quick Start

## Prerequisites

Before you begin training, ensure you have:

1. **Docker installed** with NVIDIA Container Toolkit (for GPU support)
2. **Downloaded the datasets** using instructions in [DATA.md](DATA.md)
3. **Built the Docker image**:
   ```bash
   docker build -t mistral-nli-ft .
   ```
4. **Created a Hugging Face token** (store in `.env` file as `HF_TOKEN=your_token_here`)

## Training Process Overview

The training workflow consists of these steps:

1. **Choose a configuration** - Select from predefined configs or create your own
2. **Run the training** - Execute using the `run_training.sh` script
3. **Monitor progress** - Track metrics via Weights & Biases or command line output
4. **Evaluate the model** - Run inference to check performance

## Running Training Commands

Training is executed through the `run_training.sh` script in the repository root:

```bash
# Basic usage with default configuration
./run_training.sh

# Using a specific configuration
./run_training.sh --config train/configs/ablation0.py

# Specifying GPU to use
./run_training.sh --gpus 1 --config train/configs/default.py

# Multi-GPU training with specific GPUs
./run_training.sh --gpus 0,1 --config train/configs/distributed.py

# Multi-GPU training with all available GPUs
./run_training.sh --all-gpus --config train/configs/distributed.py
```

### GPU Options

- `--gpus N` - Use specific GPU number N (default is 0)
- `--gpus 0,1,2` - Use specific GPUs with distributed training
- `--all-gpus` - Use all available GPUs with distributed training

## Training with Sample Data

For quick tests or debugging, you can use sample data. We've included a ready-to-use configuration for this purpose:

```bash
# Run with our pre-configured sample test configuration
./run_training.sh --config train/configs/sample_test.py
```

This configuration:
- Uses the small sample dataset (`data/finetune/sample_ft.jsonl`)
- Sets minimal training parameters (1 epoch, small batch size)
- Disables Weights & Biases logging
- Uses reduced LoRA parameters to minimize memory usage

If you want to create your own test configuration:

```bash
# First create a custom configuration based on the sample
cp train/configs/sample_test.py train/configs/my_test.py

# Edit my_test.py to customize parameters

# Run with your test configuration
./run_training.sh --config train/configs/my_test.py
```

## Monitoring Training

### Weights & Biases Integration

Training metrics are logged to Weights & Biases by default. To view metrics:

1. Create a Weights & Biases account if you don't have one
2. Sign in through the link provided in the console output
3. View metrics through the web dashboard

To disable Weights & Biases logging:
```bash
# Add no_wandb option when running training
./run_training.sh --config train/configs/default.py --no_wandb
```

### Command-Line Monitoring

You can also monitor training through the command-line output, which shows:
- Loss values
- Learning rate progression
- Token accuracy
- Evaluation metrics

## Common Issues

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce batch size (e.g., `batch_size = 4` or even `1`)
2. Reduce sequence length (e.g., `max_seq_length = 256`)
3. Use gradient checkpointing (enabled by default in configurations)
4. Try a smaller model or further quantization

### Training Crashes

If training crashes unexpectedly:

1. Check your Docker GPU setup (`nvidia-smi` should work inside container)
2. Verify dataset paths and formats
3. Try the `minimal_test.py` configuration which uses minimal resources
4. Check for disk space limitations (HF cache and model outputs can be large)

### Resuming Training

To resume from a checkpoint after interruption:

```bash
./run_training.sh --config train/configs/default.py --resume_from_checkpoint models/mistral-thinking-default/checkpoint-XXXX
```

Or to resume from the latest checkpoint:

```bash
./run_training.sh --config train/configs/default.py --resume_from_checkpoint latest
```

## After Training

Once training completes, the model will be saved to the directory specified in your configuration's `output_dir` parameter.

### Evaluating Your Trained Model

To evaluate your trained model on a test dataset, see [EVALUATION.md](EVALUATION.md) for complete instructions:

```bash
# Evaluate the model on the test dataset
./evaluate/run_inference.sh --model models/mistral-thinking-sample-test --data data/original_data/test.csv
```

The evaluation script will generate detailed output files in the `results/` directory, including accuracy metrics and per-example predictions with thought processes.

## Further Information

- **Data preparation**: See [DATA.md](DATA.md)
- **Evaluation**: See [EVALUATION.md](EVALUATION.md)

---

# Deep Dive: Technical Details

This section provides in-depth technical information about the training system for researchers and developers who need to understand the implementation details.

## Training Philosophy & Setup

### Goal

The objective is to instruction-tune the base Mistral-7B model using the prepared CoT dataset (`data/finetune/*.jsonl`). The model should learn to take a premise and hypothesis and generate a JSON output containing both a step-by-step `thought_process` and the final `predicted_label` (0 for no-entailment, 1 for entailment). This approach aims for both high accuracy and interpretability.

The selection of `mistralai/Mistral-7B-v0.3` as the base model was driven by several factors: its reputation for strong reasoning performance within the 7B parameter range, favorable cost and inference speed metrics, demonstrated high accuracy and precision on preliminary NLI task evaluations, and the existence of an accessible API ecosystem (e.g., `open-mistral-7b`, `open-mistral-nemo`) that proved crucial for the multi-stage Chain-of-Thought data generation and reflection pipeline.

### Parameter-Efficient Fine-Tuning (PEFT) with QLoRA

Given the size of Mistral-7B (7 billion parameters), full fine-tuning requires substantial GPU memory. We employ QLoRA to make this feasible on more accessible hardware (e.g., single or dual consumer GPUs like RTX 3090/4090).

*   **Quantization:** The base model weights are loaded in 4-bit precision using the NF4 ("NormalFloat 4") data type via the `bitsandbytes` library. Double quantization is often enabled for further memory savings. This drastically reduces the memory needed to hold the base model weights.
*   **LoRA (Low-Rank Adaptation):** Instead of updating all model weights, small, trainable "adapter" matrices are injected into specific layers of the frozen, quantized base model. Typically, these are applied to the attention mechanism's query (`q_proj`), key (`k_proj`), value (`v_proj`), and output (`o_proj`) linear layers. Only these adapter weights (a small fraction of the total parameters) are trained.
*   **Compute Precision (`bfloat16`):** While weights are stored in 4-bit, computations (like matrix multiplications during forward and backward passes) are performed using a higher precision format. `bfloat16` (Brain Floating Point) is preferred over `float16` for training stability, especially with newer GPU architectures, although it requires careful handling to avoid dtype mismatches.

### Docker Environment

Consistency across different machines (local workstations, cloud GPUs, clusters) is critical, especially with complex dependencies involving specific CUDA versions, PyTorch, and PEFT libraries. Docker is used to encapsulate the entire training environment.

*   **`Dockerfile`:** Defines the environment, starting from a PyTorch base image with CUDA 12.1, installing system dependencies (like `git`, `build-essential`), copying the code, and installing Python packages via `requirements.txt`.
*   **`requirements.txt`:** Lists key Python dependencies:
    *   `torch` (pinned to CUDA 12.1 compatible version)
    *   `transformers`, `peft`, `trl`, `datasets`, `accelerate` (core Hugging Face ecosystem)
    *   `bitsandbytes` (for QLoRA quantization)
    *   `sentencepiece`, `protobuf` (tokenizer dependencies)
    *   `flash-attn` (optional, for Flash Attention 2 optimization)
    *   `wandb` (optional, for logging)

## Configuration System

A flexible configuration system manages hyperparameters and experiment settings.

### Overview & Rationale

Managing numerous experiments with varying hyperparameters (learning rate, batch size, LoRA rank, etc.) purely through command-line arguments or complex bash scripts becomes unwieldy and difficult to reproduce. This Python-based config system addresses these issues:

*   **Readability & Maintainability:** Python files are more readable than long bash commands. Comments explain parameter choices.
*   **Structure & Defaults:** A `default.py` establishes base parameters, while specific experiment configs (e.g., `ablation2.py`) inherit from it and override only necessary values.
*   **Self-Documentation:** Each config file serves as a record of a specific experimental setup.
*   **Flexibility:** Allows easy definition of complex settings while still permitting quick command-line overrides for minor tweaks.
*   **Type Safety:** Python handles parameter types (int, float, bool) more robustly than bash.

### Components & Usage

*   **`train/configs/default.py`:** Base configuration. Contains default values for most parameters:

```python
# Model and data paths
model_id = "mistralai/Mistral-7B-v0.3"
train_data = "data/finetune/train_ft.jsonl"
eval_data = "data/finetune/dev_ft.jsonl"
output_dir = "models/mistral-thinking-default"
seed = 42

# LoRA parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Training parameters
num_epochs = 2
max_seq_length = 512
batch_size = 16
grad_accumulation_steps = 2
learning_rate = 2e-4
lr_scheduler = "cosine"
warmup_ratio = 0.03
weight_decay = 0.01
max_grad_norm = None  # No gradient clipping
```

*   **`train/configs/*.py`:** Specific experiment configurations (e.g., `ablation0.py`, `ablation1.py`, `ablation2.py`, `ablation3.py`, `minimal_test.py`, `distributed.py`). They typically start with `from configs.default import *` and then redefine specific variables.
*   **`train/config_loader.py`:** A utility script that loads the default config, then loads the specified experiment config (if any), and finally applies any overrides passed via command-line arguments.
*   **`train/train_sft.py`:** Imports and uses `config_loader.py` at the start to get the final configuration dictionary.
*   **Activation:** The `--config path/to/config.py` argument is passed to `run_training.sh`, which forwards it to `train_sft.py`. CLI args like `--batch_size 8` or `--no_wandb` are also parsed and take final precedence.

## Training Script Details

The core script `train/train_sft.py` performs the Supervised Fine-Tuning (SFT) using Hugging Face's `transformers` and `trl` libraries.

### Key Implementation Steps

1.  **Load Configuration:** Uses `config_loader.py` to merge defaults, specific config, and CLI overrides into a single config object.
2.  **Set Seed:** `transformers.set_seed(config.seed)` for reproducibility.
3.  **Load Datasets:** Loads `train` and `eval` splits from JSONL files specified in the config using `datasets.load_dataset("json", data_files=...)`.
4.  **Load Tokenizer:** `AutoTokenizer.from_pretrained(config.model_id)`. Sets `padding_side='right'` and ensures `tokenizer.pad_token` is set.
5.  **Define Quantization Config:** Creates `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`.
6.  **Load Base Model:** `AutoModelForCausalLM.from_pretrained(config.model_id, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)`.
7.  **Prepare for K-bit Training:** `peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)`.
8.  **Define LoRA Config:** Creates `LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=config.target_modules, bias="none", task_type="CAUSAL_LM")`.
9.  **Define Training Arguments:** Creates `SFTConfig` (a subclass of `TrainingArguments`) passing all hyperparameters from the config object.
10. **Initialize SFTTrainer:** `SFTTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['eval'], peft_config=peft_config, dataset_text_field="text")`.
11. **Handle BF16 Dtype Issues:** May include a monkey-patch for the model's `forward` method using `torch.cuda.amp.autocast(dtype=torch.bfloat16)` to resolve dtype conflicts that can arise with `bfloat16` and gradient checkpointing.
12. **Train:** `trainer.train()` starts the fine-tuning loop.
13. **Save Model:** `trainer.save_model()` saves the trained LoRA adapter weights (not the full model) to the `output_dir`.

## Fine-Tuning Strategy & Hyperparameters

### QLoRA Configuration Rationale

The choice of QLoRA parameters involves balancing model expressiveness, training stability, and computational resources.

*   **Target Modules:** Targeting the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) is standard practice for LoRA as these layers are crucial for learning task-specific patterns.
*   **Rank (`r`) and Alpha (`lora_alpha`):**
    *   `r` determines the rank (size) of the adapter matrices. Higher `r` allows the adapter to capture more complex patterns but increases trainable parameters.
    *   `lora_alpha` acts as a scaling factor for the adapter weights. A common heuristic is `lora_alpha = 2 * r`.
    *   **Experimental Findings:** The final configuration adopted `r=32`, `lora_alpha=64`. This higher rank was chosen to provide the necessary capacity to model the diverse reasoning paths, especially for the reflected thought examples.
*   **Dropout (`lora_dropout`):** A small dropout (e.g., 0.05) is applied to the LoRA layers for regularization.

### Key Training Hyperparameters

These parameters control the optimization process:

*   **`num_epochs` (e.g., 3):** How many times to iterate over the training dataset. Using validation loss (`eval_loss`) with early stopping is recommended.
*   **Batch Size (`per_device_train_batch_size`, `gradient_accumulation_steps`):**
    *   `per_device_train_batch_size` is limited by GPU memory (e.g., 2-4 for a 7B model on a 24GB GPU with QLoRA).
    *   `gradient_accumulation_steps` allows simulating a larger effective batch size. An effective batch size of ~32-64 is common.
*   **Learning Rate (`learning_rate`, `lr_scheduler_type`, `warmup_ratio`):**
    *   `learning_rate` (e.g., `2e-4`) is often slightly higher for PEFT than full fine-tuning. Values between `1e-4` and `3e-4` were explored.
    *   A scheduler (`linear` or `cosine`) decays the learning rate over time. `cosine` is often preferred.
    *   `warmup_ratio` (e.g., `0.03`) gradually increases the LR at the start of training for stability.
*   **Optimizer (`optim`, `weight_decay`):**
    *   `paged_adamw_8bit` is a memory-efficient AdamW variant suitable for QLoRA.
    *   `weight_decay` (e.g., `0.01`) provides L2 regularization.
*   **Sequence Length (`max_seq_length`):** Set to `512` based on analysis showing that shorter thought processes (~250-400 tokens total input+output) correlated with higher accuracy, while dramatically reducing memory usage and computation time compared to default lengths (e.g., 2048 or 4096).

## Advanced Troubleshooting

### BF16 Dtype Mismatch

This often occurs when using `bfloat16` precision (`bf16=True`) together with `gradient_checkpointing=True`. Parts of the model might remain in `float32` while others expect `bfloat16`.

Solution: Implement the `autocast` wrapper around the model's forward pass. This explicitly tells PyTorch to execute the forward pass within a `bfloat16` context:

```python
# In train_sft.py, before initializing SFTTrainer
original_forward = model.forward
def forward_with_autocast(*args, **kwargs):
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        return original_forward(*args, **kwargs)
model.forward = forward_with_autocast
```

### Poor Convergence / High Loss

Possible causes and solutions:
*   **Hyperparameters:** Adjust learning rate (try lower: `1e-4` or higher: `3e-4`), change LR scheduler, experiment with `weight_decay`.
*   **LoRA:** Try increasing LoRA rank/alpha (e.g., `r=32`, `alpha=64`).
*   **Data:** Verify the quality and format of the `data/finetune/*.jsonl` files. Ensure correct instruction formatting. Check for data imbalances.
*   **Epochs:** Ensure enough training epochs (`num_epochs`) are run, monitoring `eval_loss` closely. Use early stopping.

### Library Compatibility Issues

Version mismatches between `transformers`, `peft`, `trl`, `torch`, `accelerate`, `bitsandbytes` can cause issues.

Solutions:
*   Strictly adhere to the versions in `requirements.txt`.
*   Rebuild the Docker image after any changes to `requirements.txt`.
*   Be aware of breaking changes in library APIs.

## Training Ablations

The project includes several training ablations to explore different data strategies:

1. **Ablation 0**: Train on correct examples + corrected versions of incorrect examples from the strong model.
2. **Ablation 1**: Train only on examples where the original model's prediction was correct.
3. **Ablation 2**: Train on examples where the original model's prediction was correct + reflected examples for incorrect predictions.
3. **Ablation 3**: Train on all original examples, regardless of correctness.

The ablation configurations are available in `train/configs/` directory and can be specified with the `--config` parameter. 