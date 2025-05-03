# Fine-Tuning Mistral-7B for Chain-of-Thought NLI

This document provides a comprehensive guide to fine-tuning the Mistral-7B model (specifically `mistralai/Mistral-7B-v0.3`) for the Natural Language Inference (NLI) task using a Chain-of-Thought (CoT) dataset. It covers the setup, configuration, training process, hyperparameter rationale, execution, and troubleshooting.

**Table of Contents**

1.  [Quick Start / How to Run](#1-quick-start--how-to-run)
2.  [Training Philosophy & Setup](#2-training-philosophy--setup)
    *   [2.1 Goal](#21-goal)
    *   [2.2 Parameter-Efficient Fine-Tuning (PEFT) with QLoRA](#22-parameter-efficient-fine-tuning-peft-with-qlora)
    *   [2.3 Docker Environment](#23-docker-environment)
3.  [Configuration System](#3-configuration-system)
    *   [3.1 Overview & Rationale](#31-overview--rationale)
    *   [3.2 Components & Usage](#32-components--usage)
4.  [Training Script (`train/train_sft.py`)](#4-training-script-traintrain_sftpy)
    *   [4.1 Overview](#41-overview)
    *   [4.2 Key Steps & Implementation Details](#42-key-steps--implementation-details)
5.  [Fine-Tuning Strategy & Hyperparameters](#5-fine-tuning-strategy--hyperparameters)
    *   [5.1 QLoRA Configuration Rationale](#51-qlora-configuration-rationale)
    *   [5.2 Key Training Hyperparameters (`SFTConfig`)](#52-key-training-hyperparameters-sftconfig)
    *   [5.3 Sequence Length Optimization](#53-sequence-length-optimization)
6.  [Running Training](#6-running-training)
7.  [Weights & Biases Integration](#7-weights--biases-integration)
8.  [Troubleshooting & Common Issues](#8-troubleshooting--common-issues)

---

## 1. Quick Start / How to Run

This section provides the basic commands to get training started quickly. For more context on the components within the `train/` directory, see `train/README.md`.

**Prerequisites:**
*   Docker installed.
*   Repository cloned.
*   `.env` file created in the root with `HF_TOKEN=your_huggingface_token_here` (and optionally `WANDB_API_KEY`).
*   Required datasets generated or downloaded (see `DATA_AUGMENTATION.md` or run `data/download_data.py` via python3 or docker).

**Steps:**

1.  **Build the Docker Image (from project root):**
    ```bash
    docker build -t mistral-nli-ft .
    ```

2.  **Run Training using the `run_training.sh` wrapper (from project root):**

    *   **Using default configuration (`train/configs/default.py`):**
        ```bash
        ./run_training.sh
        ```

    *   **Using a specific configuration file (e.g., the primary 'Ablation 2' config):**
        ```bash
        ./run_training.sh --config train/configs/ablation2.py
        ```

    *   **Specifying GPU (e.g., GPU ID 1):**
        ```bash
        ./run_training.sh --config train/configs/ablation2.py --gpu 1
        ```

    *   **Overriding parameters:**
        ```bash
        ./run_training.sh --config train/configs/ablation2.py --batch_size 4 --learning_rate 1e-4 --no_wandb
        ```

---

## 2. Training Philosophy & Setup

### 2.1 Goal

The objective is to instruction-tune the base Mistral-7B model using the prepared CoT dataset (`data/finetune/*.jsonl`). The model should learn to take a premise and hypothesis and generate a JSON output containing both a step-by-step `thought_process` and the final `predicted_label` (0 for no-entailment, 1 for entailment). This approach aims for both high accuracy and interpretability.

### 2.2 Parameter-Efficient Fine-Tuning (PEFT) with QLoRA

Given the size of Mistral-7B (7 billion parameters), full fine-tuning requires substantial GPU memory. We employ QLoRA to make this feasible on more accessible hardware (e.g., single or dual consumer GPUs like RTX 3090/4090).

*   **Quantization:** The base model weights are loaded in 4-bit precision using the NF4 ("NormalFloat 4") data type via the `bitsandbytes` library. Double quantization is often enabled for further memory savings. This drastically reduces the memory needed to hold the base model weights.
*   **LoRA (Low-Rank Adaptation):** Instead of updating all model weights, small, trainable "adapter" matrices are injected into specific layers of the frozen, quantized base model. Typically, these are applied to the attention mechanism's query (`q_proj`), key (`k_proj`), value (`v_proj`), and output (`o_proj`) linear layers. Only these adapter weights (a small fraction of the total parameters) are trained.
*   **Compute Precision (`bfloat16`):** While weights are stored in 4-bit, computations (like matrix multiplications during forward and backward passes) are performed using a higher precision format. `bfloat16` (Brain Floating Point) is preferred over `float16` for training stability, especially with newer GPU architectures, although it requires careful handling to avoid dtype mismatches (see [Section 8](#8-troubleshooting--common-issues)).

### 2.3 Docker Environment

Consistency across different machines (local workstations, cloud GPUs, clusters) is critical, especially with complex dependencies involving specific CUDA versions, PyTorch, and PEFT libraries. Docker is used to encapsulate the entire training environment.

*   **`Dockerfile`:** Defines the environment, starting from a PyTorch base image with CUDA 12.1, installing system dependencies (like `git`, `build-essential`), copying the code, and installing Python packages via `requirements.txt`.
*   **`requirements.txt`:** Lists key Python dependencies:
    *   `torch` (pinned to CUDA 12.1 compatible version)
    *   `transformers`, `peft`, `trl`, `datasets`, `accelerate` (core Hugging Face ecosystem)
    *   `bitsandbytes` (for QLoRA quantization)
    *   `sentencepiece`, `protobuf` (tokenizer dependencies)
    *   `flash-attn` (optional, for Flash Attention 2 optimization)
    *   `wandb` (optional, for logging)
*   **Build Command:** `docker build -t mistral-nli-ft .` (run from project root) creates the image named `mistral-nli-ft`.

---

## 3. Configuration System

A flexible configuration system, inspired by NanoGPT and located in the `train/` directory, manages hyperparameters and experiment settings.

### 3.1 Overview & Rationale

Managing numerous experiments with varying hyperparameters (learning rate, batch size, LoRA rank, etc.) purely through command-line arguments or complex bash scripts becomes unwieldy and difficult to reproduce. This Python-based config system addresses these issues:

*   **Readability & Maintainability:** Python files are more readable than long bash commands. Comments explain parameter choices.
*   **Structure & Defaults:** A `default.py` establishes base parameters, while specific experiment configs (e.g., `ablation2.py`) inherit from it and override only necessary values.
*   **Self-Documentation:** Each config file serves as a record of a specific experimental setup.
*   **Flexibility:** Allows easy definition of complex settings while still permitting quick command-line overrides for minor tweaks.
*   **Type Safety:** Python handles parameter types (int, float, bool) more robustly than bash.

(See `project_blog.md` for more background on adopting this system based on prior project experience).

### 3.2 Components & Usage

*   **`train/configs/default.py`:** Base configuration. Contains default values for most parameters.
*   **`train/configs/*.py`:** Specific experiment configurations (e.g., `ablation0.py`, `ablation1.py`, `ablation2.py`, `ablation3.py`, `minimal_test.py`, `distributed.py`). They typically start with `from train.configs.default import *` and then redefine specific variables.
*   **`train/config_loader.py`:** A utility script that loads the default config, then loads the specified experiment config (if any), and finally applies any overrides passed via command-line arguments.
*   **`train/train_sft.py`:** Imports and uses `config_loader.py` at the start to get the final configuration dictionary.
*   **Activation:** The `--config path/to/config.py` argument is passed to `run_training.sh`, which forwards it to `train_sft.py`. CLI args like `--batch_size 8` or `--no_wandb` are also parsed and take final precedence.

---

## 4. Training Script (`train/train_sft.py`)

This is the core Python script that performs the Supervised Fine-Tuning (SFT) using Hugging Face's `transformers` and `trl` libraries.

### 4.1 Overview

The script takes the configuration parameters, sets up the model, tokenizer, datasets, and trainer, handles potential precision issues, runs the training loop, and saves the resulting adapter model.

### 4.2 Key Steps & Implementation Details

1.  **Load Configuration:** Uses `config_loader.py` to merge defaults, specific config, and CLI overrides into a single config object.
2.  **Set Seed:** `transformers.set_seed(config.seed)` for reproducibility.
3.  **Load Datasets:** Loads `train` and `eval` splits from JSONL files specified in the config (e.g., `data/finetune/train_ft.jsonl`) using `datasets.load_dataset("json", data_files=...)`.
4.  **Load Tokenizer:** `AutoTokenizer.from_pretrained(config.model_id)`. Sets `padding_side='right'` and ensures `tokenizer.pad_token` is set (often to `tokenizer.eos_token`).
5.  **Define Quantization Config:** Creates `BitsAndBytesConfig` with `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`.
6.  **Load Base Model:** `AutoModelForCausalLM.from_pretrained(config.model_id, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)`. `device_map="auto"` distributes across available GPUs, or `{"": 0}` forces to GPU 0. Flash Attention 2 is used if available.
7.  **Prepare for K-bit Training:** `peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)`. This prepares the quantized model for PEFT and enables gradient checkpointing if `config.gradient_checkpointing` is True. Gradient checkpointing trades compute for memory, allowing larger models/batches.
8.  **Define LoRA Config:** Creates `LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=config.target_modules, bias="none", task_type="CAUSAL_LM")`.
9.  **Define Training Arguments:** Creates `SFTConfig` (a subclass of `TrainingArguments`) passing all hyperparameters from the config object (`output_dir`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `optim`, `lr_scheduler_type`, `warmup_ratio`, `weight_decay`, `max_seq_length`, `bf16`, `gradient_checkpointing`, logging/eval/save settings, `report_to`, `seed`, `remove_unused_columns=False`, etc.).
10. **Initialize SFTTrainer:** `SFTTrainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset['train'], eval_dataset=dataset['eval'], peft_config=peft_config, dataset_text_field="text")`. Note the `dataset_text_field="text"` assuming the preprocessed data has the instruction string under the "text" key.
11. **Handle BF16 Dtype Issues:** Includes checks for `model.lm_head.weight.dtype` and potential casting. Crucially, it might include a monkey-patch for the model's `forward` method using `torch.cuda.amp.autocast(dtype=torch.bfloat16)` to resolve dtype conflicts that can arise with `bfloat16` and gradient checkpointing. *Refer to `train/FINETUNE_GUIDE.md` or the `train_sft.py` source for the specific implementation of this patch if needed.*
12. **Train:** `trainer.train()` starts the fine-tuning loop.
13. **Save Model:** `trainer.save_model()` saves the trained LoRA adapter weights (not the full model) to the `output_dir`.

---

## 5. Fine-Tuning Strategy & Hyperparameters

### 5.1 QLoRA Configuration Rationale

The choice of QLoRA parameters involves balancing model expressiveness, training stability, and computational resources.

*   **Target Modules:** Targeting the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) is standard practice for LoRA as these layers are crucial for learning task-specific patterns.
*   **Rank (`r`) and Alpha (`lora_alpha`):**
    *   `r` determines the rank (size) of the adapter matrices. Higher `r` allows the adapter to capture more complex patterns but increases trainable parameters.
    *   `lora_alpha` acts as a scaling factor for the adapter weights. A common heuristic is `lora_alpha = 2 * r`.
    *   **Experimentation Findings & Recommendation:**
        *   Initial runs with `r=16`, `lora_alpha=32` proved effective, achieving good convergence and performance within reasonable training times (e.g., ~2 hours for 1 epoch with effective batch size 16). This configuration is considered a **strong, recommended starting point** for this project based on validated results.
        *   Higher rank settings (`r=32`, `lora_alpha=64`) were explored (e.g., in `train/configs/ablation3.py`), partly motivated by the hypothesis that the complex/reflected data might benefit from increased adapter capacity. However, when combined with other changes like larger batch sizes, this configuration did not consistently outperform the `r=16` setup and sometimes converged less effectively.
        *   The interaction between LoRA rank, batch size, and the specific dataset characteristics is complex. While `r=32/a=64` is available for experimentation, `r=16/a=32` represents the more validated effective configuration found so far.
*   **Dropout (`lora_dropout`):** A small dropout (e.g., 0.05) is applied to the LoRA layers for regularization.

### 5.2 Key Training Hyperparameters (`SFTConfig`)

These parameters control the optimization process:

*   **`num_train_epochs` (e.g., 3):** How many times to iterate over the training dataset. Experimentation showed continued convergence beyond 1 epoch, especially with the richer reflected dataset. Using validation loss (`eval_loss`) with early stopping (e.g., `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, and a `patience` value set via a custom callback or by monitoring logs) is highly recommended.
*   **Batch Size (`per_device_train_batch_size`, `gradient_accumulation_steps`):**
    *   `per_device_train_batch_size` is limited by GPU memory (e.g., 2-4 for a 7B model on a 24GB GPU with QLoRA).
    *   `gradient_accumulation_steps` allows simulating a larger effective batch size (`per_device_batch_size * num_gpus * grad_accum_steps`). An effective batch size of ~32-64 is common, but smaller effective batch sizes (e.g., 16) sometimes performed better in experiments here.
*   **Learning Rate (`learning_rate`, `lr_scheduler_type`, `warmup_ratio`):**
    *   `learning_rate` (e.g., `2e-4`) is often slightly higher for PEFT than full fine-tuning. Values between `1e-4` and `3e-4` were explored.
    *   A scheduler (`linear` or `cosine`) decays the learning rate over time. `cosine` is often preferred.
    *   `warmup_ratio` (e.g., `0.03`) gradually increases the LR at the start of training for stability.
*   **Optimizer (`optim`, `weight_decay`):**
    *   `paged_adamw_8bit` is a memory-efficient AdamW variant suitable for QLoRA.
    *   `weight_decay` (e.g., `0.01`) provides L2 regularization.
*   **Logging/Saving (`logging_steps`, `eval_strategy`, `eval_steps`, `save_strategy`, `save_steps`, `save_total_limit`):** Control how often metrics are logged, evaluation is performed, and checkpoints are saved. Frequent evaluation (`eval_steps` ~250-500) and saving (`save_steps` ~500-1000) are useful but add overhead. `save_total_limit` (e.g., 2-3) prevents excessive disk usage.
*   **Other:**
    *   `bf16=True`: Use bfloat16 mixed precision.
    *   `gradient_checkpointing=True`: Crucial for memory savings.
    *   `remove_unused_columns=False`: Prevents potential errors by stopping the Trainer from removing dataset columns.

### 5.3 Sequence Length Optimization (`max_seq_length`)

*   **Value:** Set to `512`.
*   **Rationale:** Analysis of the generated CoT data showed that shorter thought processes (~250-400 tokens total input+output) correlated with higher accuracy. Prompt engineering efforts focused on encouraging conciseness, which also improved initial generation accuracy. Setting `max_seq_length=512` comfortably accommodates these shorter sequences while dramatically reducing memory usage and computation time compared to default lengths (e.g., 2048 or 4096). This optimization was crucial for feasibility and performance. (See `project_blog.md`).

---

## 6. Running Training

As shown in the [Quick Start](#1-quick-start--how-to-run), training is initiated via the `run_training.sh` wrapper script from the project root directory.

```bash
./run_training.sh --config <path_to_config> [optional_overrides]
```

This script handles:
*   Setting up necessary environment variables.
*   Constructing the `docker run` command.
*   Mounting required volumes (`/app`, `/data`, `/models`, `/hf_cache`).
*   Passing the specified GPU ID to the container.
*   Executing `python train/train_sft.py` inside the container, forwarding the `--config` argument and any other command-line overrides.

---

## 7. Weights & Biases Integration

Weights & Biases (WandB) is integrated for experiment tracking.

*   **Setup:**
    1.  Ensure `wandb` is in `requirements.txt` and the Docker image is built/rebuilt.
    2.  Add `WANDB_API_KEY=your_wandb_api_key` to the `.env` file in the project root.
    3.  In the chosen `train/configs/*.py` file, set:
        *   `use_wandb = True`
        *   `wandb_project = "your_project_name"` (e.g., "mistral7b-nli-cot")
        *   `wandb_name = "your_specific_run_name"` (e.g., "ablation2-r16-lr2e4")
    4.  Alternatively, use CLI overrides: `./run_training.sh --config ... --use_wandb --wandb_project ... --wandb_name ...` (or `--no_wandb` to disable).
    5.  The `SFTConfig` argument `report_to` will be automatically set to `"wandb"` if `use_wandb` is true in the final config.
*   **Logged Metrics:** The `SFTTrainer` automatically logs training/evaluation loss, learning rate, throughput (samples/sec), and GPU memory usage.
*   **Custom Metrics:** To log NLI-specific metrics like accuracy, precision, recall, F1, a `compute_metrics` function needs to be implemented in `train_sft.py` and passed to the `SFTTrainer`. This function typically takes an `EvalPrediction` object, extracts predictions and labels, calculates metrics, and returns a dictionary. (Note: Check `train_sft.py` to see if this is currently implemented).

---

## 8. Troubleshooting & Common Issues

Fine-tuning large models can be challenging. Here are common issues and potential solutions encountered during this project, largely based on details originally in `train/FINETUNE_GUIDE.md`:

*   **Out-of-Memory (OOM) Errors:**
    *   **Cause:** Model, activations, gradients exceed GPU VRAM.
    *   **Solutions:**
        *   Decrease `per_device_train_batch_size` (e.g., to 1 or 2).
        *   Increase `gradient_accumulation_steps` to maintain effective batch size.
        *   Decrease `max_seq_length` (already optimized to 512 here).
        *   Ensure `gradient_checkpointing=True` in the config (critical for memory saving).
        *   Ensure `bitsandbytes` 4-bit quantization is active.

*   **Slow Training Speed:**
    *   **Cause:** Suboptimal hardware usage or configuration.
    *   **Solutions:**
        *   Ensure Flash Attention 2 is enabled and active (`attn_implementation="flash_attention_2"`). Requires compatible hardware (e.g., Ampere/Hopper GPUs) and recent PyTorch/CUDA. Check installation logs.
        *   Use the `paged_adamw_8bit` optimizer.
        *   Increase `per_device_train_batch_size` if memory allows.
        *   Reduce logging/evaluation frequency (`logging_steps`, `eval_steps`) if bottlenecking.
        *   Use `packing=True` in `SFTConfig` *only if* sequences are significantly shorter than `max_seq_length` on average (can hurt performance otherwise). It was disabled (`packing=False`) here for stability.

*   **BF16 Dtype Mismatch (`RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`):**
    *   **Cause:** Often occurs when using `bfloat16` precision (`bf16=True`) together with `gradient_checkpointing=True`. Parts of the model might remain in `float32` while others expect `bfloat16`.
    *   **Solutions:**
        *   Implement the `autocast` wrapper around the model's forward pass as potentially shown in `train_sft.py`. This explicitly tells PyTorch to execute the forward pass within a `bfloat16` context. The pattern looks like:
          ```python
          # In train_sft.py, potentially before initializing SFTTrainer
          original_forward = model.forward
          def forward_with_autocast(*args, **kwargs):
              with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                  return original_forward(*args, **kwargs)
          model.forward = forward_with_autocast
          ```
        *   Ensure model components like `lm_head` are explicitly cast to `bfloat16` if needed (the `train_sft.py` script may include checks for this).

*   **Poor Convergence / High Loss:**
    *   **Cause:** Suboptimal hyperparameters, data issues, or model capacity mismatch.
    *   **Solutions:**
        *   **Hyperparameters:** Adjust learning rate (try lower: `1e-4` or higher: `3e-4`), change LR scheduler, experiment with `weight_decay`.
        *   **LoRA:** Try increasing LoRA rank/alpha (e.g., `r=32`, `alpha=64`), but be mindful of potential negative interactions found in experiments here (see Section 5.1).
        *   **Data:** Verify the quality and format of the `data/finetune/*.jsonl` files. Ensure correct instruction formatting. Check for data imbalances or high noise levels.
        *   **Epochs:** Ensure enough training epochs (`num_train_epochs`) are run, monitoring `eval_loss` closely. Use early stopping.

*   **Library Compatibility Issues:**
    *   **Cause:** Version mismatches between `transformers`, `peft`, `trl`, `torch`, `accelerate`, `bitsandbytes`.
    *   **Solutions:**
        *   Strictly adhere to the versions in `requirements.txt`.
        *   Rebuild the Docker image after any changes to `requirements.txt`.
        *   Be aware of breaking changes in library APIs (e.g., `trl>=0.12.0` uses `processing_class` not `tokenizer` in `SFTTrainer`; `SFTConfig` uses `eval_strategy` not `evaluation_strategy`). Check library release notes if updating.

*   **Tensor Size Mismatch Errors (`The size of tensor a (X) must match the size of tensor b (Y)`):**
    *   **Cause:** Often related to data collation, padding, or unused columns when using custom datasets or specific Trainer arguments.
    *   **Solutions:**
        *   Set `remove_unused_columns=False` in `SFTConfig`. This prevents the Trainer from potentially dropping columns needed later.
        *   Try `per_device_train_batch_size=1` with increased `gradient_accumulation_steps`.
        *   Ensure consistent sequence lengths or proper padding/truncation handled by the tokenizer.
        *   Disable packing (`packing=False` in `SFTConfig`) if enabled.
