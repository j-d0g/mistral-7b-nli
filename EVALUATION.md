# Model Evaluation & Inference

This document details the process for running inference with the fine-tuned Mistral-7B NLI model and evaluating its performance. It covers the architecture, usage, optimizations, outputs, and findings related to the evaluation pipeline.

**Table of Contents**

1.  [Quick Start / How to Run Inference](#1-quick-start--how-to-run-inference)
2.  [Evaluation Philosophy & Goals](#2-evaluation-philosophy--goals)
3.  [Inference Architecture & Design Rationale](#3-inference-architecture--design-rationale)
    *   [3.1 Key Components & Optimizations](#31-key-components--optimizations)
    *   [3.2 Design Decisions Rationale](#32-design-decisions-rationale)
4.  [Core Implementation](#4-core-implementation)
    *   [4.1 Unified Script (`run_inference.sh`)](#41-unified-script-run_inferencesh)
    *   [4.2 Python Script (`evaluate/sample_model.py`)](#42-python-script-evaluatesample_modelpy)
5.  [Output Formats & Interpretation](#5-output-formats--interpretation)
    *   [5.1 JSON Output](#51-json-output)
    *   [5.2 CSV Output](#52-csv-output)
6.  [Performance Characteristics](#6-performance-characteristics)
    *   [6.1 Resource Usage](#61-resource-usage)
    *   [6.2 Speed / Throughput](#62-speed--throughput)
7.  [Key Evaluation Findings & Historical Context](#7-key-evaluation-findings--historical-context)
8.  [Troubleshooting Inference](#8-troubleshooting-inference)

---

## 1. Quick Start / How to Run Inference

This section provides the basic commands to run inference using the primary wrapper script. For more context on the components within the `evaluate/` directory, see `evaluate/README.md`.

**Prerequisites:**
*   Docker installed and configured for GPU usage.
*   Repository cloned.
*   `.env` file created in the root with `HF_TOKEN=your_huggingface_token_here`.
*   Docker image built (`docker build -t mistral-nli-ft .`).
*   Fine-tuned model adapters available in the `models/` directory (or downloaded via `models/download_model.py`).
*   Input data file (CSV format) available (e.g., `data/original_data/test.csv` or `data/sample/demo.csv`).

**Steps (using `run_inference.sh` from project root):**

1.  **Run with default parameters:** (Uses a default model path and `data/sample/demo.csv`)
    ```bash
    ./run_inference.sh
    ```

2.  **Run with a specific model adapter directory and dataset:**
    ```bash
    # Example using a specific checkpoint and the test set
    ./run_inference.sh --model models/Mistral_Thinking_Abl2/checkpoint-2000 --data data/original_data/test.csv
    ```

3.  **Specify the GPU:** (Default is GPU 0)
    ```bash
    ./run_inference.sh --model models/Mistral_Thinking_Abl2/checkpoint-2000 --data data/original_data/test.csv --gpu 1
    ```

Outputs will be saved to the `results/` directory.

---

## 2. Evaluation Philosophy & Goals

The primary goal of the evaluation phase is to assess the performance of the fine-tuned Mistral-7B model on the NLI task, particularly on unseen data (like the hidden test set or the provided `dev.csv` / `test.csv`). Key objectives include:

*   **Accuracy:** Measuring the classification accuracy (entailment vs. no-entailment) compared to ground truth labels (if available).
*   **Interpretability:** Analyzing the generated `thought_process` to understand the model's reasoning and identify potential failure modes.
*   **Efficiency:** Ensuring the inference process is fast and resource-efficient, making evaluation feasible.
*   **Robustness:** Testing the model's ability to handle variations in input and produce consistently parseable JSON outputs.

---

## 3. Inference Architecture & Design Rationale

The inference pipeline is designed for efficiency, reproducibility, and ease of use.

### 3.1 Key Components & Optimizations

*   **Docker-Based Execution:** Ensures a consistent environment with correct dependencies (CUDA, PyTorch, Transformers, bitsandbytes, etc.), identical to the training environment.
*   **Unified Wrapper Script (`run_inference.sh`):** Provides a simple command-line interface for common inference scenarios, handling Docker orchestration.
*   **Core Python Script (`evaluate/sample_model.py`):** Contains the main logic for loading models, tokenizing data, running generation, and parsing outputs.
*   **4-bit Quantization (QLoRA Inference):** The base Mistral-7B model is loaded in 4-bit using `bitsandbytes` (NF4 type, double quantization) to drastically reduce VRAM requirements. The trained LoRA adapters are loaded on top.
*   **Batch Processing:** Inputs are processed in batches (default size 32 in `sample_model.py`, though OOM might require smaller batches set manually) to improve GPU throughput.
*   **Optimized Sequence Length (`max_length=512`):** Based on training data analysis and prompt engineering, a shorter sequence length is used, significantly reducing computation and memory compared to default model lengths.
*   **Flash Attention 2:** Leveraged if the hardware and libraries support it, further accelerating the attention mechanism.
*   **Consistent Prompting:** Uses the same `[INST]...[/INST]` instruction format employed during fine-tuning.
*   **Flexible Data Handling:** Automatically detects if the input CSV has a `label` column and calculates accuracy accordingly.
*   **Structured Output Parsing (`evaluate/parse_predictions.py`):** Includes robust logic to parse the generated JSON (containing `thought_process` and `predicted_label`) from the model's potentially noisy text output, handling variations like multiple JSON objects.

### 3.2 Design Decisions Rationale

(Adapted from `evaluate/SOLUTION_SUMMARY.md` and `project_blog.md`)

*   **Why 4-bit Quantization?** Essential for fitting the 7B model onto consumer GPUs (e.g., RTX 4090 24GB), reducing VRAM needs from ~28GB+ for FP16 to ~12-16GB, while maintaining acceptable inference quality for the PEFT adapters.
*   **Why Docker?** Guarantees environment consistency and reproducibility across different systems, avoiding dependency conflicts, especially with CUDA/PyTorch versions.
*   **Why Batch Processing?** Maximizes GPU utilization and significantly speeds up inference compared to processing samples individually.
*   **Why Sequence Length 512?** Analysis showed shorter CoT (<400 tokens) correlated with higher accuracy. Prompting for conciseness improved generation, making 512 a safe and highly efficient maximum length, boosting speed and reducing memory.
*   **Why Chain-of-Thought Support?** Included to leverage the fine-tuning strategy, enhance interpretability, and potentially improve accuracy on complex examples.
*   **Why Robust JSON Parsing?** Initial findings showed models might generate imperfect JSON or extra text. Robust parsing (using `evaluate/parse_predictions.py` logic within `sample_model.py`) is critical for accurate extraction of `predicted_label`.

---

## 4. Core Implementation

### 4.1 Unified Script (`run_inference.sh`)

This root-level bash script is the primary entry point for running inference.

*   **Functionality:** Handles Docker command construction, volume mounting (`/app`, `/data`, `/models`, `/hf_cache`), GPU selection (`--gpus device=...`), and execution of `evaluate/sample_model.py` inside the container.
*   **Parameters:**
    *   `--model`/`-m`: Path to the fine-tuned adapter directory or checkpoint (required). Default: `models/mistral-7b-nli-cot`.
    *   `--data`/`-d`: Path to the input CSV data file (required). Default: `data/sample/demo.csv`.
    *   `--gpu`/`-g`: GPU ID to use (optional). Default: `0`.
    *   `--help`/`-h`: Display help.
*   **Limitations:** Does not currently expose parameters like `batch_size` or `max_length` from `sample_model.py`. For finer control, `sample_model.py` must be run directly.

### 4.2 Python Script (`evaluate/sample_model.py`)

This script contains the core inference logic.

*   **Key Steps:**
    1.  Parses arguments (model path, data path, output path, batch size, max length, etc.).
    2.  Loads the tokenizer associated with the base Mistral model.
    3.  Sets up 4-bit `BitsAndBytesConfig`.
    4.  Loads the base Mistral model (`AutoModelForCausalLM.from_pretrained`) with the quantization config.
    5.  Loads the PEFT LoRA adapters onto the base model using `PeftModel.from_pretrained`.
    6.  Merges LoRA weights into the base model (optional but common for inference) or uses adapters directly.
    7.  Loads the dataset (CSV) using `datasets.load_dataset`.
    8.  Defines the inference prompt template, matching the training format.
    9.  Iterates through the dataset in batches:
        *   Formats prompts for the batch.
        *   Tokenizes the prompts (left-padding recommended for generation).
        *   Runs `model.generate()` with appropriate parameters (`max_new_tokens`, `do_sample=False`, etc.).
        *   Decodes the generated outputs.
        *   Uses robust parsing logic (from `evaluate/parse_predictions.py`) to extract `thought_process` and `predicted_label` from the decoded text.
    10. Compiles results (including original data, generated text, parsed thoughts/predictions).
    11. Calculates accuracy if labels were present.
    12. Saves detailed results to the specified output JSON file.
    13. Optionally saves a simplified CSV output.
*   **Configurable Parameters (via direct call):** `--model_id`, `--test_file`, `--output_file`, `--batch_size`, `--max_length`, `--use_cot`, etc.

---

## 5. Output Formats & Interpretation

Inference runs generate outputs in the `results/` directory with descriptive names (`[model_name]-[dataset_name]-[timestamp]`).

### 5.1 JSON Output (`.json`)

This is the primary, detailed output file.

*   **Top-Level Keys:**
    *   `model`: Path/name of the model used.
    *   `accuracy`: Overall accuracy (present only if input data had labels).
    *   `inference_time_seconds`: Wall-clock time for the prediction loop.
    *   `samples_per_second`: Calculated throughput.
    *   `batch_size`: Batch size used.
    *   `max_length`: Max sequence length used.
    *   `config` / other metadata may be included.
*   **`results` Key:** A list, where each element is a dictionary representing one input sample:
    *   Original data fields (e.g., `premise`, `hypothesis`).
    *   `true_label` (if available in input data).
    *   `predicted_label`: The model's final parsed prediction (0 or 1).
    *   `thought_process`: The parsed step-by-step reasoning.
    *   `full_output`: The complete raw text generated by the model before parsing.
    *   `prediction_valid`: Boolean indicating if the output was successfully parsed into the expected JSON format.

### 5.2 CSV Output (`.csv`)

A simplified, flattened version for quick analysis or submission.

*   **Columns:** Typically includes original data (`premise`, `hypothesis`), `true_label` (if available), and the final `predicted_label`.

---

## 6. Performance Characteristics

(Based on `evaluate/SOLUTION_SUMMARY.md` and other observations)

### 6.1 Resource Usage

*   **GPU Memory:** ~12-16GB VRAM required for the 4-bit quantized 7B model with adapters loaded. (Exact usage depends on batch size, sequence length).
*   **CPU:** Moderate usage, mainly for tokenization and post-processing.
*   **Disk:** ~15GB for Hugging Face model cache (base model + adapters). Result files are typically small.

### 6.2 Speed / Throughput

*   **Processing Rate:** Approximately 1-2 samples per second observed on an RTX 4090 GPU with default settings (Batch Size 32, Seq Length 512).
*   **Total Runtime:** Inference on the 1977-sample test set completed in ~20-30 minutes during testing.
*   **CoT Overhead:** Generating the Chain-of-Thought reasoning adds overhead compared to direct classification (estimated ~1.5-2x slower if direct classification were implemented).

---

## 7. Key Evaluation Findings & Historical Context

(Summarized from `evaluate/README.md`)

*   **Fine-tuning Impact:** Fine-tuning with the CoT dataset dramatically improved performance over the base Mistral-7B model on the NLI task. Accuracy jumped significantly (e.g., from ~53% baseline to ~91% on the test set with Ablation 2 model after fixing evaluation logic).
*   **Importance of Parsing:** Initial evaluation was skewed low due to naive parsing of the model's output JSON. Implementing robust parsing logic (`evaluate/parse_predictions.py`) that handles potential inconsistencies (like multiple JSONs) was crucial for accurate assessment.
*   **Historical Issues:** Earlier development faced challenges with tokenizer configurations (pad token setup during training) that potentially impacted model output generation, which were later rectified.

---

## 8. Troubleshooting Inference

Common issues when running `run_inference.sh` or `evaluate/sample_model.py`:

*   **CUDA Out of Memory (OOM) Errors:**
    *   **Cause:** Batch size too large for GPU VRAM.
    *   **Solution:** The `run_inference.sh` script uses a default batch size (e.g., 32) internally. If OOM occurs, you **must run `evaluate/sample_model.py` directly** inside Docker, specifying a smaller `--batch_size` (e.g., 16, 8, 4).

*   **Slow Inference Speed:**
    *   **Cause:** GPU not utilized correctly; Flash Attention not active.
    *   **Solution:** Verify Docker GPU setup (`nvidia-smi` inside container). Ensure Flash Attention 2 is installed in the Docker image and compatible with the hardware.

*   **`FileNotFoundError`:**
    *   **Cause:** Incorrect path to model adapters or input data CSV.
    *   **Solution:** Double-check paths provided to `--model` and `--data`. Ensure the specified model directory contains `adapter_config.json`, `adapter_model.bin`, etc.

*   **Invalid Prediction Output / Low Accuracy:**
    *   **Cause:** Wrong model loaded; data mismatch; prompt mismatch; parsing failure.
    *   **Solution:** Verify `--model` path points to the correct fine-tuned adapters. Check input data format. Ensure the prompt structure in `sample_model.py` aligns with the training prompt format. Examine the `full_output` and `prediction_valid` fields in the results JSON to diagnose parsing issues. 