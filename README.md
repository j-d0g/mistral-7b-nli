# Mistral-7b Fine-Tuning for NLI with Chain-of-Thought

This project focuses on fine-tuning the Mistral-7B language model for Natural Language Inference (NLI) tasks, specifically using Chain-of-Thought (CoT) reasoning to improve classification performance and interpretability.

## Project Goal

The primary objective is to instruction-tune Mistral-7B using a custom NLI dataset augmented with CoT reasoning. The final model should accurately classify premise-hypothesis pairs as either entailment (1) or no-entailment (0), maximizing performance on a hidden test set.

## Data

The core data is organized as follows:

*   `data/original_data/`: Original NLI premise-hypothesis pairs with labels (train.csv, dev.csv, test.csv).
*   `data/original_thoughts/`: JSON Lines files containing examples augmented with Chain-of-Thought (`thought_process`) and the model's original `predicted_label`, generated using `scripts/generate_thoughts.py`.
*   `data/reflected_thoughts/`: Contains reflection data for examples where the model prediction was incorrect, generated using `scripts/generate_thoughts_reflected.py`.
*   `data/finetune/`: Prepared data for fine-tuning in the instruction format expected by the SFT trainer.

> **Note:** All scripts default to using sample data if no specific paths are provided. This prevents accidental overwriting of important data during testing.

## Optimized Inference

We've optimized the inference process to efficiently process the 1977-sample NLI test set:

### Optimization Techniques

1. **4-bit Quantization**: Using `bitsandbytes` for efficient memory usage
2. **Batch Processing**: Optimal batch size of 32 for maximizing throughput
3. **Sequence Length Reduction**: Reduced from 2048 → 512 tokens (with actual inputs ranging from 212-465 tokens)
4. **Checkpoint System**: Saves progress every batch and can resume from interruptions
5. **Consistent Prompting**: Uses the exact prompt format from fine-tuning for optimal results

### Performance Gains

* Initial runtime estimate: ~2hr42min with batch size 8
* Optimized runtime: ~50min with batch size 32 (>3x speedup)
* GPU Memory Usage: 15.7GB/24GB VRAM (efficient usage while maintaining performance)

### Running Optimized Inference

To run inference on the test set:

```bash
./run_inference.sh
```

This script:
- Uses the optimized parameters (batch size 32, max length 512)
- Loads the 4-bit quantized Mistral 7B v0.3 model
- Runs with Chain-of-Thought reasoning enabled
- Saves results with timestamps for tracking

## Methodology

### 1. CoT Data Generation (Completed)

*   The `scripts/generate_thoughts.py` script is used to query the Mistral API (`open-mistral-7b` model) for each example in train and dev CSV files.
*   The script prompts the model to produce a step-by-step reasoning (`thought_process`) and a final classification label (`predicted_label`) in JSON format.
*   It supports three prompt types: `initial_generation`, `scoring`, and `regeneration`.
*   Analysis scripts are used to verify data integrity, check for duplicates, and calculate baseline performance metrics of the original model's predictions.

### 2. Reflection on Incorrect Examples (Optional)

*   The `scripts/generate_thoughts_reflected.py` script is used to generate improved reasoning for examples where the model prediction was incorrect.
*   This separate script takes the original thought process JSON and identifies the incorrect examples, then asks a (potentially stronger) model to reflect on the errors and generate improved reasoning.
*   The output includes the original thought process, error analysis, and improved reasoning, all preserving the correct label.
*   Reflection results are saved in `data/reflected_thoughts/` directory.

### 3. SFT Data Preparation

*   The `scripts/prepare_ft_data.py` script creates a dataset by combining correct examples from original predictions with reflected examples for incorrect predictions.
*   The script inputs the original thoughts and the reflected thoughts, filters for correct examples from the original dataset, and combines them with all the reflected examples.
*   This approach ensures the highest quality training data: correct examples are reused as-is, while incorrect examples are replaced with their reflected, improved versions.
*   The `scripts/prepare_finetuning_data.py` is also available for more general data preparation options.
*   All outputs use Mistral-style instruction tags `[INST]...[/INST]` to frame the task, with the target completion being the JSON string containing the `thought_process` and `predicted_label`.
    ```
    <s>[INST] Premise: ...\nHypothesis: ...\n\n...instruction... [/INST] {"thought_process": "...", "predicted_label": ...} </s>
    ```

### 4. Fine-Tuning Strategy

We use QLoRA for parameter-efficient fine-tuning.

*   **Base Model:** `mistralai/Mistral-7B-v0.3`
*   **QLoRA Config:** `r=32`, `lora_alpha=64`, dropout `0.05`, target modules `["q_proj", "k_proj", "v_proj", "o_proj"]`.
*   **Training Script:** `scripts/run_sft.py` (uses `transformers.Trainer` / `trl.SFTTrainer`).
*   **Hyperparameters:** 3-5 epochs (with early stopping patience 3 based on eval loss), LR `2e-4` (linear decay, 3% warmup), AdamW optimizer (WD `0.01`), effective batch size 64, `bf16` precision.

#### Training Ablations:

1.  **Ablation 1 (Correct Only):** Train on examples where the original model's prediction was correct.
2.  **Ablation 2 (Reflected Thought Process - *Primary Goal*):**
    *   Incorporate thought process reflections for originally incorrect examples with a stronger model.
    *   Combine high-quality reflected examples with original correct examples.
    *   Implemented in `scripts/prepare_ft_data.py`.
3.  **Ablation 3 (Unmodified):** Train on all original examples, regardless of correctness.

### 5. Docker Setup

*   A `Dockerfile` is provided to build a container image with all necessary dependencies (PyTorch, CUDA, Transformers, PEFT, TRL, bitsandbytes, etc.).
*   Training can be executed within this Docker container on a remote workstation with GPUs.
*   `requirements.txt` lists the Python dependencies.

### 6. Evaluation

*   Primary evaluation metric is accuracy on the hidden test set.
*   During training, validation loss is monitored for early stopping and checkpointing.
*   Validation accuracy, precision, recall, and F1 can also be tracked if a `compute_metrics` function is added to `run_sft.py`.

## Complete Pipeline

The complete data preparation pipeline consists of the following steps:

1. **Generate Initial Thoughts** (generate_thoughts.py)
   - Input: Original CSV data (premise, hypothesis, true_label)
   - Output: JSON with original thought processes + predicted_label + correct flag

2. **Generate Reflections** (generate_thoughts_reflected.py)
   - Input: Original thought JSONs (from step 1)
   - Output: JSON with improved thought processes for incorrect examples only

3. **Prepare Fine-tuning Data** (prepare_ft_data.py)
   - Input: Original thoughts + Reflected thoughts
   - Output: JSONL formatted for fine-tuning (combines correct originals + all reflections)

4. **Optional: Score Thought Processes** (score_thoughts.py)
   - Input: Original and/or reflected thought JSONs
   - Output: Scored examples with quality assessment

For more detailed pipeline information and example commands, see [scripts/README.md](scripts/README.md).

## Repository Structure

```
.
├── Dockerfile
├── requirements.txt
├── run_inference.sh            # Main script to run optimized inference on test set
├── sample_model.py            # Optimized implementation of NLI inference with 4-bit quantization
├── prompts.py                  # Centralized prompt template definitions
├── data/
│   ├── original_data/         # Original NLI datasets (CSV)
│   ├── original_thoughts/     # Original model thought processes (JSON)
│   ├── reflected_thoughts/    # Reflected thought processes for incorrect examples (JSON)
│   └── finetune/              # Prepared data for fine-tuning (JSONL)
├── logs/                      # Organized logging directory
│   ├── thoughts/              # Logs from thought generation process
│   ├── reflections/           # Logs from reflection generation process
│   └── score/                 # Logs from scoring processes
├── models/                    # Directory to store trained models/adapters
├── results/                   # Inference results (predictions, checkpoints, metrics)
├── scripts/
│   ├── generate_thoughts.py            # Script for augmenting original dataset with CoT data
│   ├── generate_thoughts_reflected.py  # Script for generating reflected CoT data on inaccurate initial predictions
│   ├── score_thoughts.py               # Script for generating a scoring and self-improvement loop for thought processes
│   ├── prepare_ft_data.py              # Script to prepare Ablation 2 data (correct + reflections)
│   ├── prepare_finetuning_data.py      # General script to format data for SFT with various options
│   └── run_sft.py                      # Script to run QLoRA SFT
├── service/                   # Service modules for prediction, reflection, and scoring
│   ├── prediction_service.py        # Service for generating predictions & initial thought processes
│   ├── reflection_service.py        # Service for generating reflections & improved thought processes
│   └── scoring_service.py           # Service for generating scored & improved thought processes
├── README.md
└── ... (Other files)
```

## Code Organization

The codebase is organized into several key components:

### Inference
- `sample_model.py` - Main inference script for running models on test data
- `prompts.py` - Contains prompts for the NLI task, including Chain-of-Thought templates

### Prediction Parsing
- `parse_predictions.py` - Core script for parsing structured predictions from model outputs
- `parse_predictions_with_tracking.py` - Enhanced version that tracks extraction methods
- `strict_evaluation.py` - Academic evaluation with strict parsing criteria

### Analysis
- `extraction_analysis.py` - Detailed analysis of extraction methods across models
- `compare_fixed_results.py` - Compares results before and after improved extraction

### Shell Scripts
- `test_*.sh` - Scripts for running inference on different models/datasets
- `analyze_extraction_methods.sh` - Script to run comprehensive extraction analysis

### Extraction Logic

All NLI models output their predictions in text format, which must be parsed to extract structured predictions (0/1 labels). The extraction logic follows this priority:

1. Parse JSON objects with "predicted_label" or "label" fields
2. Look for explicit statements like "final label: 0" 
3. Search for conclusion statements like "is entailed" or "not entailed"
4. For Chain-of-Thought outputs, check step 3 conclusions
5. Default to the majority class (typically 1 for NLI)

The improved extraction logic in `sample_model.py` and parsing scripts ensures accurate prediction extraction across different output formats, avoiding biases that could impact model evaluation. 