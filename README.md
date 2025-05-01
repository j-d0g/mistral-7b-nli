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