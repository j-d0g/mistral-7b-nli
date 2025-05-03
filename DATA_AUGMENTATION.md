# Data Augmentation: Generating Chain-of-Thought NLI Data

This document details the process used to augment the original NLI dataset with Chain-of-Thought (CoT) reasoning using Large Language Models (LLMs), and prepare it for fine-tuning.

## 1. Overview & Rationale

The primary goal of this phase was to transform the standard NLI dataset (premise, hypothesis, label) into a richer format suitable for instruction fine-tuning an autoregressive model like Mistral-7B. This involved prompting LLMs to generate a step-by-step `thought_process` explaining how they arrived at a `predicted_label` for each NLI pair.

A two-stage approach was adopted:

1.  **Initial CoT Generation:** Use a cost-effective and relatively strong base model (`open-mistral-7b`) to generate initial thoughts for all examples. Analysis showed this model often produced simpler, accurate reasoning for this dataset (see `project_blog.md` for details).
2.  **Reflection on Errors:** For examples where the initial model's prediction was incorrect, use a more powerful reasoning model (`open-mistral-nemo`) to generate a *corrected* thought process, guided by the true label. This specifically targets harder examples and aims to teach the fine-tuned model more robust reasoning paths.

The final fine-tuning dataset combines the successful initial generations with these corrected reflections.

## 2. Data Generation & Preparation Pipeline

The process involves several scripts executed sequentially:

1.  **Generate Initial Thoughts (`scripts/generate_thoughts.py`)**
    *   **Input:** Original CSV data (e.g., `data/original_data/train.csv`) containing `premise`, `hypothesis`, `label`.
    *   **Process:** Prompts an LLM (e.g., `open-mistral-7b`) for each row to generate a `thought_process` and `predicted_label`. Compares `predicted_label` to the true `label` to add a `correct` flag.
    *   **Output:** JSON Lines file (e.g., `data/original_thoughts/train_thoughts.jsonl`) with `premise`, `hypothesis`, `label`, `thought_process`, `predicted_label`, `correct`.

2.  **Generate Reflected Thoughts (`scripts/generate_thoughts_reflected.py`)**
    *   **Input:** The JSONL file from Step 1 (e.g., `data/original_thoughts/train_thoughts.jsonl`).
    *   **Process:** Filters for examples where the `correct` flag is `false`. Prompts a potentially stronger LLM (e.g., `open-mistral-nemo`) with the original premise, hypothesis, *true label*, and the flawed initial thought process, asking it to generate an *improved* thought process that reaches the true label.
    *   **Output:** JSON Lines file (e.g., `data/reflected_thoughts/train_reflections.jsonl`) containing only the initially incorrect examples, but now with fields like `improved_thought_process` and preserving the original `premise`, `hypothesis`, `label`.

3.  **Prepare Fine-tuning Data (`scripts/prepare_ft_data.py`)**
    *   **Input:** Both the original thoughts JSONL (Step 1) and the reflected thoughts JSONL (Step 2).
    *   **Process:** Combines examples where `correct` was `true` from the original thoughts with *all* examples from the reflected thoughts. Formats each combined example into the specific instruction template required by the SFT Trainer (using `<s>[INST] ... [/INST] ... </s>` tags). The target completion is the JSON string containing the thought process (either original or reflected) and the predicted label.
    *   **Output:** Final JSON Lines file (e.g., `data/finetune/train_ft.jsonl`) ready for fine-tuning.

4.  **Optional: Score Thought Processes (`scripts/score_thoughts.py`)**
    *   **Input:** Original and/or reflected thought JSONLs.
    *   **Process:** Uses an LLM (potentially a very strong one like Mixtral) to evaluate the quality of the generated `thought_process`. (Note: This was found to be expensive and less reliable than the reflection approach for improving the data).
    *   **Output:** Scored examples.
    *   **Purpose:** Primarily for analysis and understanding the quality of generated reasoning.

## 3. Script Details

> **Note:** Scripts default to using sample data files (e.g., `data/sample/demo.csv`) if specific input/output paths are not provided. This prevents accidental overwriting of main datasets during testing. API keys are required for models accessed via API and should be configured appropriately (e.g., via environment variables).

### 3.1 `scripts/generate_thoughts.py`

Generates initial Chain-of-Thought reasoning and predictions.

**Parameters:**

*   `--input-csv`: Path to the input CSV file (default: `data/sample/demo.csv`).
*   `--output-json`: Path to save the output JSON Lines file (default: auto-generated based on input/model).
*   `--failed-csv`: Path to save failed examples (default: auto-generated).
*   `--api`: API provider: `mistral` or `deepseek` (required).
*   `--model-name`: Specific model name (e.g., `open-mistral-7b`, `deepseek-chat`) (required).
*   `--workers`: Number of parallel workers (default: 1). Use more for faster processing (e.g., 6).
*   `--start-index`, `--end-index`: Process a subset of rows from the input CSV.
*   `--system-prompt`: Which prompt template from `prompts.py` to use (default: `initial_generation`).

**Example Usage:**

```bash
# Generate thoughts for train set using open-mistral-7b
python scripts/generate_thoughts.py \
  --input-csv data/original_data/train.csv \
  --output-json data/original_thoughts/train_thoughts.jsonl \
  --api mistral \
  --model-name open-mistral-7b \
  --workers 6

# Generate thoughts for dev set
python scripts/generate_thoughts.py \
  --input-csv data/original_data/dev.csv \
  --output-json data/original_thoughts/dev_thoughts.jsonl \
  --api mistral \
  --model-name open-mistral-7b \
  --workers 6
```

### 3.2 `scripts/generate_thoughts_reflected.py`

Generates improved reasoning (reflections) for examples where the initial prediction was incorrect.

**Parameters:**

*   `--input-thoughts-json`: Path to the JSONL file from `generate_thoughts.py` (default: `data/sample/sample_thoughts.jsonl`).
*   `--output-reflection-json`: Path to save reflection results JSONL (default: auto-generated).
*   `--failed-csv`: Path to save failed examples (default: auto-generated).
*   `--api`: API provider: `mistral` or `deepseek` (required).
*   `--model-name`: Specific model name for reflection (e.g., `open-mistral-nemo`) (required).
*   `--workers`: Number of parallel workers (default: 1).
*   `--max-retries`: Max retries for API calls (default: 5).

**Example Usage:**

```bash
# Generate reflections for train set using open-mistral-nemo
python scripts/generate_thoughts_reflected.py \
  --input-thoughts-json data/original_thoughts/train_thoughts.jsonl \
  --output-reflection-json data/reflected_thoughts/train_reflections.jsonl \
  --api mistral \
  --model-name open-mistral-nemo \
  --workers 6

# Generate reflections for dev set
python scripts/generate_thoughts_reflected.py \
  --input-thoughts-json data/original_thoughts/dev_thoughts.jsonl \
  --output-reflection-json data/reflected_thoughts/dev_reflections.jsonl \
  --api mistral \
  --model-name open-mistral-nemo \
  --workers 6
```

### 3.3 `scripts/prepare_ft_data.py`

Prepares the final fine-tuning dataset by combining correct original thoughts with all reflected thoughts. This is the primary script for creating the training data used in the main fine-tuning runs.

**Parameters:**

*   `--original-thoughts`: Path to the original thoughts JSONL file (default: `data/sample/sample_thoughts.jsonl`).
*   `--reflected-thoughts`: Path to the reflected thoughts JSONL file (default: `data/sample/sample_reflections.jsonl`).
*   `--output-file`: Path to save the processed JSON Lines file for fine-tuning (default: `data/sample/sample_ft.jsonl`).

**Example Usage:**

```bash
# Prepare train data
python scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/train_thoughts.jsonl \
  --reflected-thoughts data/reflected_thoughts/train_reflections.jsonl \
  --output-file data/finetune/train_ft.jsonl

# Prepare dev data
python scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/dev_thoughts.jsonl \
  --reflected-thoughts data/reflected_thoughts/dev_reflections.jsonl \
  --output-file data/finetune/dev_ft.jsonl
```

### 3.4 Other Scripts

*   **`scripts/prepare_finetuning_data.py`**: A more general-purpose script for formatting data into the SFT instruction format. It includes options like `--filter-correct` (to replicate Ablation 1 data) but is largely superseded by `prepare_ft_data.py` for the main reflection-based approach.
    *   `--input-file` (required), `--output-file` (required), `--filter-correct`, `--label-field`.
*   **`scripts/score_thoughts.py`**: Used experimentally to evaluate CoT quality using an LLM scorer.
    *   `--input-json` (required), `--output-dir`, `--failed-csv`, `--api` (required), `--model-name`, `--workers`, `--start-index`, `--end-index`.

## 4. Data Formats

*   **Input (`data/original_data/*.csv`)**: Standard CSV with columns like `premise`, `hypothesis`, `label`.
*   **Intermediate (`data/original_thoughts/*.jsonl`, `data/reflected_thoughts/*.jsonl`)**: JSON Lines format. Each line is a JSON object.
    *   *Original Thoughts:* Contains keys like `premise`, `hypothesis`, `label` (ground truth), `thought_process` (generated CoT), `predicted_label` (model's prediction), `correct` (boolean: `predicted_label == label`).
    *   *Reflected Thoughts:* Contains similar keys, but includes fields like `improved_thought_process` representing the corrected reasoning. Structure might vary slightly based on script implementation details but preserves essential NLI info.
*   **Fine-tuning Output (`data/finetune/*.jsonl`)**: JSON Lines format. Each line is a JSON object with a single key, `text`, containing the fully formatted instruction string for the SFT trainer:
    ```json
    {"text": "<s>[INST] Premise: ...\nHypothesis: ...\n\n...instruction... [/INST] {\"thought_process\": \"...\", \"predicted_label\": ...} </s>"}
    ```
    (Where `thought_process` is either the original correct one or the improved reflected one).

## 5. Dependencies & Setup

*   **Python Environment:** Requires Python 3.x. Libraries like `requests`, `tqdm`, `pandas`, `python-dotenv` are needed (install via `pip`).
*   **API Keys:** Access to LLM APIs (Mistral, DeepSeek, etc.) requires API keys, typically configured via a `.env` file or environment variables.
*   **(Optional) Docker:** While generation scripts can run locally, using the project's Docker container ensures all dependencies are met consistently, especially if integrating with subsequent training steps. 