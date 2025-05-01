# NLI Chain-of-Thought Scripts Guide

This document provides detailed instructions for running the various scripts in this project, along with example commands and explanations of their parameters.

## Table of Contents

1. [Initial Thought Generation](#initial-thought-generation)
2. [Reflection on Incorrect Examples](#reflection-on-incorrect-examples)
3. [Scoring Thought Processes](#scoring-thought-processes)
4. [Preparing Fine-Tuning Data](#preparing-fine-tuning-data)
5. [Complete Pipeline](#complete-pipeline)

## Generating Thoughts

The `generate_thoughts.py` script generates chain-of-thought reasoning for NLI examples using LLM APIs.

> **Note:** By default, scripts use sample data files if no paths are explicitly provided. This avoids accidental overwriting of important data.

#### Usage

Generate thought processes for train set:
```bash
python scripts/generate_thoughts.py \
  --input-csv data/original_data/train.csv \
  --output-json data/original_thoughts/train_thoughts.json \
  --api mistral \
  --model-name open-mistral-7b \
  --workers 6
```

Generate thought processes for dev set:
```bash
python scripts/generate_thoughts.py \
  --input-csv data/original_data/dev.csv \
  --output-json data/original_thoughts/dev_thoughts.json \
  --api mistral \
  --model-name open-mistral-7b \
  --workers 6
```

Or run with defaults (on sample data):
```bash
python scripts/generate_thoughts.py --api mistral --model-name open-mistral-7b
```

#### Parameters

- `--input-csv`: Path to the input CSV file containing NLI examples (defaults to sample.csv)
- `--output-json`: Path to save the output JSON file (auto-generated based on model and input file)
- `--failed-csv`: Path to save failed examples (auto-generated if not specified)
- `--api`: Which API to use - `mistral` or `deepseek` (required)
- `--model-name`: Name of the LLM (e.g., `open-mistral-7b`, `deepseek-chat`)
- `--workers`: Number of parallel workers (default: 1)
- `--start-index`, `--end-index`: Process a subset of examples from the input CSV
- `--system-prompt`: Prompt type from `prompts.py` (choices: `initial_generation`, `scoring`, `regeneration`)

## Generate Reflected Thoughts

The `generate_thoughts_reflected.py` script generates improved reasoning for examples where the initial model prediction was incorrect.

#### Parameters

- `--input-thoughts-json`: Path to the input JSON file with original thought processes (defaults to sample_thoughts.json)
- `--output-reflection-json`: Path to save reflection results (defaults to sample_reflections.json)
- `--failed-csv`: Path to save failed examples (auto-generated if not specified)
- `--api`: Which API to use - `mistral` or `deepseek` (required)
- `--model-name`: Name of the model for reflection (required)
- `--workers`: Number of worker processes (default: 1)
- `--max-retries`: Maximum number of retries for LLM calls (default: 5)

#### Usage

Generate reflections for train set:
```bash
python scripts/generate_thoughts_reflected.py \
  --input-thoughts-json data/original_thoughts/train_thoughts.json \
  --output-reflection-json data/reflected_thoughts/train_reflections.json \
  --api mistral \
  --model-name open-mistral-nemo \
  --workers 6
```

Generate reflections for dev set:
```bash
python scripts/generate_thoughts_reflected.py \
  --input-thoughts-json data/original_thoughts/dev_thoughts.json \
  --output-reflection-json data/reflected_thoughts/dev_reflections.json \
  --api mistral \
  --model-name open-mistral-nemo \
  --workers 6
```

Or run with defaults (on sample data):
```bash
python scripts/generate_thoughts_reflected.py --api mistral --model-name open-mistral-nemo
```

## Scoring Thought Processes

The `score_thoughts.py` script evaluates the quality of chain-of-thought reasoning.

#### Usage

Score original thought processes:
```bash
python scripts/score_thoughts.py \
  --input-json data/original_thoughts/train_thoughts.json \
  --output-dir data/scored_thoughts/train_original \
  --api mistral \
  --model-name open-mixtral-8x7b \
  --workers 6
```

Score reflected thought processes:
```bash
python scripts/score_thoughts.py \
  --input-json data/reflected_thoughts/train_reflections.json \
  --output-dir data/scored_thoughts/train_reflected \
  --api mistral \
  --model-name open-mixtral-8x7b \
  --workers 6
``` 

#### Parameters

- `--input-json`: Path to the input JSON file or directory containing JSONs with thought processes to score (required)
- `--output-dir`: Directory to save gold and low standard outputs (auto-generated if not specified)
- `--failed-csv`: Path to save details of failed examples (auto-generated if not specified)
- `--api`: Which API to use - `mistral` or `deepseek` (required)
- `--model-name`: Name of the model to use for scoring (defaults depend on the API selected)
- `--workers`: Number of worker processes (default: 1)
- `--start-index`, `--end-index`: Process a subset of examples from the input JSON

#### Field Naming Support

The script handles multiple field naming conventions for compatibility:
- Thought process field: `thought_process`, `improved_thought_process`, or `initial_thought`
- Prediction field: `label`, `prediction`, or `predicted_label`
- True label field: `true_label`

The script standardizes these fields to `thought_process` and `predicted_label` internally for consistent processing.

## Preparing Fine-Tuning Data

### Ablation 2 Data Preparation (Correct + Reflected)

The `prepare_ft_data.py` script prepares data for Ablation 2, which combines correct examples from the original model with reflected examples for the incorrect cases.

#### Usage

Prepare train data:
```bash
python scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/train_thoughts.json \
  --reflected-thoughts data/reflected_thoughts/train_reflections.json \
  --output-file data/finetune/train_ft.jsonl
```

Prepare dev data:
```bash
python scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/dev_thoughts.json \
  --reflected-thoughts data/reflected_thoughts/dev_reflections.json \
  --output-file data/finetune/dev_ft.jsonl
```

Or run with defaults (on sample data):
```bash
python scripts/prepare_ft_data.py
```

#### Parameters

- `--original-thoughts`: Path to the original thoughts JSON Lines file (defaults to sample_thoughts.json)
- `--reflected-thoughts`: Path to the reflected thoughts JSON Lines file (defaults to sample_reflections.json)
- `--output-file`: Path to save the processed JSON Lines file for fine-tuning (defaults to sample_ablation2.jsonl)

### General Fine-Tuning Data Preparation

The `prepare_finetuning_data.py` script provides more general options for preparing fine-tuning data.

#### Usage

```bash
python scripts/prepare_finetuning_data.py \
  --input-file data/original_thoughts/train_thoughts.json \
  --output-file data/finetune/train_unmodified.jsonl

# To filter for only correct examples (Ablation 1)
python scripts/prepare_finetuning_data.py \
  --input-file data/original_thoughts/train_thoughts.json \
  --output-file data/finetune/train_correct_only.jsonl \
  --filter-correct
```

#### Parameters

- `--input-file`: Path to the input JSON Lines file (required)
- `--output-file`: Path to save the processed JSON Lines file (required)
- `--filter-correct`: If set, only include examples where 'correct' is true
- `--label-field`: Field to use for the target label ('prediction' or 'true_label')

## Complete Pipeline

The complete data preparation pipeline consists of the following steps:

1. **Generate Initial Thoughts** (generate_thoughts.py)
   - Input: Original CSV data (premise, hypothesis, true_label)
   - Output: JSON with original thought processes + predicted_label + correct flag
   - Purpose: Get CoT reasoning and predictions for all examples

2. **Generate Reflections** (generate_thoughts_reflected.py)
   - Input: Original thought JSONs (from step 1)
   - Output: JSON with improved thought processes for incorrect examples only
   - Purpose: Generate better reasoning for examples where the model was wrong

3. **Prepare Fine-tuning Data** (prepare_ft_data.py)
   - Input: Original thoughts + Reflected thoughts
   - Output: JSONL formatted for fine-tuning (combines correct originals + all reflections)
   - Purpose: Create the final dataset for model fine-tuning

4. **Optional: Score Thought Processes** (score_thoughts.py)
   - Input: Original and/or reflected thought JSONs
   - Output: Scored examples with quality assessment
   - Purpose: Analyze the quality of reasoning before/after reflection

This pipeline allows for creating high-quality training data that contains correct original thought processes and improved reflections for examples that were initially incorrect.

For a test run of the entire pipeline on sample data:

```bash
# Step 1: Generate initial thoughts
python scripts/generate_thoughts.py --api mistral --model-name open-mistral-7b

# Step 2: Generate reflections for incorrect examples
python scripts/generate_thoughts_reflected.py --api mistral --model-name open-mistral-nemo

# Step 3: Prepare fine-tuning data
python scripts/prepare_ft_data.py
```