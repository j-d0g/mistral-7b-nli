# NLI Dataset Preparation

This document provides instructions for preparing the datasets needed for fine-tuning the Mistral-7B model on NLI tasks, with both quick start guides and in-depth technical explanations.

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Downloading Pre-Generated Datasets](#downloading-pre-generated-datasets)
  - [Dataset Structure](#dataset-structure)
  - [Next Steps](#next-steps)
- [Deep Dive: Data Preparation Pipeline](#deep-dive-data-preparation-pipeline)
  - [Complete Pipeline Workflow](#complete-pipeline-workflow)
  - [Stage 1: Generate Thoughts](#stage-1-generate-thoughts)
  - [Stage 2: Generate Reflections](#stage-2-generate-reflections)
  - [Stage 3: Prepare Fine-tuning Data](#stage-3-prepare-fine-tuning-data)
  - [Dependencies Between Stages](#dependencies-between-stages)
  - [Best Practices](#best-practices)

---

# Quick Start

## Prerequisites

Before you begin with the datasets, ensure you have:

1. **Docker installed** with NVIDIA Container Toolkit (for GPU support)
2. **Built the Docker image**:
   ```bash
   docker build -t mistral-nli-ft .
   ```
3. **Created a Hugging Face token** (store in `.env` file as `HF_TOKEN=your_token_here`)

## Downloading Pre-Generated Datasets

The fastest way to get started is to download the pre-generated datasets from Hugging Face:

```bash
# From the project root directory:
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/download_data.py
```

This will download all the necessary datasets and place them in the appropriate directories.

## Dataset Structure

After downloading, you'll have these key datasets:

*   `data/original_data/`: CSV files with premise-hypothesis pairs and labels (train.csv, dev.csv, test.csv)
*   `data/original_thoughts/`: JSON files with Chain-of-Thought reasoning for each example
*   `data/reflected_thoughts/`: JSON files with improved reasoning for examples that were initially incorrect
*   `data/finetune/`: JSONL files formatted for fine-tuning, including:
    - `train_ft.jsonl` - Standard training data
    - `dev_ft.jsonl` - Validation dataset
    - `sample_ft.jsonl` - Small sample dataset for testing

## Next Steps

After downloading the datasets, you can proceed to [training the model](TRAINING.md):

```bash
# Run training with the default configuration
./run_training.sh
```

---

# Deep Dive: Data Preparation Pipeline

This section explains how the datasets were created and how to generate your own datasets.

## Complete Pipeline Workflow

The data preparation involves three main stages, processed separately for training and validation datasets:

```
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐     ┌────────────────┐
│ Original    │     │ Original        │     │ Reflected         │     │ Fine-tuning    │
│ Data (CSV)  │────▶│ Thoughts (JSON) │────▶│ Thoughts (JSON)   │────▶│ Data (JSONL)   │
└─────────────┘     └─────────────────┘     └───────────────────┘     └────────────────┘
      │                     │                       │                        │
      │                     │                       │                        │
      ▼                     ▼                       ▼                        ▼
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐     ┌────────────────┐
│ train.csv   │────▶│train_thoughts.json────▶│train_reflections.json──▶│train_ft.jsonl  │
└─────────────┘     └─────────────────┘     └───────────────────┘     └────────────────┘
      │                     │                       │                        │
      │                     │                       │                        │
      ▼                     ▼                       ▼                        ▼
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐     ┌────────────────┐
│ dev.csv     │────▶│dev_thoughts.json│────▶│dev_reflections.json ────▶│dev_ft.jsonl    │
└─────────────┘     └─────────────────┘     └───────────────────┘     └────────────────┘
```

## Stage 1: Generate Thoughts

In this stage, we add Chain-of-Thought reasoning to the original CSV examples. This involves prompting the Mistral API to generate step-by-step reasoning for each example.

```bash
# Process training data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/generate_thoughts.py \
  --api mistral \
  --input-csv data/original_data/train.csv \
  --output-json data/original_thoughts/train_thoughts.json \
  --workers 6

# Process validation data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/generate_thoughts.py \
  --api mistral \
  --input-csv data/original_data/dev.csv \
  --output-json data/original_thoughts/dev_thoughts.json \
  --workers 6
```

The script outputs a JSON file containing each example with:
- Original premise and hypothesis
- Generated thought process
- Predicted label
- True label

## Stage 2: Generate Reflections

This stage identifies incorrect examples from Stage 1 and generates improved reasoning. It uses a potentially stronger model to reflect on the errors and provide better explanations.

```bash
# Process training data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/generate_thoughts_reflected.py \
  --api mistral \
  --model-name open-mistral-7b \
  --input-thoughts-json data/original_thoughts/train_thoughts.json \
  --output-reflection-json data/reflected_thoughts/train_reflections.json \
  --workers 6

# Process validation data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/generate_thoughts_reflected.py \
  --api mistral \
  --model-name open-mistral-7b \
  --input-thoughts-json data/original_thoughts/dev_thoughts.json \
  --output-reflection-json data/reflected_thoughts/dev_reflections.json \
  --workers 6
```

The output includes:
- Only examples where the predicted label was incorrect
- The original thought process
- Error analysis
- Improved reasoning that leads to the correct label

## Stage 3: Prepare Fine-tuning Data

The final stage combines:
1. Correct examples from the original thoughts (Stage 1)
2. Reflected examples with improved reasoning (Stage 2)

This creates a high-quality dataset for fine-tuning.

```bash
# Process training data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/train_thoughts.json \
  --reflected-thoughts data/reflected_thoughts/train_reflections.json \
  --output-file data/finetune/train_ft.jsonl

# Process validation data
docker run --rm --gpus all -v $(pwd):/app -w /app mistral-nli-ft python3 scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/dev_thoughts.json \
  --reflected-thoughts data/reflected_thoughts/dev_reflections.json \
  --output-file data/finetune/dev_ft.jsonl
```

The resulting JSONL files format the data with Mistral-style instruction tags, ready for fine-tuning.

## Dependencies Between Stages

The pipeline has clear dependencies:

- **Stage 1 → Stage 2**: The reflection process requires the output from the thought generation stage
- **Stage 2 → Stage 3**: The fine-tuning preparation requires both:
  - Original thoughts (to get correct examples)
  - Reflected thoughts (to get improved reasoning for incorrect examples)

## Best Practices

- Maintain consistent naming conventions throughout the pipeline
- Use the same parameters for both training and validation datasets
- Set `--workers 6` to maximize throughput and API rate limits
- Check the summary files generated after each stage
- Always validate the final JSONL files before fine-tuning

## Uploading Your Custom Datasets (Optional)

If you generate your own datasets, you can upload them to share:

```bash
# From the project root directory:
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/upload_data.py
```

## Further Information

- **Training process**: See [TRAINING.md](TRAINING.md)
- **Evaluation**: See [EVALUATION.md](EVALUATION.md) 