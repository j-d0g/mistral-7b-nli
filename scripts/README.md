# Scripts for NLI Data Generation and Preparation

This directory contains Python scripts used to generate Chain-of-Thought (CoT) data and prepare it for fine-tuning the NLI model.

## Core Scripts & Usage Examples

**1. Generate Initial Thoughts (`generate_thoughts.py`):**
Creates CoT reasoning and predictions from a base CSV dataset using an LLM API.

```bash
# Example using Mistral API on train.csv
python scripts/generate_thoughts.py \
  --input-csv data/original_data/train.csv \
  --output-json data/original_thoughts/train_thoughts.json \
  --api mistral \
  --model-name open-mistral-7b \
  --workers 6
```

**2. Generate Reflected Thoughts (`generate_thoughts_reflected.py`):**
Takes the output from `generate_thoughts.py`, identifies examples where the initial prediction was incorrect, and generates improved reasoning using an LLM API (potentially a stronger one).

```bash
# Example using Mistral API (Nemo model) on train_thoughts.json
python scripts/generate_thoughts_reflected.py \
  --input-thoughts-json data/original_thoughts/train_thoughts.json \
  --output-reflection-json data/reflected_thoughts/train_reflections.json \
  --api mistral \
  --model-name open-mistral-nemo \
  --workers 6
```

**3. Prepare Fine-tuning Data (`prepare_ft_data.py`):**
Combines the *correct* examples from the initial thought generation with *all* examples from the reflected thought generation to create the final dataset for SFT (Supervised Fine-Tuning).

```bash
# Example combining train thoughts and reflections
python scripts/prepare_ft_data.py \
  --original-thoughts data/original_thoughts/train_thoughts.json \
  --reflected-thoughts data/reflected_thoughts/train_reflections.json \
  --output-file data/finetune/train_ft.jsonl
```

**Other Scripts:**

*   `score_thoughts.py`: (Optional) Used for experimental scoring of thought process quality.
*   `prepare_finetuning_data.py`: A more general script for formatting data with basic filtering (superseded by `prepare_ft_data.py` for the main workflow).

---

*For detailed explanations of the data augmentation process, rationale, script parameters, and the full project workflow, please refer to the **[DATA_AUGMENTATION.md](../DATA_AUGMENTATION.md)** document and the main **[README.md](../README.md)** at the project root.*