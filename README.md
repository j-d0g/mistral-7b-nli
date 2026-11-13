# Mistral-7B fine-tuning for NLI with chain-of-thought

Fine-tuning `mistralai/Mistral-7B-v0.3` with QLoRA (4-bit NF4 quantisation plus LoRA adapters) to classify
premise–hypothesis pairs as entailment (1) or no-entailment (0) while writing out a short chain-of-thought
(CoT) rationale before the label.

The training data was built with a two-stage pipeline: `open-mistral-7b` (via the Mistral API) wrote a
rationale and label for every pair; where its label disagreed with the gold label, the larger
`open-mistral-nemo` (12B) rewrote the rationale given the gold label and the flawed reasoning
("Reflection-CoT"). The fine-tuned model is evaluated zero-shot on a held-out test split.

## Results

All numbers below reproduce from public files; the commands are listed under
[Reproducing the numbers](#reproducing-the-numbers). Precision, recall and F1 are macro-averaged over the
two classes.

### Held-out test set (1,977 pairs, `data/original_data/test.csv`)

| Model | Training config | Accuracy | Precision | Recall | F1 | MCC |
|-------|-----------------|---------:|----------:|-------:|---:|----:|
| Mistral-7B-v0.3, zero-shot, no fine-tuning | – | 53.77% | 60.49% | 52.32% | 41.51% | 0.099 |
| Ablation 0 (effective batch 16, r=16) | `train/configs/ablation0_best.py` | 89.23% | 89.21% | 89.25% | 89.22% | 0.785 |
| **Ablation 1 (effective batch 32, r=16)** | `train/configs/ablation1_best.py` | **89.58%** | **89.57%** | **89.58%** | **89.57%** | **0.791** |
| Ablation 2 (effective batch 64, r=32) | `train/configs/ablation2.py` | 89.33% | 89.38% | 89.27% | 89.30% | 0.786 |

Ablation 1 is the best configuration: accuracy rises from 53.77% to 89.58% (+35.81 percentage points, a
66.60% relative improvement) and macro-F1 from 41.51% to 89.57% (a 115.78% relative improvement). The
zero-shot base model predicts "entailment" for 94% of pairs (1,861 of 1,977), which is why its F1 is far below its accuracy.
The three fine-tuned configurations are within 0.35 percentage points of each other.

Two provenance notes that matter when reading the table:

- The result files record which training run produced them: ablation 0 is `mistral-thinking-abl0-ext`,
  ablation 1 is `mistral-thinking-default-epochs2`, ablation 2 is `mistral-thinking-abl3`. The ablation 2
  test result comes from the `ablation2.py` configuration (learning rate 1e-4, 3 epochs), not from
  `ablation2_best.py`: the `ablation2_best.py` run (`mistral-thinking-abl3-final`) was trained on
  `train_ft_final.jsonl`, which merges the test pairs into training, so it is only reported on the dev split
  (88.72% accuracy).
- The training logs show that the ablation 0 and ablation 2 runs saw about 27,200 training rows per epoch,
  while the ablation 1 run saw the full 35,597-row `train_ft.jsonl`. The ~27,200-row file is not in the
  public dataset. See [Data and splits](#data-and-splits).

### Course hidden test set (3,302 pairs, unofficial)

The 3,302 unlabelled pairs in `data/original_data/submission.csv` are the course's hidden test set. The
model's predictions were scored by the course staff after the deadline, unofficially, because the wrong
predictions file had been submitted through the official channel. The scores reported back were:

| Metric | Score |
|--------|------:|
| Accuracy | 0.9667 |
| Weighted F1 | 0.9667 |
| Macro F1 | 0.9666 |
| Macro precision | 0.9670 |
| Macro recall | 0.9664 |
| Weighted precision | 0.9668 |
| MCC | 0.9334 |

These scores cannot be recomputed from this repository because the hidden labels are not public. Two
3,302-row prediction files exist in the results dataset: `nlistral-ablation1-submission.csv` (model
`mistral-thinking-default-epochs2`, the ablation 1 run) and `nlistral-ablation0-submission.csv` (model
`mistral-thinking-abl0-ext`).

## Data and splits

The labelled data is a binary NLI corpus supplied as part of a university coursework, published (with the
generated rationales) in the Hugging Face dataset
[`jd0g/Mistral-NLI-Thoughts`](https://huggingface.co/datasets/jd0g/Mistral-NLI-Thoughts) under the `data/`
prefix. The split used for every number in this repository is the one in `data/original_data/`:

| File | Rows | Entailment / no-entailment | Role |
|------|-----:|---------------------------:|------|
| `train.csv` | 35,593 | 18,338 / 17,255 | Fine-tuning |
| `dev.csv` | 1,977 | 1,048 / 929 | Validation during training (`eval_loss`, early stopping) |
| `test.csv` | 1,977 | 1,021 / 956 | Held-out evaluation; all test-set numbers above |
| `submission.csv` | 3,302 | unlabelled | Course hidden test set (inputs only) |

That is 39,547 labelled pairs, split 90.0% / 5.0% / 5.0%. No (premise, hypothesis) pair appears in more
than one of the three labelled files (exact-match check in `scripts/analysis/compute_test_metrics.py`).

The generated files follow the same split:

| File | Rows | Notes |
|------|-----:|-------|
| `original_thoughts/train_thoughts.json` | 35,597 | `open-mistral-7b` rationale + label for every training pair (4 pairs appear twice). 27,066 (76.03%) agree with the gold label, 8,531 (23.97%) do not. |
| `original_thoughts/dev_thoughts.json` | 1,977 | 1,480 (74.86%) agree with gold |
| `reflected_thoughts/train_reflections.json` | 8,531 | `open-mistral-nemo` rewrites for the 8,531 disagreeing training pairs |
| `reflected_thoughts/dev_reflections.json` | 497 | Rewrites for the disagreeing dev pairs |
| `finetune/train_ft.jsonl` | 35,597 | Training file: correct original rationales plus reflected rationales, in Mistral `[INST]` format |
| `finetune/dev_ft.jsonl` | 1,977 | Validation file |
| `finetune/train_ft_final.jsonl` | 37,571 | `train_ft.jsonl` plus 1,973 test pairs. Used only for the `ablation2_best.py` run and `mistral-thinking-abl0-merged`; **models trained on it must not be evaluated on `test.csv`.** |

So 23.97% of the training rationales were rewritten by the 12B model, and the fine-tuning file covers every
training pair.

## Repository structure

```
.
├── README.md                    # This file: overview, results, data and splits
├── REPORT.md                    # Method and results write-up
├── BLOG.md                      # Chronological notes on the experiments
├── DATA.md                      # How the rationales were generated (pipeline, commands)
├── TRAINING.md                  # QLoRA training (configs, commands)
├── EVALUATION.md                # Inference and evaluation
├── MODEL_CARD.md                # Model card (source for Hugging Face)
├── DATASET_CARD.md              # Dataset card (source for Hugging Face)
├── Dockerfile, requirements.txt
├── run_training.sh              # Training via Docker
├── run_inference.sh             # Inference via Docker
├── run_metrics.sh               # Parse predictions and compute metrics via Docker
├── run_notebook.sh              # Jupyter in Docker
├── demo.ipynb                   # Interactive demo
├── data/                        # Dataset download/upload scripts and small samples
├── train/                       # train_sft.py, config_loader.py, configs/
├── evaluate/                    # sample_model.py (inference), parse_predictions.py (metrics)
├── scripts/                     # generate_thoughts.py, generate_thoughts_reflected.py, prepare_ft_data.py, analysis/
├── models/                      # Model download/upload scripts
├── results/                     # Results download/upload scripts
├── metrics/                     # Figures and the metrics JSON behind them
├── figures/                     # Diagrams
├── llm/, service/, utils/       # API clients, generation/reflection/scoring services, prompts, JSON helpers
└── tests/                       # pytest suite for the services and JSON validation
```

## Documentation map

1. [DATA.md](DATA.md): generating rationales and reflections, and building the fine-tuning files.
2. [TRAINING.md](TRAINING.md): QLoRA fine-tuning with the Python config system.
3. [EVALUATION.md](EVALUATION.md): running inference and computing metrics.

[REPORT.md](REPORT.md) is the write-up of method and results; [BLOG.md](BLOG.md) is the chronological
account of what was tried. [MODEL_CARD.md](MODEL_CARD.md) and [DATASET_CARD.md](DATASET_CARD.md) are the
sources for the Hugging Face cards.

## Quick start

```bash
git clone https://github.com/j-d0g/mistral-7b-nli.git
cd mistral-7b-nli
docker build -t mistral-nli-ft .

# Download the datasets (public; HF_TOKEN in .env is optional)
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/download_data.py

# Train (quick smoke test on the sample data, then a real config)
./run_training.sh --config train/configs/quick_test.py
./run_training.sh --config train/configs/ablation1_best.py

# Evaluate on the held-out test split
./run_inference.sh --model models/nlistral-ablation1-best --data data/original_data/test.csv
```

The data-generation scripts (`scripts/generate_thoughts.py`, `scripts/generate_thoughts_reflected.py`,
`scripts/prepare_ft_data.py`) only call the Mistral API and can be run with plain Python. Training and
inference need a GPU with 16 GB or more of VRAM; Docker is recommended for those because of the CUDA,
`bitsandbytes` and `flash-attn` dependencies. `./run_notebook.sh` starts Jupyter in the container for
`demo.ipynb`.

## Reproducing the numbers

```bash
# Results dataset (public): per-example predictions for every evaluated run
python3 results/download_results.py

# Test-set metrics (accuracy, macro P/R/F1, MCC) and the train/test overlap check
python scripts/analysis/compute_test_metrics.py

# Reasoning-length versus accuracy, fine-tuned model (metrics/finetuned_token_vs_accuracy.json)
python scripts/analysis/analyze_finetuned_results.py

# Reasoning-length versus agreement with gold, open-mistral-7b rationales (metrics/original_token_vs_accuracy.json)
python scripts/analysis/analyze_original_thoughts.py

# Dataset and token statistics behind metrics/card_metrics.json
python scripts/analysis/analyze_token_lengths.py --train-file data/finetune/train_ft.jsonl \
    --dev-file data/finetune/dev_ft.jsonl --test-file data/original_data/test.csv

# Training curves from the public trainer_state.json logs (metrics/training_dynamics.png)
python scripts/analysis/plot_training_dynamics.py
```

Token counts in the analysis scripts are estimated as characters / 4, not with the Mistral tokenizer.

## Hugging Face

- Adapters: [`jd0g/nlistral-7b-qlora`](https://huggingface.co/jd0g/nlistral-7b-qlora) (gated; subfolders
  `nlistral-7b-qlora-ablation{0,1,2}-best`).
- Training runs with checkpoints and logs: [`jd0g/Mistral-7B-NLI-v0.3`](https://huggingface.co/jd0g/Mistral-7B-NLI-v0.3).
- Data: [`jd0g/Mistral-NLI-Thoughts`](https://huggingface.co/datasets/jd0g/Mistral-NLI-Thoughts).
- Results: [`jd0g/nlistral-7b-results`](https://huggingface.co/datasets/jd0g/nlistral-7b-results).

## Requirements

- Docker with the NVIDIA Container Toolkit
- An NVIDIA GPU with at least 16 GB of VRAM for training and inference
- A Hugging Face account and token for the gated adapter repository (the datasets are public)
- A Mistral API key to regenerate the rationales

## Licence

Code and documentation in this repository are released under the Apache License 2.0 (see
[LICENSE](LICENSE)). The underlying premise–hypothesis pairs were supplied as coursework data and are
redistributed in the Hugging Face dataset as received.

## Acknowledgements

- Mistral AI for Mistral-7B-v0.3 and the `open-mistral-7b` / `open-mistral-nemo` API models
- Hugging Face for `transformers`, `peft`, `trl` and `datasets`
- Dettmers et al. (2023) for QLoRA and Wei et al. (2022) for chain-of-thought prompting

*Generative AI disclosure: AI tools were used to help with iterative development of this solution, code
refactoring and modularisation, visualisations, and writing and maintaining the documentation. They were not
used to decide or steer the approach, which followed my own ideas and research interests.*
