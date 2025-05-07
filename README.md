# Mistral-7b Fine-Tuning for NLI with Chain-of-Thought
## Project Overview

This project focuses on fine-tuning the Mistral-7B language model for Natural Language Inference (NLI) tasks, specifically using Chain-of-Thought (CoT) reasoning to improve classification performance and interpretability.

The primary objective is to instruction-tune Mistral-7B using a custom NLI dataset augmented with CoT reasoning. The trained model can accurately classify premise-hypothesis pairs as either entailment (1) or no-entailment (0), while providing interpretable reasoning.

## Repository Structure

```
.
├── Dockerfile                   # Docker configuration
├── requirements.txt             
├── DATA.md                      # Dataset documentation
├── TRAINING.md                  # Training documentation
├── EVALUATION.md                # Evaluation documentation
├── BLOG.md                      # Experimental journey narrative
├── HUB_MODEL.md                 # Model card documentation
├── HUB_DATASET.md               # Dataset card documentation
├── REPORT.md                     # Research findings and methodology

├── run_training.sh              # Training script via Docker 
├── run_inference.sh             # Inference script via Docker
├── run_metrics.sh               # Compute Metrics script via Docker

├── data/                        # Augmented Data Directory
│   ├── original_data/           # Original CSV Data Files
│   ├── original_thoughts/       # Augmented JSON thought files
│   ├── reflected_thoughts/      # Reflected JSON thought files
│   ├── finetune/                # Final JSONL finetuning data
│   └── download_data.py         # Dataset download from HF

├── train/                       # Training components
│   ├── train_sft.py             # Main training implementation
│   ├── config_loader.py         # Configuration loading utility
│   └── configs/                 # Training configurations

├── evaluate/                    # Evaluation components
│   └── sample_model.py          # Model sampling implementation

├── scripts/                     # Data Augmentation & Preparation Scripts
│   ├── generate_thoughts.py     # Generate CoT reasoning
│   ├── generate_thoughts_reflected.py # Generate reflections
│   ├── prepare_ft_data.py       # Prepare fine-tuning data
│   └── analysis/                # Data analysis & visualization

├── models/                      # Model storage
│   └── download_model.py        # Model download from HF

├── results/                     # Evaluation results storage
│   └── download_results.py      # Benchmarks download from HF

├── metrics/                     # Visualization outputs and metrics
├── figures/                     # Diagrams and visualizations
├── utils/                       # Utility functions and helpers
├── logs/                        # Log files from training runs
├── tests/                       # Test files for the project
├── llm/                         # LLM interface utilities
└── service/                     # API and service implementations
```


## Documentation Directions

This repository is split into three parts:

1. **[DATA.md](DATA.md)** - Synthetic Chain-of-Thought augmentation of NLI Dataset.
2. **[TRAINING.md](TRAINING.md)** - Quantized model fine-tuning with QLoRA.
3. **[EVALUATION.md](EVALUATION.md)** - Model loading, inference and evaluation.

Each document includes both a Quick Start guide for getting up and running quickly, as well as a Deep Dive section with technical detail.
The quick start sections give you the option to download datasets, models and run inference quickly, or to reproduce the results as I did 
through each of the steps, from generating thoughts & reflections to training your own QLoRA adaptors.

Additional documentation:

* **[README.md](README.md)** - Project overview and setup instructions.
* **[REPORT.md](REPORT.md)** - Write-up on methodology and results.
* **[BLOG.md](BLOG.md)** - Chronological brain-dump style narrative of the experimental journey.
* **[HUB_DATASET.md](HUB_DATASET.md)** - Dataset card on HuggingFace.
* **[HUB_MODEL.md](HUB_MODEL.md)** - Model card on HuggingFace.


### Key Directories

* **data/** - Datasets and processing scripts
* **train/** - Training implementation and configs
* **evaluate/** - Inference and metrics code
* **results/** - Outputs from model evaluation
* **models/** - Storage for trained model checkpoints

### Recommended Reading Path

* **First-time users:** Start with **[README.md](README.md)**, then follow the three core pillars in order 
(**[DATA.md](DATA.md)** → **[TRAINING.md](TRAINING.md)** → **[EVALUATION.md](EVALUATION.md)**)
* **Understanding Research & Methodology:** Read **[REPORT.md](REPORT.md)** to follow a formal narrative, or start with **[BLOG.md](BLOG.md)** if you prefer an informal succinct read.

## Quick Start

To get started with the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mistral-7b-nli.git
   cd mistral-7b-nli
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t mistral-nli-ft .
   ```

3. **Download the augmented datasets**:
   ```bash
   docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/download_data.py
   ```

4. **Train a model**:
   ```bash
   ./run_training.sh --config train/configs/quick_test.py
   ```

5. **Evaluate your model**:
   ```bash
   ./run_inference.sh --model models/nlistral-ablation1 --data data/original_data/test.csv
   ```

## Docker Usage & Environment Setup

This project uses Docker to ensure reproducibility, particularly for GPU-intensive operations:

- **Data Augmentation**: The thought generation and reflection scripts (`scripts/generate_thoughts.py`, `scripts/generate_thoughts_reflected.py`, etc.) are **computationally lightweight** and can be run directly on your local machine using Python. This is the **recommended approach** for the data preparation phase as it avoids Docker overhead for API-based operations.

- **Training & Inference**: All training, fine-tuning, and model inference operations require specific GPU libraries and dependencies. For these operations, **using Docker is strongly recommended** to ensure compatibility and reproducibility across different hardware environments. The `run_training.sh` and `run_inference.sh` scripts are specifically designed to work with the Docker container.

You can choose the appropriate approach based on which part of the pipeline you're working with:

```bash
# For data augmentation (local Python recommended)
python3 scripts/generate_thoughts.py --api mistral --input-csv data/original_data/train.csv --output-json data/original_thoughts/train_thoughts.json

# For training and inference (Docker required)
./run_training.sh --config train/configs/quick_test.py
./run_inference.sh --model models/nlistral-ablation0 --data data/original_data/sample.csv
```

## Project Highlights

- **Chain-of-Thought Reasoning**: Models are trained to generate step-by-step reasoning along with the final classification.
- **Data Augmentation Pipeline**: Multi-stage pipeline with reflection on incorrect examples to improve training data quality.
- **Parameter-Efficient Training**: QLoRA fine-tuning enables training on consumer GPUs.
- **Configurable Experiments**: Python-based configuration system for easily defining and running experiments.
- **Optimized Inference**: Quantized models and batch processing for efficient evaluation.

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with at least 16GB VRAM
- Hugging Face account/token (for downloading datasets and models)


## Acknowledgments

- The Mistral AI team for releasing the Mistral-7B model
- Hugging Face for their transformers, PEFT, and TRL libraries
- The Chain-of-Thought paper authors


***Generative AI Disclaimer**: AI Tool(s) were used to aid in iterative development of this solution, as well as mass code refactoring, modularisation, visualizations and development & maintenance of documentation. Without the use of Generative AI as a tool, I would not have been able to have iterated through all the cycles of my solution in time given the constraints. Generative AI was not used to dictate or steer my solution, but rather steered with intent from my own ideas and research interests.*