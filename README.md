# Mistral-7b Fine-Tuning for NLI with Chain-of-Thought

This project focuses on fine-tuning the Mistral-7B language model for Natural Language Inference (NLI) tasks, specifically using Chain-of-Thought (CoT) reasoning to improve classification performance and interpretability.

## Project Overview

The primary objective is to instruction-tune Mistral-7B using a custom NLI dataset augmented with CoT reasoning. The trained model can accurately classify premise-hypothesis pairs as either entailment (1) or no-entailment (0), while providing interpretable reasoning.

## Documentation Directions

This repository is split into three parts:

1. **[DATA.md](DATA.md)** - Synthetic Chain-of-Thought augmentation of NLI Dataset
2. **[TRAINING.md](TRAINING.md)** - Model training and fine-tuning
3. **[EVALUATION.md](EVALUATION.md)** - Model evaluation and inference

Each document includes both a Quick Start guide for getting up and running quickly, as well as a Deep Dive section with technical details.

Additional documentation:

* **[README.md](README.md)** - Project overview and setup instructions
* **[REPORT.md](REPORT.md)** - Write-up on methodology and results
* **[BLOG.md](BLOG.md)** - Chronological narrative of the experimental journey

### Key Directories

* **data/** - Datasets and processing scripts
* **train/** - Training implementation and configs
* **evaluate/** - Inference and metrics code
* **scripts/** - Supporting utilities for data processing and analysis
* **results/** - Outputs from model evaluation
* **metrics/** - Visualization outputs and performance metrics
* **models/** - Storage for trained model checkpoints

### Recommended Reading Path

* **First-time users:** Start with README.md, then follow the three core pillars in order (DATA → TRAINING → EVALUATION)
* **Understanding results:** Start with EVALUATION.md, then explore REPORT.md for deeper analysis
* **Understanding methodology:** Read BLOG.md for the narrative journey, then REPORT.md for formalized approach

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

3. **Download the datasets**:
   ```bash
   docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/download_data.py
   ```

4. **Train a model**:
   ```bash
   ./run_training.sh --config train/configs/sample_test.py
   ```

5. **Evaluate your model**:
   ```bash
   ./run_inference.sh --model models/mistral-thinking-sample-test --data data/original_data/test.csv
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
./run_training.sh --config train/configs/sample_test.py
./run_inference.sh --model models/mistral-thinking-sample-test --data data/sample/demo.csv
```

## Repository Structure

```
.
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
├── DATA.md                      # Dataset documentation
├── TRAINING.md                  # Training documentation
├── EVALUATION.md                # Evaluation documentation
├── BLOG.md                      # Experimental journey narrative
├── HUB_MODEL.md                 # Model card documentation
├── HUB_DATASET.md               # Dataset card documentation
├── REPORT.md                     # Research findings and methodology
├── prompts.py                   # Centralized prompt templates
├── run_training.sh              # Training wrapper script
├── run_inference.sh             # Inference wrapper script
├── run_metrics.sh               # Metrics generation script
├── data/                        # Dataset files and scripts
│   ├── original_data/           # Original NLI datasets (CSV)
│   ├── original_thoughts/       # Generated thought processes
│   ├── reflected_thoughts/      # Improved reasoning for incorrect examples
│   ├── finetune/                # Formatted training data (JSONL)
│   └── download_data.py         # Dataset download script
├── train/                       # Training components
│   ├── train_sft.py             # Main training implementation
│   ├── config_loader.py         # Configuration loading utility
│   └── configs/                 # Training configurations
├── evaluate/                    # Evaluation components
│   └── sample_model.py          # Model sampling implementation
├── scripts/                     # Data preparation scripts
│   ├── generate_thoughts.py     # Generate CoT reasoning
│   ├── generate_thoughts_reflected.py # Generate reflections
│   ├── prepare_ft_data.py       # Prepare fine-tuning data
│   └── analysis/                # Data analysis & visualization
├── models/                      # Model storage
│   └── download_model.py        # Model download script
├── results/                     # Evaluation results storage
├── metrics/                     # Visualization outputs and metrics
├── figures/                     # Diagrams and visualizations
├── utils/                       # Utility functions and helpers
├── logs/                        # Log files from training runs
├── tests/                       # Test files for the project
├── wandb/                       # Weights & Biases logging data
├── llm/                         # LLM interface utilities
└── service/                     # API and service implementations
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Mistral AI team for releasing the Mistral-7B model
- Hugging Face for their transformers, PEFT, and TRL libraries
- The Chain-of-Thought paper authors

*Generative AI Disclaimer: AI Tools were used to aid in iterative development of this solution, as well as mass code refactoring, modularisation, visualizations and development & maintenance of documentation. Without the use of Generative AI as a tool, I would not have been able to have iterated through all the cycles of my solution in time given the constraints. Generative AI was not used to dictate or steer my solution, but rather steered with intent from my own ideas and research interests.*