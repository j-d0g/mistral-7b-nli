# Mistral-7b Fine-Tuning for NLI with Chain-of-Thought

This project focuses on fine-tuning the Mistral-7B language model for Natural Language Inference (NLI) tasks, specifically using Chain-of-Thought (CoT) reasoning to improve classification performance and interpretability.

## Project Overview

The primary objective is to instruction-tune Mistral-7B using a custom NLI dataset augmented with CoT reasoning. The trained model can accurately classify premise-hypothesis pairs as either entailment (1) or no-entailment (0), while providing interpretable reasoning.

## Documentation

We provide comprehensive documentation for each stage of the project:

* [DATA.md](DATA.md) - Preparing and understanding the datasets
* [TRAINING.md](TRAINING.md) - Fine-tuning the model
* [EVALUATION.md](EVALUATION.md) - Evaluating model performance

Each document includes both a Quick Start guide for getting up and running quickly, as well as a Deep Dive section with technical details.

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
├── PAPER.md                     # Research findings and methodology
├── prompts.py                   # Centralized prompt templates
├── run_training.sh              # Training wrapper script
├── run_inference.sh             # Inference wrapper script
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
├── models/                      # Model storage
│   └── download_model.py        # Model download script
├── scripts/                     # Data preparation scripts
│   ├── generate_thoughts.py     # Generate CoT reasoning
│   ├── generate_thoughts_reflected.py # Generate reflections
│   ├── prepare_ft_data.py       # Prepare fine-tuning data
│   └── analysis/                # Data analysis & visualization
│       ├── analyze_token_lengths.py    # Token length analysis
│       ├── generate_card_visualizations.py # Create visualizations
│       └── README.md            # Analysis scripts documentation
└── results/                     # Evaluation results storage
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