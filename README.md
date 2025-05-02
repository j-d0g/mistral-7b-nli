p# Mistral-7b Fine-Tuning for NLI with Chain-of-Thought

This project focuses on fine-tuning the Mistral-7B language model for Natural Language Inference (NLI) tasks, specifically using Chain-of-Thought (CoT) reasoning to improve classification performance and interpretability.

## Project Goal

The primary objective is to instruction-tune Mistral-7B using a custom NLI dataset augmented with CoT reasoning. The final model should accurately classify premise-hypothesis pairs as either entailment (1) or no-entailment (0), maximizing performance on a hidden test set.

## Data

The core data is organized as follows:

*   `data/original_data/`: Original NLI premise-hypothesis pairs with labels (train.csv, dev.csv, test.csv).
*   `data/original_thoughts/`: JSON Lines files containing examples augmented with Chain-of-Thought (`thought_process`) and the model's original `predicted_label`, generated using `scripts/generate_thoughts.py`.
*   `data/reflected_thoughts/`: Contains reflection data for examples where the model prediction was incorrect, generated using `scripts/generate_thoughts_reflected.py`.
*   `data/finetune/`: Prepared data for fine-tuning in the instruction format expected by the SFT trainer.
*   `data/sample/`: Contains small sample datasets (including demo.csv) for quick testing.

> **Note:** All scripts default to using sample data if no specific paths are provided. This prevents accidental overwriting of important data during testing.

### Downloading the Datasets

We provide a convenient script to download all the necessary datasets from Hugging Face:

```bash
# Method 1: Use Docker (recommended)
# First build the Docker image if you haven't already
docker build -t mistral-nli-ft .

# Download datasets through Docker
docker run --rm -v $(pwd):/app -w /app mistral-nli-ft python3 data/download_data.py

# Method 2: Set up a virtual environment (if not using Docker)
python3 -m venv venv
source venv/bin/activate
pip install requests tqdm python-dotenv
cd data
python3 download_data.py
```

The script will:
1. Download all necessary CSV and JSONL files from the specified HF repository
2. Organize them in the appropriate directories
3. Run the data preparation script to generate the fine-tuning datasets

For private repositories, add your HF token to a `.env` file or pass it directly with `--token YOUR_TOKEN`.

See `data/README.md` for more details on the dataset structure and formats.

## Optimized Inference

We've optimized the inference process to efficiently process the 1977-sample NLI test set:

### Optimization Techniques

1. **4-bit Quantization**: Using `bitsandbytes` for efficient memory usage
2. **Batch Processing**: Optimized batch sizes for maximizing throughput
3. **Sequence Length Reduction**: Reduced from 2048 → 512 tokens (with actual inputs ranging from 212-465 tokens)
4. **Checkpoint System**: Saves progress every batch and can resume from interruptions
5. **Consistent Prompting**: Uses the exact prompt format from fine-tuning for optimal results
6. **Flexible Data Handling**: Automatically detects labeled vs. unlabeled datasets and adapts behavior

### Performance Gains

* Initial runtime estimate: ~2hr42min with batch size 8
* Optimized runtime: ~50min with batch size 32 (>3x speedup)
* GPU Memory Usage: 15.7GB/24GB VRAM (efficient usage while maintaining performance)

### Running Unified Inference

Our consolidated inference script (`evaluate/run_inference.sh`) handles all inference scenarios through a simple parameter system. The script automatically detects whether your dataset has labels and adjusts the output accordingly.

Example (basic usage from the repository root):
```bash
# Run with default parameters (demo dataset and default model)
./evaluate/run_inference.sh

# Run with a specific model and dataset
./evaluate/run_inference.sh --model models/mistral-thinking-abl0 --data data/original_data/test.csv

# Run with a specific checkpoint
./evaluate/run_inference.sh --model models/mistral-thinking-abl0/checkpoint-2000

# Use a specific GPU
./evaluate/run_inference.sh --gpu 1 
```

This unified approach:
- Eliminates script duplication (formerly multiple test_*.sh scripts)
- Creates consistent, descriptively-named output files
- Uses the same implementation for all scenarios
- Provides better error handling and user feedback

## Training with Configurations

We've implemented a config-based training system inspired by Karpathy's NanoGPT approach:

```bash
# Run with the default configuration
./train.sh

# Use a specific configuration
./train.sh --config train/configs/ablation1.py

# Use ablation2 configuration on gpu 1
./train.sh --config train/configs/ablation2.py --gpu 1

# Override specific parameters
./train.sh --config train/configs/ablation1.py --batch_size 8 --no_wandb
```

### Benefits of the Configuration System:

* **Declarative Configuration**: Parameters defined in clean Python files instead of bash scripts
* **Intuitive Overrides**: Command-line parameters take precedence over config values
* **Reduced Duplication**: Default values exist in only one place
* **Self-Documenting Code**: Config files include comments and explanations
* **Simplified Experimentation**: Create new configurations by copying and editing files
* **Better Type Handling**: Proper typing of parameters (integers, floats, booleans)

### Understanding the Training System

The training system is composed of:

1. **train.sh**: A minimal Docker wrapper script that passes arguments to the Python code
2. **train/train_sft.py**: The main script that loads config and handles training
3. **train/config_loader.py**: A utility for loading Python configuration files
4. **train/configs/default.py**: Default configuration values for all training runs

This design separates infrastructure concerns (Docker, environment) from application logic (training parameters, model configuration).

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
*   **Hyperparameters:** 3-5 epochs (with early stopping patience 3 based on eval loss), LR `2e-4` (linear decay, 3% warmup), AdamW optimizer (WD `0.01`), effective batch size 64, `bf16` precision.

#### Training Ablations:

1.  **Ablation 1 (Correct Only):** Train on examples where the original model's prediction was correct.
2.  **Ablation 2 (Reflected Thought Process - *Primary Goal*):**
    *   Incorporate thought process reflections for originally incorrect examples with a stronger model.
    *   Combine high-quality reflected examples with original correct examples.
    *   Implemented in `scripts/prepare_ft_data.py`.
3.  **Ablation 3 (Unmodified):** Train on all original examples, regardless of correctness.

### 5. Docker Setup

*   A `Dockerfile` is provided to build a container image with all necessary dependencies (PyTorch, CUDA, Transformers, PEFT, TRL, bitsandbytes, wandb, etc.).
*   Training and inference are executed within this Docker container on a remote workstation with GPUs.
*   The train.sh script automatically handles mounting volumes, GPU selection, and running the Python code within the container.
*   `requirements.txt` lists the Python dependencies installed in the Docker image.

Before running training or inference, build the Docker image:
```bash
docker build -t mistral-nli-ft .
```

### 6. Evaluation

*   Primary evaluation metric is accuracy on the hidden test set.
*   During training, validation loss is monitored for early stopping and checkpointing.
*   Validation accuracy, precision, recall, and F1 can also be tracked if a `compute_metrics` function is added to `run_sft.py`.

## Hugging Face Hub Repository

Fine-tuned model checkpoints from various ablation runs are available on the Hugging Face Hub:

[**jd0g/Mistral-v0.3-Thinking_NLI**](https://huggingface.co/jd0g/Mistral-v0.3-Thinking_NLI)

The Hub repository's README provides details on the available checkpoints and how to load them.

## Downloading Model Checkpoints

We provide convenient scripts to download the fine-tuned Mistral-7B NLI model from the Hugging Face repository:

1. **Using the download_model.py script (recommended)**:
   This script will download only the essential adapter files needed for inference from the Mistral_Thinking_Abl2 checkpoint.

   ```bash
   # Navigate to the project root
   cd /path/to/mistral-7b-nli
   
   # Create the models directory if it doesn't exist
   mkdir -p models
   
   # Run the download script
   python download_model.py
   ```
   
   The script will download the model files to `models/mistral_thinking_abl2/` directory.

2. **Using the Docker-based download_checkpoints.sh script**:
   For downloading the complete model with all checkpoints:

   > **Important Note**: The Hugging Face repository is **private** and requires authentication. You must have access to the repository and provide a valid Hugging Face token to download the model.

   ```bash
   # Set your Hugging Face token as an environment variable
   export HF_TOKEN=your_hugging_face_token_here

   # Download the model
   ./models/download_checkpoints.sh

   # Or provide the token directly
   ./models/download_checkpoints.sh --token your_hugging_face_token_here
   ```

## Repository Structure

```
.
├── Dockerfile
├── requirements.txt
├── prompts.py                  # Centralized prompt template definitions
├── train.sh                    # Main wrapper script for training with configs
├── train/                      # Training components
│   ├── train_sft.py            # Main training implementation
│   ├── config_loader.py        # Utility for loading config files
│   └── configs/                # Python-based training configurations
│       ├── default.py          # Base configuration for all training runs
│       ├── initial_test_run.py # Configuration for initial test run
│       ├── ablation1.py        # Configuration for Ablation 1 experiment
│       └── ablation2.py        # Configuration for Ablation 2 experiment
├── evaluate/                   # Evaluation components
│   ├── run_inference.sh        # Unified inference script with parameters
│   ├── sample_model.py         # Core inference implementation
│   └── README_INFERENCE.md     # Documentation for inference
├── data/
│   ├── original_data/          # Original NLI datasets (CSV)
│   ├── original_thoughts/      # Original model thought processes (JSON)
│   ├── reflected_thoughts/     # Reflected thought processes for incorrect examples (JSON)
│   ├── sample/                 # Small sample datasets for quick testing
│   └── finetune/               # Prepared data for fine-tuning (JSONL)
├── models/                     # Directory to store trained models/adapters
│   ├── download_models.py      # Script to download model checkpoints
│   ├── download_checkpoints.sh # Wrapper script for download_models.py
│   └── README.md               # Documentation for using the checkpoints
├── results/                    # Inference results (predictions, metrics)
└── scripts/                    # Data preparation and utility scripts
    ├── generate_thoughts.py              # Script for augmenting original dataset with CoT data
    ├── generate_thoughts_reflected.py    # Script for generating reflected CoT data
    ├── prepare_ft_data.py                # Script to prepare Ablation 2 data
    └── prepare_finetuning_data.py        # General script to format data for SFT
```