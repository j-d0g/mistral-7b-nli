# Evaluating NLI Models

This document provides instructions for evaluating NLI models on test datasets, whether you've trained your own models or downloaded pre-trained ones.

## Table of Contents

- [Quick Start](#quick-start)
- [Evaluation Workflow](#evaluation-workflow)
- [Option 1: Evaluating Your Own Trained Models](#option-1-evaluating-your-own-trained-models)
- [Option 2: Evaluating Downloaded Models](#option-2-evaluating-downloaded-models)
- [Understanding Results](#understanding-results)
- [Advanced Options](#advanced-options)

## Quick Start

```bash
# Evaluate a model you trained (on unlabeled data - for predictions only)
./run_inference.sh --model models/mistral-thinking-sample-test --data data/sample/demo.csv

# Evaluate a downloaded model (on labeled data - for accuracy metrics)
./run_inference.sh --model models/mistral-thinking-default-epochs2 --data data/original_data/sample.csv
```

## Evaluation Workflow

The evaluation process follows these steps:

1. **Prepare a model**: Either train your own or download a pre-trained model
2. **Select a test dataset**: A CSV file containing premise-hypothesis pairs (with or without ground truth labels)
3. **Run inference**: Execute the evaluation script to generate predictions and measure performance
4. **Analyze results**: Review the output files in the `results/` directory

> **Important**: The script automatically detects whether your input CSV has a `label` column:
> - **With labels**: The script will calculate and report accuracy metrics
> - **Without labels**: The script will only generate predictions

### Example Dataset Types:
- `data/original_data/sample.csv`: Contains labeled data (includes a `label` column)
- `data/sample/demo.csv`: Contains unlabeled data (no `label` column)

## Option 1: Evaluating Your Own Trained Models

If you've trained your own model following the instructions in [TRAINING.md](TRAINING.md), you can evaluate it directly:

```bash
# Use the model output directory specified in your training config
./run_inference.sh --model models/mistral-thinking-sample-test --data data/original_data/test.csv
```

You can also evaluate a specific checkpoint from your training run:

```bash
./run_inference.sh --model models/mistral-thinking-sample-test/checkpoint-20 --data data/original_data/test.csv
```

## Option 2: Evaluating Downloaded Models

### Downloading Pre-trained Models

The repository includes a script to download pre-trained models from Hugging Face:

```bash
# Download the default model
python models/download_model.py

# Or run with Docker
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 models/download_model.py
```

This will download the model files to the `models/` directory. By default, it downloads the "mistral-thinking-default-epochs2" model, which is the best-performing model.

You can edit the `MODEL_PATHS` list in `models/download_model.py` to download other available models:

```python
MODEL_PATHS = [
    "mistral-thinking-default-epochs2", # Best model
    # "mistral-thinking-abl0",
    # "mistral-thinking-abl0-ext", # Second Best Model
    # "mistral-thinking-abl2", # Third Best Model
    # "mistral-thinking-abl3", # Second Best Model
    # Uncomment to download additional models
]
```

> **Note**: Downloading models requires a Hugging Face token with access to the repository. Add your token to the `.env` file as `HF_TOKEN=your_token_here`.

### Running Evaluation on Downloaded Models

After downloading, evaluate the model using:

```bash
./run_inference.sh --model models/mistral-thinking-default-epochs2 --data data/original_data/test.csv
```

## Understanding Results

The evaluation script generates two output files in the `results/` directory:

1. **JSON file** (`results/[model_name]-[dataset_name]-[timestamp].json`): 
   - **Contains**: 
     - Model configuration
     - Overall accuracy, precision, recall, and F1 score (if input data had labels)
     - Inference time and throughput statistics
     - Per-example results with premise, hypothesis, true label (if available), predicted label, and thought process
   - **Purpose**: Provides detailed information for analysis and debugging

2. **CSV file** (`results/[model_name]-[dataset_name]-[timestamp].csv`):
   - **Contains**:
     - Premise and hypothesis
     - True label (if available in input data)
     - Predicted label
   - **Purpose**: Simplified format for quick review or importing into other tools

### Checkpoint Files

The script also saves checkpoint files during processing (`results/checkpoint_[model_name]-[dataset_name]-[timestamp].json`), which can be useful for debugging or recovering from interruptions.

## Advanced Options

### GPU Selection

By default, the evaluation runs on GPU 0. To use a different GPU:

```bash
./run_inference.sh --model models/mistral-thinking-default-epochs2 --data data/original_data/test.csv --gpu 1
```

### Batch Size

The default batch size is 16. This is set as a fixed parameter in the script and cannot be changed via command-line arguments.

### Running Directly via Python

For more control, you can run the underlying Python script directly:

```bash
docker run --gpus device=0 --rm -v $(pwd):/app -w /app mistral-nli-ft \
    python evaluate/sample_model.py \
    --model_id models/mistral-thinking-default-epochs2 \
    --test_file data/original_data/test.csv \
    --batch_size 16 \
    --use_cot
```

## Further Information

- **Evaluation implementation details**: See [evaluate/README.md](evaluate/README.md)
- **Training details**: See [TRAINING.md](TRAINING.md)
- **Data preparation**: See [data/README.md](data/README.md) 