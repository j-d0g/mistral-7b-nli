# Mistral 7B NLI Inference

This repository contains scripts for running inference on 4-bit quantized Mistral v0.3 models from HuggingFace on an NLI (Natural Language Inference) task. The solution is optimized for running on an NVIDIA RTX 4090 GPU.

## Requirements

- NVIDIA RTX 4090 GPU
- Docker and Docker Compose
- 16GB+ system RAM
- 100GB+ disk space (for model caching and results)
- Ubuntu 20.04 or newer

## Resource Usage & Performance

- Memory Usage: ~12GB GPU VRAM (4-bit quantized model)
- Estimated inference time: ~20-30 minutes for 1977 samples
- Throughput: ~1-2 samples per second (varies based on prompt complexity)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mistral-7b-nli.git
   cd mistral-7b-nli
   ```

2. Build the Docker image:
   ```bash
   docker-compose build
   ```

## Running Inference

We provide a unified inference script (`run_inference.sh`) that handles all inference scenarios through a simple parameter system. The script automatically detects whether your dataset has labels and adjusts the output accordingly.

### Basic Usage

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

### Available parameters:

- `--model`, `-m`: Path to the model (default: models/mistral-7b-nli-cot)
- `--data`, `-d`: Path to the test data CSV (default: data/sample/demo.csv)
- `--gpu`, `-g`: GPU ID to use (default: 0)
- `--help`, `-h`: Show help message

### Fixed settings (not configurable via command line):

- Batch size: 16
- Max sequence length: 512
- Chain-of-Thought reasoning: Always enabled
- Save checkpoint frequency: Every batch

### Output files

The script automatically generates descriptive filenames based on the model and dataset:
- JSON output: `results/[model-name]-[dataset-name]-[timestamp].json`
- CSV output: `results/[model-name]-[dataset-name]-[timestamp].csv`

## Advanced Configuration

For more advanced configurations, you can directly invoke the `evaluate/sample_model.py` script within the Docker container:

```bash
# Make sure you are in the repository root directory
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  -w /app \
  mistral-nli-ft \
  python evaluate/sample_model.py \
  --model_id "models/mistral-thinking-abl0/checkpoint-2000" \
  --test_file "data/original_data/test.csv" \
  --output_file "results/custom-predictions.json" \
  --output_csv "results/custom-predictions.csv" \
  --batch_size 32 \
  --max_length 512 \
  --use_cot \
  --resume # Optionally resume from a previous checkpoint
```

### Available arguments for sample_model.py:

- `--model_id`: HuggingFace model ID or local path (e.g., a checkpoint directory)
- `--test_file`: Path to the test CSV file (default: `data/original_data/test.csv`)
- `--output_file`: Path to save detailed JSON results (default: `results/predictions.json`)
- `--output_csv`: Optional path to save predictions as a single-column CSV (default: `None`)
- `--batch_size`: Batch size for inference (default: 32)
- `--max_length`: Maximum sequence length (default: 512)
- `--use_cot`: Enable Chain-of-Thought reasoning (flag, default: False)
- `--save_every`: Save checkpoint after this many batches (default: 1)
- `--resume`: Resume from checkpoint if available (flag, default: False)
- `--gpu_id`: GPU ID to use for inference (default: 0)

## Handling Labeled and Unlabeled Data

The inference system automatically detects whether your dataset has a 'label' column:

- **With labels**: Computes and reports accuracy metrics
- **Without labels**: Generates predictions without performance metrics

This allows seamless use with both evaluation data (which has ground truth labels) and new, unseen data (which doesn't have labels).

## Understanding Results

The output JSON files contain:

- `model`: Name of the model used
- `accuracy`: Overall accuracy on the test set (only when labels are present)
- `inference_time_seconds`: Total time for inference
- `samples_per_second`: Processing throughput
- `use_cot`: Whether Chain-of-Thought reasoning was used
- `results`: List of all sample predictions with:
  - `premise`: The input premise
  - `hypothesis`: The input hypothesis
  - `true_label`: The ground truth (0 or 1) - only when labels are present
  - `predicted_label`: The model's prediction (0 or 1)
  - `correct`: Whether the prediction was correct - only when labels are present
  - `output`: The model's raw output text

## Troubleshooting

1. **Out of Memory Errors**: Reduce the batch size in `run_inference.sh` if you encounter GPU memory issues
2. **Slow inference**: Make sure you've set up proper GPU acceleration for Docker
3. **JSON parsing errors**: Adjust the extraction logic in `extract_prediction()` function if the model's output format changes 