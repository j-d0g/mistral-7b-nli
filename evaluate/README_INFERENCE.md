# Mistral-7B NLI Inference

This guide explains how to run inference on fine-tuned Mistral-7B models for Natural Language Inference (NLI) tasks.

## Requirements

- NVIDIA GPU with 12GB+ VRAM
- Docker
- 16GB+ system RAM

## Running Inference

We provide a unified inference script that handles all inference scenarios through a simple parameter system.

### Basic Usage

```bash
# Run with default parameters (demo dataset and default model)
./run_inference.sh

# Run with a specific model and dataset
./run_inference.sh --model models/Mistral_Thinking_Abl0 --data data/original_data/test.csv

# Run with a specific checkpoint
./run_inference.sh --model models/Mistral_Thinking_Abl2/checkpoint-2000

# Use a specific GPU
./run_inference.sh --gpu 1
```

### Available Parameters

- `--model`, `-m`: Path to the model (default: models/mistral-7b-nli-cot)
- `--data`, `-d`: Path to the test data CSV (default: data/sample/demo.csv)
- `--gpu`, `-g`: GPU ID to use (default: 0)
- `--help`, `-h`: Show help message

### Output Files

The script automatically generates descriptive filenames based on the model and dataset:
- JSON output: `results/[model-name]-[dataset-name]-[timestamp].json`
- CSV output: `results/[model-name]-[dataset-name]-[timestamp].csv`

## Understanding Results

The output JSON files contain:

- `model`: Name of the model used
- `accuracy`: Overall accuracy on the test set (only when labels are present)
- `inference_time_seconds`: Total time for inference
- `samples_per_second`: Processing throughput
- `use_cot`: Whether Chain-of-Thought reasoning was used
- `results`: List of all sample predictions with details for each example

## Handling Different Datasets

The inference system automatically detects whether your dataset has labels:

- **With labels**: Computes and reports accuracy metrics
- **Without labels**: Generates predictions without performance metrics

## Advanced Configuration

For more advanced configurations, you can directly invoke the `evaluate/sample_model.py` script:

```bash
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -w /app \
  mistral-nli-ft \
  python evaluate/sample_model.py \
  --model_id "models/Mistral_Thinking_Abl2/checkpoint-2000" \
  --test_file "data/original_data/test.csv" \
  --output_file "results/custom-predictions.json" \
  --batch_size 32 \
  --max_length 512 \
  --use_cot
```

## Troubleshooting

1. **Out of Memory Errors**: Reduce the batch size if you encounter GPU memory issues
2. **Slow inference**: Make sure you've set up proper GPU acceleration for Docker
3. **Missing files**: Check that both the model and data files exist at the specified paths 