# Mistral 7B NLI Inference

This repository contains scripts for running inference on 4-bit quantized Mistral v0.3 models from HuggingFace on an NLI (Natural Language Inference) task with 1977 test samples. The solution is optimized for running on an NVIDIA RTX 4090 GPU.

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

### Standard Inference (without Chain-of-Thought)

```bash
./run_inference.sh
```

This will run inference using the base Mistral-7B-v0.3 model, producing direct classification results saved to `results/mistral-v0.3-base-predictions.json`.

### Chain-of-Thought (CoT) Inference

```bash
./run_cot_inference.sh
```

This enables Chain-of-Thought reasoning, which generates detailed reasoning paths before making predictions. Results are saved to `results/mistral-v0.3-cot-predictions.json`.

### Fine-tuned Model Inference

To run inference with a custom fine-tuned model from HuggingFace:

```bash
./run_finetuned_inference.sh "huggingface_model_id_or_path" [cot]
```

Examples:

```bash
# Run standard inference with a fine-tuned model
./run_finetuned_inference.sh "your-username/mistral-7b-nli-finetuned"

# Run CoT inference with a fine-tuned model
./run_finetuned_inference.sh "your-username/mistral-7b-nli-finetuned" cot
```

## Advanced Configuration

For more advanced configurations, you can directly use the Python script:

```bash
docker-compose run --rm mistral-nli-inference python run_nli_inference.py \
  --model_id "mistralai/Mistral-7B-v0.3" \
  --batch_size 8 \
  --max_length 2048 \
  --test_file "data/original_data/test.csv" \
  --output_file "results/custom-predictions.json" \
  --use_cot
```

### Available arguments:

- `--model_id`: HuggingFace model ID or local path
- `--test_file`: Path to the test CSV file
- `--output_file`: Path to save predictions
- `--batch_size`: Batch size for inference (default: 8)
- `--max_length`: Maximum sequence length (default: 2048)
- `--use_cot`: Enable Chain-of-Thought reasoning

## Understanding Results

The output JSON files contain:

- `model`: Name of the model used
- `accuracy`: Overall accuracy on the test set
- `inference_time_seconds`: Total time for inference
- `samples_per_second`: Processing throughput
- `use_cot`: Whether Chain-of-Thought reasoning was used
- `results`: List of all sample predictions with:
  - `premise`: The input premise
  - `hypothesis`: The input hypothesis
  - `true_label`: The ground truth (0 or 1)
  - `predicted_label`: The model's prediction (0 or 1)
  - `correct`: Whether the prediction was correct
  - `output`: The model's raw output text

## Troubleshooting

1. **Out of Memory Errors**: Reduce the batch size to 4 or 2 if you encounter GPU memory issues
2. **Slow inference**: Make sure you've set up proper GPU acceleration for Docker
3. **JSON parsing errors**: Adjust the extraction logic in `extract_prediction()` function if the model's output format changes 