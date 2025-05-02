#!/bin/bash

# Script to run inference with parameterized options
# This script replaces all the separate test_*.sh scripts

# Fixed parameters
BATCH_SIZE=16
MAX_LENGTH=512
SAVE_EVERY=1
# Always use Chain-of-Thought reasoning
USE_COT=true

# Default parameters that can be changed
MODEL_PATH="models/mistral-7b-nli-cot"
TEST_FILE="data/sample/demo.csv"
GPU_ID=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model|-m)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data|-d)
      TEST_FILE="$2"
      shift 2
      ;;
    --gpu|-g)
      GPU_ID="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model, -m PATH           Path to the model (default: models/mistral-7b-nli-cot)"
      echo "  --data, -d PATH            Path to the test data CSV (default: data/sample/demo.csv)"
      echo "  --gpu, -g ID               GPU ID to use (default: 0)"
      echo "  --help, -h                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Generate basename for output files based on model and data
MODEL_NAME=$(basename "$MODEL_PATH")
DATA_NAME=$(basename "$TEST_FILE" .csv)

# Set up timestamp and output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/${MODEL_NAME}-${DATA_NAME}-${TIMESTAMP}.json"
OUTPUT_CSV="results/${MODEL_NAME}-${DATA_NAME}-${TIMESTAMP}.csv"

# Create necessary directories
mkdir -p results

# Display configuration
echo "=== Mistral 7B NLI Inference ==="
echo "Model: $MODEL_PATH"
echo "Test data: $TEST_FILE"
echo "GPU ID: $GPU_ID"
echo "Output JSON: $OUTPUT_FILE"
echo "Output CSV: $OUTPUT_CSV"
echo "Timestamp: $TIMESTAMP"
echo "Using fixed settings:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max length: $MAX_LENGTH"
echo "  - Chain-of-Thought reasoning: Enabled"
echo "  - Save checkpoint every: $SAVE_EVERY batch"

# Run the inference
echo "Starting inference..."
docker run --rm --gpus device=$GPU_ID \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  -w /app \
  --env-file .env \
  mistral-nli-ft \
  python evaluate/sample_model.py \
  --model_id "$MODEL_PATH" \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --test_file "$TEST_FILE" \
  --output_file "$OUTPUT_FILE" \
  --output_csv "$OUTPUT_CSV" \
  --save_every $SAVE_EVERY \
  --gpu_id $GPU_ID \
  --use_cot

# Check execution status
if [ $? -ne 0 ]; then
  echo "Error: Inference failed"
  exit 1
fi

echo "Inference completed successfully"
echo "Results saved to: $OUTPUT_FILE"
if [ -f "$OUTPUT_CSV" ]; then
  echo "CSV predictions saved to: $OUTPUT_CSV"
fi

echo "Done!" 