#!/bin/bash

# Create necessary directories
mkdir -p results
mkdir -p data/sample

# Create a sample of the test data (100 rows) if it doesn't exist
if [ ! -f data/sample/test_sample.csv ]; then
  echo "Creating a sample of 100 rows from the test data..."
  docker run --rm \
    -v $(pwd):/app \
    -w /app \
    mistral-nli-ft \
    python -c "import pandas as pd; \
    df = pd.read_csv('data/original_data/test.csv'); \
    sampled = df.sample(100, random_state=42); \
    sampled.to_csv('data/sample/test_sample.csv', index=False); \
    print(f'Created sample with {len(sampled)} rows')"
else
  echo "Sample file data/sample/test_sample.csv already exists."
fi

# Set parameters
MODEL_PATH="mistralai/Mistral-7B-v0.3"  # Path to the BASE model
BATCH_SIZE=16  # Reduced batch size for quicker processing
MAX_LENGTH=512
SAVE_EVERY=1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/base-sample-${TIMESTAMP}.json" # Changed output file name

echo "=== Base Mistral 7B NLI Quick Sample Test ===" # Changed title
echo "Model: $MODEL_PATH"
echo "Sample size: 100 examples"
echo "Batch size: $BATCH_SIZE"
echo "Output file: $OUTPUT_FILE"

# Run inference on GPU 0 with Chain-of-Thought reasoning enabled
echo "Starting inference on sample data using BASE model..." # Changed description
# Note: Removed the -v $(pwd)/models:/app/models mapping for the base model
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  -w /app \
  mistral-nli-ft \
  python sample_model.py \
  --model_id "$MODEL_PATH" \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --test_file "data/sample/test_sample.csv" \
  --output_file "$OUTPUT_FILE" \
  --save_every $SAVE_EVERY \
  --gpu_id 0 \
  --use_cot # Keeping CoT prompt format for comparison

RESULT=$?
if [ $RESULT -ne 0 ]; then
  echo "Error: Sample inference failed with exit code $RESULT"
  exit 1
fi

echo "=== Sample Inference Completed! ==="
echo "Results saved to: $OUTPUT_FILE"
echo "Total runtime: $SECONDS seconds"

# Kill logic remains the same
if [ "$1" = "kill-previous" ]; then
  echo "Attempting to kill the previous full inference job..."
  ps aux | grep "sample_model.py" | grep -v grep | awk '{print $2}' | xargs -r kill
  echo "Done. Check if the job was successfully terminated."
fi 