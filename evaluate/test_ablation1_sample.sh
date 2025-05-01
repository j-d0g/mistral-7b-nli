#!/bin/bash

# Create necessary directories
mkdir -p results
mkdir -p data/sample

# Create a sample of the test data (100 rows)
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

# Set parameters - using the checkpoint directory
CHECKPOINT_PATH="models/mistral-7b-nli-cot-ablation1/checkpoint-1250"  # Path to the checkpoint
BATCH_SIZE=16  # Reduced batch size for quicker processing
MAX_LENGTH=512
SAVE_EVERY=1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/checkpoint-1250-sample-${TIMESTAMP}.json"

echo "=== Checkpoint 1250 Quick Sample Test ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Sample size: 100 examples"
echo "Batch size: $BATCH_SIZE"
echo "Output file: $OUTPUT_FILE"

# Run inference on GPU 0 with Chain-of-Thought reasoning enabled
echo "Starting inference on sample data..."
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  -w /app \
  mistral-nli-ft \
  python sample_model.py \
  --model_id "$CHECKPOINT_PATH" \
  --batch_size $BATCH_SIZE \
  --max_length $MAX_LENGTH \
  --test_file "data/sample/test_sample.csv" \
  --output_file "$OUTPUT_FILE" \
  --save_every $SAVE_EVERY \
  --gpu_id 0 \
  --use_cot

RESULT=$?
if [ $RESULT -ne 0 ]; then
  echo "Error: Sample inference failed with exit code $RESULT"
  exit 1
fi

echo "=== Sample Inference Completed! ==="
echo "Results saved to: $OUTPUT_FILE"
echo "Total runtime: $SECONDS seconds" 