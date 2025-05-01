#!/bin/bash

# Create necessary directories
mkdir -p results

# Set parameters
MODEL_PATH="models/mistral-7b-nli-cot"  # Path to the fine-tuned model
BATCH_SIZE=32
MAX_LENGTH=512
SAVE_EVERY=1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/finetuned-mistral-cot-${TIMESTAMP}.json"

echo "=== Fine-tuned Mistral 7B NLI Inference with Chain-of-Thought ==="
echo "Model: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Max length: $MAX_LENGTH"
echo "Timestamp: $TIMESTAMP"
echo "Output file: $OUTPUT_FILE"

# Run inference on GPU 0 with Chain-of-Thought reasoning enabled
echo "Starting inference on GPU 0 with batch size $BATCH_SIZE and CoT reasoning"
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
  --test_file "data/original_data/test.csv" \
  --output_file "$OUTPUT_FILE" \
  --save_every $SAVE_EVERY \
  --gpu_id 0 \
  --use_cot

RESULT=$?
if [ $RESULT -ne 0 ]; then
  echo "Error: Inference failed with exit code $RESULT"
  exit 1
fi

echo "=== Inference Completed Successfully! ==="
echo "Results saved to: $OUTPUT_FILE"
echo "Total runtime: $SECONDS seconds" 