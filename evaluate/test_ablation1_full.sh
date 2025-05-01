#!/bin/bash

# Script to run inference on the full test set using checkpoint-1250

# Set parameters
CHECKPOINT_PATH="models/mistral-7b-nli-cot-ablation1/checkpoint-1250"  # Path to the checkpoint
BATCH_SIZE=16  # Reduced batch size for quicker processing
MAX_LENGTH=512
SAVE_EVERY=50
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="results/checkpoint-1250-fulltest-${TIMESTAMP}.json"

# Run inference on GPU 0 with Chain-of-Thought reasoning enabled
echo "Starting inference on full test set using ${CHECKPOINT_PATH}..."
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/models:/app/models \
  -w /app \
  mistral-nli-ft \
  python sample_model.py \
  --model_id ${CHECKPOINT_PATH} \
  --test_file data/original_data/test.csv \
  --output_file ${OUTPUT_FILE} \
  --batch_size ${BATCH_SIZE} \
  --max_length ${MAX_LENGTH} \
  --save_every ${SAVE_EVERY} \
  --use_cot

echo "Inference completed. Results saved to ${OUTPUT_FILE}"
echo "Now applying improved extraction to the results..."

# Apply the improved extraction logic to get accurate predictions
docker run --rm --gpus device=0 \
  -v $(pwd):/app \
  -w /app \
  mistral-nli-ft \
  python fix_predictions_with_tracking.py ${OUTPUT_FILE} --output_file "results/fixed-${OUTPUT_FILE##*/}"

echo "Done! Fixed results saved to results/fixed-${OUTPUT_FILE##*/}" 