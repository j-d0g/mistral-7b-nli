#!/bin/bash

# Simple wrapper to run Mistral-7B NLI fine-tuning inside Docker
# Passes all arguments directly to the Python training script

# Default GPU ID is 0
GPU_ID=0
echo "Using GPU: $GPU_ID"

# Run training in Docker
docker run --rm \
  --gpus "device=$GPU_ID" \
  -v "$(pwd):/app" \
  -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
  --env-file .env \
  mistral-nli-ft \
  python3 train/train_sft.py "$@" 