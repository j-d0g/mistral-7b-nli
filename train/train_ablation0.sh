#!/bin/bash

# Ensure necessary directories exist
mkdir -p hf_cache
mkdir -p models/mistral-7b-nli-cot-ablation0-pt2

# Make the training script executable
chmod +x scripts/train_sft.py

echo "Running Mistral-7B fine-tuning for Chain-of-Thought NLI on GPU 1..."

# Run the Docker container with the training script on GPU 1 only
docker run --rm --gpus '"device=1"' \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  --env-file .env \
  mistral-nli-ft \
  bash -c "pip install wandb && python scripts/train_sft.py --num_epochs 2"

echo "Training script execution finished." 