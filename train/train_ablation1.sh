#!/bin/bash

# Ensure necessary directories exist
mkdir -p hf_cache
mkdir -p models/mistral-7b-nli-cot-ablation1

# Make the training script executable
chmod +x scripts/train_sft.py

echo "Running Mistral-7B fine-tuning for Chain-of-Thought NLI (Ablation 1: Effective Batch Size 32)..."

# Run the Docker container with the training script on GPU 1 only
docker run --rm --gpus '"device=1"' \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  --env-file .env \
  mistral-nli-ft \
  bash -c "pip install wandb && python scripts/train_sft.py \
    --output_dir models/mistral-7b-nli-cot-ablation1 \
    --batch_size 16 \
    --grad_accumulation_steps 2 \
    --warmup_ratio 0.05 \
    --num_epochs 3"

echo "Training script execution finished." 