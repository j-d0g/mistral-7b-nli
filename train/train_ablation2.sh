#!/bin/bash

# Ensure necessary directories exist
mkdir -p hf_cache
mkdir -p models/mistral-7b-nli-cot-ablation2

# Make the training script executable
chmod +x scripts/train_sft.py

echo "Running Mistral-7B fine-tuning for Chain-of-Thought NLI (Ablation 2: Mixed Data Optimization)..."

# Run the Docker container with the training script on GPU 1 only
docker run --rm --gpus '"device=1"' \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  --env-file .env \
  mistral-nli-ft \
  bash -c "pip install wandb && python scripts/train_sft.py \
    --output_dir models/mistral-7b-nli-cot-ablation2 \
    --batch_size 16 \
    --grad_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --num_epochs 3"

echo "Training script execution finished." 