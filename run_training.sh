#!/bin/bash

# Wrapper to run Mistral-7B NLI fine-tuning inside Docker
# Passes all arguments directly to the Python training script

# Parse GPU options
USE_ALL_GPUS=false
GPU_IDS="0"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --all-gpus)
      USE_ALL_GPUS=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

if [ "$USE_ALL_GPUS" = true ]; then
  # Use all available GPUs with distributed training
  echo "Using all available GPUs"
  docker run --rm \
    --gpus all \
    -v "$(pwd):/app" \
    -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
    --env-file .env \
    mistral-nli-ft \
    torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) train/train_sft.py --distributed_training True "$@"
elif [[ "$GPU_IDS" == *","* ]]; then
  # Multiple specific GPUs with distributed training
  echo "Using GPUs: $GPU_IDS"
  docker run --rm \
    --gpus "device=$GPU_IDS" \
    -v "$(pwd):/app" \
    -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
    --env-file .env \
    mistral-nli-ft \
    torchrun --nproc_per_node=$(echo $GPU_IDS | tr ',' '\n' | wc -l) train/train_sft.py --distributed_training True "$@"
else
  # Single GPU, no distributed training
  echo "Using GPU: $GPU_IDS"
  docker run --rm \
    --gpus "device=$GPU_IDS" \
    -v "$(pwd):/app" \
    -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
    --env-file .env \
    mistral-nli-ft \
    python3 train/train_sft.py "$@"
fi 