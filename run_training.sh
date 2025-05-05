#!/bin/bash

# Wrapper to run Mistral-7B NLI fine-tuning inside Docker
# Handles GPU selection, Docker setup, and passes arguments to train/train_sft.py

# Ensure Hugging Face cache directory exists
HF_CACHE_DIR="hf_cache"
mkdir -p "${HF_CACHE_DIR}"
echo "Using local cache directory: ${HF_CACHE_DIR}"

# Parse GPU options
# Default: Use GPU 0 if no options are provided
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
    -v "$(pwd)/${HF_CACHE_DIR}:/root/.cache/huggingface" \
    --env-file .env \
    mistral-nli-ft \
    torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) train/train_sft.py --distributed_training True "$@"
elif [[ "$GPU_IDS" == *","* ]]; then
  # Multiple specific GPUs with distributed training
  echo "Using GPUs: $GPU_IDS for distributed training"
  NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
  docker run --rm \
    --gpus all \
    -e CUDA_VISIBLE_DEVICES=$GPU_IDS \
    -v "$(pwd):/app" \
    -v "$(pwd)/${HF_CACHE_DIR}:/root/.cache/huggingface" \
    --env-file .env \
    mistral-nli-ft \
    torchrun --nproc_per_node=$NUM_GPUS train/train_sft.py --distributed_training True "$@"
else
  # Single GPU (defaulting to GPU 0 if no --gpus specified), no distributed training
  echo "Using single GPU: $GPU_IDS"
  docker run --rm \
    --gpus "device=$GPU_IDS" \
    -v "$(pwd):/app" \
    -v "$(pwd)/${HF_CACHE_DIR}:/root/.cache/huggingface" \
    --env-file .env \
    mistral-nli-ft \
    python3 train/train_sft.py "$@"
fi 