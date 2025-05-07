#!/bin/bash

# Script to start Jupyter Notebook server inside Docker with GPU access

# Build the Docker image if it doesn't exist (optional, you can comment this out if you already built it)
# docker build -t mistral-nli-ft .

# Set default GPU to use
GPU_ID=${1:-0}

echo "Starting Jupyter Notebook server in Docker with GPU $GPU_ID..."
echo "Once the server starts, copy the URL with the token and open it in your browser."

# Run Docker container with:
# - GPU access
# - Port 8888 exposed for Jupyter
# - Current directory mounted as /app
# - Environment variables from .env
docker run --rm \
  --gpus device=$GPU_ID \
  -p 8888:8888 \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -w /app \
  --env-file .env \
  mistral-nli-ft \
  jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

echo "Jupyter server stopped." 