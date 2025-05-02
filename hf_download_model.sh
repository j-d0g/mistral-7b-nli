#!/bin/bash
# Mistral-7B-NLI Model Downloader with HF Token
#
# This script uses Docker to download the Mistral-7B-NLI model adapter files from Hugging Face,
# passing the HF_TOKEN from the .env file to the Docker container.
#
# The model files will be downloaded to: models/mistral_thinking_abl2/

# Read HF_TOKEN from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep HF_TOKEN)
  if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not found in .env file"
    exit 1
  fi
  echo "Found HF_TOKEN in .env file"
else
  echo "Error: .env file not found"
  exit 1
fi

# Create the models directory if it doesn't exist
mkdir -p models/mistral_thinking_abl2

# Check if the Docker image exists, build if not
if ! docker images | grep -q "mistral-nli-ft"; then
  echo "Building Docker image mistral-nli-ft..."
  docker build -t mistral-nli-ft .
fi

# Run the download script in Docker, passing the HF_TOKEN
echo "Downloading Mistral-7B-NLI model adapter files using Docker..."
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  -e HF_TOKEN=$HF_TOKEN \
  -e SKIP_INFERENCE=1 \
  -e RUNNING_IN_DOCKER=1 \
  mistral-nli-ft \
  python3 models/download_model.py

echo "Done! Model files saved to models/mistral_thinking_abl2/"
echo "To use these files for inference, you'll need to download the base model separately with your HF token."

# Clean up temporary files (may require sudo as they could be owned by root)
if [ -d "temp_download" ]; then
  echo "Cleaning up temporary directory..."
  sudo rm -rf temp_download
  echo "Temporary files removed."
fi 