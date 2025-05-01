# Use an official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Install essential system packages and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.py

# Set default environment variables (optional)
# ENV WANDB_PROJECT=mistral-nli-sft
# ENV HF_HOME=/app/huggingface_cache # Optional: Define cache dir inside container

# (Optional) Define entrypoint or default command
# ENTRYPOINT ["python", "scripts/run_sft.py"]
# CMD ["--help"] # Default command if no args provided to `docker run`

# Expose port if needed (e.g., for TensorBoard or a web UI, not strictly necessary for training)
# EXPOSE 6006 