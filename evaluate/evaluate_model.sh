#!/bin/bash

# Ensure necessary directories exist
mkdir -p results

# Make the evaluation script executable
chmod +x scripts/evaluate_models.py

echo "Starting model evaluations for NLI with Chain-of-Thought..."

# 1. Run evaluation on base Mistral-7B model
echo "Evaluating Base Mistral-7B-v0.3..."
docker run --rm --gpus all \
  -v $(pwd):/app \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  --env-file .env \
  mistral-nli-ft \
  bash -c "pip install scikit-learn tqdm pandas && python scripts/evaluate_models.py \
    --model-type hf \
    --model-path mistralai/Mistral-7B-v0.3 \
    --use-4bit \
    --input-csv data/original_data/test.csv \
    --output-dir results \
    --limit 100"

# Find previous models if they exist (excluding the new ablations)
PREV_MODELS=$(find models -maxdepth 1 -type d -not -name "mistral-7b-nli-cot*" -not -name "models" | head -1)

# 2. Run evaluation on previously fine-tuned model (if available)
if [ -n "$PREV_MODELS" ]; then
  echo "Evaluating Previous Fine-Tuned Model: $PREV_MODELS..."
  docker run --rm --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/hf_cache:/root/.cache/huggingface \
    --env-file .env \
    mistral-nli-ft \
    bash -c "pip install scikit-learn tqdm pandas && python scripts/evaluate_models.py \
      --model-type hf \
      --model-path mistralai/Mistral-7B-v0.3 \
      --adapter-path $PREV_MODELS \
      --use-4bit \
      --input-csv data/original_data/test.csv \
      --output-dir results \
      --limit 100"
else
  echo "No previous fine-tuned models found."
fi

# 3. Run evaluation on Ablation 1 (if training has completed)
if [ -d "models/mistral-7b-nli-cot-ablation1" ]; then
  echo "Evaluating Ablation 1 model..."
  docker run --rm --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/hf_cache:/root/.cache/huggingface \
    --env-file .env \
    mistral-nli-ft \
    bash -c "pip install scikit-learn tqdm pandas && python scripts/evaluate_models.py \
      --model-type hf \
      --model-path mistralai/Mistral-7B-v0.3 \
      --adapter-path models/mistral-7b-nli-cot-ablation1 \
      --use-4bit \
      --input-csv data/original_data/test.csv \
      --output-dir results \
      --limit 100"
else
  echo "Ablation 1 model not found, skipping evaluation."
fi

# 4. Run evaluation on Ablation 2 (if training has completed)
if [ -d "models/mistral-7b-nli-cot-ablation2" ]; then
  echo "Evaluating Ablation 2 model..."
  docker run --rm --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/hf_cache:/root/.cache/huggingface \
    --env-file .env \
    mistral-nli-ft \
    bash -c "pip install scikit-learn tqdm pandas && python scripts/evaluate_models.py \
      --model-type hf \
      --model-path mistralai/Mistral-7B-v0.3 \
      --adapter-path models/mistral-7b-nli-cot-ablation2 \
      --use-4bit \
      --input-csv data/original_data/test.csv \
      --output-dir results \
      --limit 100"
else
  echo "Ablation 2 model not found, skipping evaluation."
fi

# 5. Compare results across all models
echo "All evaluations complete. Results available in the results directory."
echo "Summary of model accuracies:"
if ls results/*_metrics.json 1> /dev/null 2>&1; then
  grep -h "Accuracy" results/*_metrics.json | sort
else
  echo "No metrics files found yet."
fi 