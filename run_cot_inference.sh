#!/bin/bash

# Create the results directory if it doesn't exist
mkdir -p results

# Run docker-compose with CoT reasoning enabled
docker-compose run --rm mistral-nli-inference python run_nli_inference.py \
  --model_id "mistralai/Mistral-7B-v0.3" \
  --batch_size 8 \
  --test_file "data/original_data/test.csv" \
  --output_file "results/mistral-v0.3-cot-predictions.json" \
  --use_cot

echo "Chain-of-Thought inference completed!" 