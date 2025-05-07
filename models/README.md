# Models Directory

This directory stores trained model checkpoints and provides utilities for downloading pre-trained models.

## Content

After training, this directory will contain subdirectories for each training run, such as:
- `nlistral-ablation0/` - Base ablation0 configuration
- `nlistral-ablation0-best/` - Optimized ablation0 configuration
- `nlistral-ablation1/` - Best overall model (ablation1 optimized)
- `nlistral-ablation2/` - Base ablation2 configuration 
- `nlistral-ablation2-best/` - Optimized ablation2 configuration

Each model directory contains:
- Adapter weights (LoRA parameters)
- Training configuration
- Tokenizer files
- Training logs

## Utilities

- `download_model.py`: Downloads pre-trained models from Hugging Face

## Usage

To download a pre-trained model:

```bash
# Download a specific model
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 models/download_model.py --model nlistral-ablation0
```

For details on how to train your own models or use downloaded models for inference, refer to:
- [TRAINING.md](../TRAINING.md) - Fine-tuning instructions
- [EVALUATION.md](../EVALUATION.md) - Evaluation guidelines 