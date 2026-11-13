# Models Directory

This directory stores trained model checkpoints and provides utilities for downloading pre-trained models.

## Content

After training, this directory will contain a subdirectory per run, named by the `output_dir` of the config (for example `nlistral-ablation1-best/`). `download_model.py` places the published adapters in:
- `nlistral-ablation0/` - adapter from `ablation0_best.py`
- `nlistral-ablation1/` - adapter from `ablation1_best.py` (best on the held-out test set)
- `nlistral-ablation2/` - adapter from `ablation2_best.py`

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

## Notes

- The adapters are stored in the gated Hugging Face repository [`jd0g/nlistral-7b-qlora`](https://huggingface.co/jd0g/nlistral-7b-qlora); request access on its page
- Full training runs with checkpoints and logs are in [`jd0g/Mistral-7B-NLI-v0.3`](https://huggingface.co/jd0g/Mistral-7B-NLI-v0.3)
- The upload and download scripts require a Hugging Face API token in the `.env` file (`HF_TOKEN`)