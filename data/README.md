# NLI Dataset

This directory contains the datasets for the Mistral-7B NLI fine-tuning project.

## Directory Structure

- `original_data/`: Contains the original NLI datasets (train, dev, test)
- `finetune/`: Processed data for fine-tuning
- `original_thoughts/`: Original thought generations for NLI examples
- `reflected_thoughts/`: Reflections on the NLI reasoning process
- `scored_thoughts/`: Thought examples with scores from an earlier, abandoned LLM-as-a-judge stage (not used for fine-tuning)

## Usage

### Downloading the Dataset

To download the dataset from Hugging Face:

```bash
python data/download_data.py
```

```bash
# From the project root directory:
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/download_data.py
```

This will download all dataset files from the Hugging Face repository.

### Uploading the Dataset

To upload any changes to the dataset to Hugging Face:

```bash
python data/upload_data.py
```

```bash
# From the project root directory:
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 data/upload_data.py
```

This will upload all relevant data directories to the repository.

## Notes

- The dataset is stored in the public Hugging Face repository [`jd0g/Mistral-NLI-Thoughts`](https://huggingface.co/datasets/jd0g/Mistral-NLI-Thoughts), under the `data/` prefix
- Downloading needs no token; uploading requires a Hugging Face token in the `.env` file (`HF_TOKEN`)
- Row counts and the split are documented in the README's "Data and splits" section