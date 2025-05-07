# NLI Dataset

This directory contains the datasets for the Mistral-7B NLI fine-tuning project.

## Directory Structure

- `original_data/`: Contains the original NLI datasets (train, dev, test)
- `finetune/`: Processed data for fine-tuning
- `original_thoughts/`: Original thought generations for NLI examples
- `reflected_thoughts/`: Reflections on the NLI reasoning process
- `scored_thoughts/`: Thought examples with evaluation scores

## Usage

### Downloading the Dataset

To download the dataset from Hugging Face:

```bash
python data/download_data.py
```

This will download all dataset files from the Hugging Face repository.

### Uploading the Dataset

To upload any changes to the dataset to Hugging Face:

```bash
python data/upload_data.py
```

This will upload all relevant data directories to the repository.

## Notes

- The dataset is stored in the Hugging Face repository: `jd0g/Mistral-NLI-Thoughts`
- The upload and download scripts require a Hugging Face API token in `.env` file (HF_TOKEN) 