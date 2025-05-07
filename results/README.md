# Experimental Results

This directory stores the output files from running inference and evaluation scripts.

## Uploading Results to Hugging Face

To back up or share your results, you can upload the entire `results/` directory to a Hugging Face dataset repository:

```bash
# Ensure your .env file has your HF_TOKEN
# Build the Docker image if needed: docker build -t mistral-nli-ft .

# Upload results to the default dataset (jd0g/nlistral-7b-results) or your own
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  -e HF_USERNAME=your_username \
  -e HF_RESULTS_REPO_NAME=your_results_repo_name \
  mistral-nli-ft \
  python3 results/upload_results.py
```

If `HF_USERNAME` and `HF_RESULTS_REPO_NAME` are not specified, it defaults to `jd0g/nlistral-7b-results`. The script will create the dataset repository if it doesn't exist.

## Downloading Results from Hugging Face

To retrieve results from a Hugging Face dataset repository:

```bash
# Download results from the default dataset or your own
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  -e HF_USERNAME=your_username \
  -e HF_RESULTS_REPO_NAME=your_results_repo_name \
  mistral-nli-ft \
  python3 results/download_results.py
```

This will download the entire dataset content into the local `results/` directory, overwriting existing files if they have the same name.

## Output Format

Inference runs typically produce:

*   **`.json` file**: Contains detailed output for each sample, including premise, hypothesis, predicted label, and the model's generated text (which includes the chain-of-thought reasoning).
*   **`.csv` file**: A simple CSV containing just the `prediction` column with 0/1 values, useful for quick analysis or scoring.

Files are named using the pattern: `[model_name]-[dataset_name]-[labelled|unlabelled].[extension]` 