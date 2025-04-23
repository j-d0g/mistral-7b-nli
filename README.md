# Mistral NLI with Chain-of-Thought Refinement

## Overview

This project focuses on fine-tuning Mistral-7B models for Natural Language Inference (NLI) tasks. The core idea is to augment a standard NLI dataset (premise, hypothesis, label) with Chain-of-Thought (CoT) reasoning to improve model performance.

A key challenge is generating high-quality CoT reasoning. This repository implements a multi-stage workflow to address this:

1.  **Initial CoT Generation:** Use the Mistral API (`prediction_service.py`) to generate an initial CoT (`thought_process`) and predict the NLI label for each example in the base dataset. Since the true label isn't provided during this step, the predictions and reasoning can be noisy or incorrect.
2.  **Scoring and Refinement:** Pass the augmented dataset (including the initial thoughts and predicted labels) through a `scoring_service.py`. This service uses Mistral models (potentially stronger ones like Mixtral) to evaluate the quality of the generated thoughts, assign a score, potentially correct the predicted label, and generate an *improved* thought process.
3.  **Gold-Standard Dataset Creation:** Filter or utilize the outputs from the scoring service based on the quality score to create a high-quality, "gold-standard" dataset containing reliable CoT reasoning and accurate labels.
4.  **Fine-tuning:** Use this gold-standard dataset for the final fine-tuning process of a Mistral-7B model (details of the fine-tuning process itself might be outside the immediate scope of the services defined here, but the data generation pipeline supports it).

The services are designed to interact with the Mistral API and produce structured JSON outputs.

## Features

*   **Mistral API Integration:** Classes (`llm/mistral.py`) for interacting with the official Mistral AI API.
*   **Prediction Service:** Generates initial CoT reasoning and NLI labels (`service/prediction_service.py`).
*   **Scoring Service:** Evaluates and refines CoT reasoning and labels (`service/scoring_service.py`).
*   **Pydantic Validation:** Uses Pydantic models (`models/response_models.py`) for robust validation of JSON responses from the API.
*   **Modular Structure:** Code is organized into services, LLM abstractions, models, and utilities.
*   **JSON Output:** Services are designed to produce structured JSON containing thoughts and labels.
*   **Testing Suite:** Includes unit and integration tests (`tests/`) using `pytest`.

## Repository Structure

```
.
├── data/              # (Optional) Base NLI datasets
├── llm/               # Language model API clients (e.g., mistral.py)
├── models/            # Pydantic models for data validation
├── notebooks/         # Jupyter notebooks for experimentation
├── scripts/           # Utility and test scripts
├── service/           # Core logic for prediction and scoring
├── tests/             # Pytest tests for services and utilities
├── utils/             # Shared utility functions (e.g., json_helpers.py)
├── .env               # Environment variables (API keys) - Gitignored
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd mistral-7b-nli
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    *   Create a file named `.env` in the project root.
    *   Add your Mistral API key to it:
        ```env
        MISTRAL_API_KEY=your_mistral_api_key_here
        ```

## Usage

The core logic resides in the `service` modules. You can import and use the `predict_label` and `generate_score` functions in your own scripts or notebooks.

*   **Prediction Service (`predict_label`):** Takes a premise, hypothesis, true label (for benchmarking), an LLM client instance, model details, and file paths. It calls the Mistral API to generate thoughts and a predicted label, returning a JSON-like dictionary.
*   **Scoring Service (`generate_score`):** Takes premise, hypothesis, the *initial thoughts* and *predicted label* from the prediction service, true label, LLM client, model details, and file paths. It calls the Mistral API to generate a score, an improved thought process, and a potentially corrected label.

**Example Scripts:**

*   `scripts/validate_llm_json.py`: Utility to test JSON validation.
*   `scripts/generate_thoughts.py`: Script for generating initial thoughts and predictions.
*   `scripts/score_thoughts.py`: Script for scoring and refining generated thoughts.

To run the generate thoughts script (requires `.env` file):
```bash
python scripts/generate_thoughts.py
```

## Testing

This project uses `pytest`. To run the tests:

1.  Ensure you have installed dependencies (`pip install -r requirements.txt`). You might need `pytest` specifically (`pip install pytest`).
2.  Make sure your `.env` file is configured if you want to run integration tests that hit the actual API.
3.  Run the tests from the project root:
    ```bash
    python -m pytest tests/ -v
    ```

    *   Tests requiring an API key are marked with `@pytest.mark.skipif` and will be skipped if the `MISTRAL_API_KEY` environment variable is not found.
