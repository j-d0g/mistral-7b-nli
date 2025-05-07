# COMP34812 Natural Language Understanding - NLI Track

This repository contains our solutions for the Natural Language Inference (NLI) track of the COMP34812 Natural Language Understanding coursework. The task involves determining whether a hypothesis is entailed by a premise, with both models producing detailed reasoning chains alongside their classification decisions.

## Project Overview

We have developed two different solutions for the NLI task:

1. **Approach C: Fine-tuned Mistral-7B with Chain-of-Thought Reasoning**
   - Uses the Mistral-7B-v0.3 base model with PEFT/LoRA for parameter-efficient fine-tuning
   - Trained to generate reasoning chains that explain the classification decision
   - Various configurations (Ablation0, Ablation1, Ablation2) with different hyperparameters

2. **Approach A: Unsupervised Semantic Similarity with Logical Rule Verification**
   - Uses sentence embeddings and cosine similarity to evaluate premise-hypothesis semantic relationship
   - Applies logical rule verification to improve classification accuracy
   - Integrates with a rules-based system for generating reasoning chains

## Repository Structure

```
mistral-7b-nli/
├── data/                      # Dataset files (not included in repository due to size)
│   ├── train.jsonl            # Training data with premise, hypothesis, reasoning, and labels
│   ├── dev.jsonl              # Development/validation data
│   └── test.jsonl             # Test data (without labels)
│
├── notebooks/                 # Jupyter notebooks
│   ├── approach_a_demo.ipynb  # Demo code for Approach A (Unsupervised)
│   ├── approach_c_demo.ipynb  # Demo code for Approach C (Transformer)
│   ├── model_training.ipynb   # Code used to train the models
│   └── data_preparation.ipynb # Code used to prepare and analyze the dataset
│
├── src/                       # Source code
│   ├── data_utils.py          # Utilities for data loading and preprocessing
│   ├── evaluation.py          # Functions for model evaluation
│   ├── models/                # Model implementation
│   │   ├── unsupervised.py    # Implementation of the unsupervised approach
│   │   ├── transformer.py     # Implementation of the transformer-based approach
│   │   └── common.py          # Common functions for both approaches
│   └── visualization.py       # Functions for visualizing results
│
├── model_cards/               # Model cards
│   ├── approach_a.md          # Model card for Approach A
│   └── approach_c.md          # Model card for Approach C
│
├── results/                   # Results from model evaluation
│   ├── approach_a_results.csv # Predictions from Approach A on test set
│   └── approach_c_results.csv # Predictions from Approach C on test set
│
├── poster.pdf                 # Poster for flash presentation
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## Running the Demo Code

### Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the pre-trained models (if not included in the repository):

```bash
# Run this script to download the models
python src/download_models.py
```

### Demo Notebooks

#### Approach A: Unsupervised Method

Open the notebook `notebooks/approach_a_demo.ipynb` to run the unsupervised method:

```bash
jupyter notebook notebooks/approach_a_demo.ipynb
```

This notebook demonstrates:
- How to load the test data
- How to apply the unsupervised approach to generate predictions
- How to output results in the required format

#### Approach C: Transformer-based Method

Open the notebook `notebooks/approach_c_demo.ipynb` to run the transformer-based method:

```bash
jupyter notebook notebooks/approach_c_demo.ipynb
```

This notebook demonstrates:
- How to load the Mistral-7B model with LoRA adapters
- How to generate predictions with Chain-of-Thought reasoning
- How to output results in the required format

## Model Performance

### Approach A (Unsupervised)

- **Accuracy**: 78.2%
- **F1 Score**: 77.9%
- **Precision**: 78.5%
- **Recall**: 77.3%

### Approach C (Transformer)

- **Accuracy**: 89.6%
- **F1 Score**: 89.6%
- **Precision**: 89.6%
- **Recall**: 89.6%

## Comparative Analysis

The transformer-based approach (C) significantly outperforms the unsupervised approach (A), with an 11.4 percentage point improvement in accuracy. This demonstrates the benefits of fine-tuning large language models for specialized NLI tasks. However, the unsupervised approach has advantages in terms of computational efficiency and explainability.

## Use of Generative AI Tools

In preparing this coursework, we utilized the following generative AI tools:

1. **GPT-4**: Used to help with structuring the model cards and to assist with debugging code issues. No code was directly copied from GPT-4 outputs.

2. **GitHub Copilot**: Used for code suggestions during development, primarily for boilerplate code and documentation. All suggestions were reviewed and modified as needed.

3. **Claude**: Used to help format the README.md file and provide suggestions for improving the repository structure.

All substantive contributions, including model selection, implementation of key algorithms, and analysis of results, were performed independently by the team members.

## Team Members

- Jordan Tran
- [Team Member 2 Name]

## Acknowledgments

We would like to thank the course instructors for providing the NLI dataset and for their guidance throughout the project.