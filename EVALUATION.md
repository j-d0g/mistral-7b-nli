# Evaluating the Mistral-7B NLI Model

This document provides instructions for evaluating NLI models on test datasets, whether you've trained your own models or downloaded pre-trained ones.

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Evaluating a Model](#evaluating-a-model)
  - [Interactive Demo Notebook](#interactive-demo-notebook)
  - [Understanding Results](#understanding-results)
  - [Common Issues](#common-issues)
- [Deep Dive: Evaluation Details](#deep-dive-evaluation-details)
  - [Input File Format](#input-file-format)
  - [Evaluation Script Architecture](#evaluation-script-architecture)
  - [Command Examples](#command-examples)
  - [Output File Structure](#output-file-structure)
  - [Metrics and Calculations](#metrics-and-calculations)
  - [Advanced Evaluation Options](#advanced-evaluation-options)
  - [Performance Optimization](#performance-optimization)

---

# Quick Start
For a deeper dive on into the code & experimentation, see - [Deep Dive: Evaluation Details](#deep-dive-evaluation-tetails)
.

To follow the data augmentation process or to train your own QLoRA adaptors, follow **[DATA.md](DATA.md)**, **[TRAINING.md](TRAINING.md)**.

For more methodology/results oriented details, check out the **[REPORT.md](REPORT.md)**.

## Prerequisites

Before you begin evaluating models, ensure you have:

1. **Docker installed** with NVIDIA Container Toolkit (for GPU support)
2. **Downloaded the datasets** using instructions in [DATA.md](DATA.md)
3. **Built the Docker image**:
   ```bash
   docker build -t mistral-nli-ft .
   ```
4. **Either a trained model using instructions in [TRAINING.md](TRAINING.md) or a downloaded model checkpoint**

> **Note:** Unlike the instructions in instructions in [DATA.md](DATA.md), which can be run locally with Python, **Docker is required** for model evaluation due to the complex GPU dependencies, model quantization requirements, and compatibility needs. The `run_inference.sh` script is specifically designed to work within the Docker container environment to ensure consistent results across different hardware setups.

## Evaluating a Model

### Option 1: Evaluating Your Own Trained Model

If you've already trained a model using the instructions in [TRAINING.md](TRAINING.md), you can evaluate it directly:

```bash
# Basic usage with default parameters
./run_inference.sh --model models/nlistral-ablation1 --data data/original_data/test.csv

# Specifying GPU to use
./run_inference.sh --model models/nlistral-ablation1 --data data/original_data/test.csv --gpu 1
```

### Option 2: Evaluating a Downloaded Model

You can also download our pre-trained models and evaluate them:

```bash
# First download a model (one-time operation)
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 models/download_model.py --model nlistral-ablation0

# Then evaluate it
./run_inference.sh --model models/nlistral-ablation0 --data data/original_data/test.csv
```

### Option 3: Interactive Demo Notebook

For a more user-friendly evaluation experience, especially when testing with custom examples, you can use our interactive Jupyter notebook:

1. **Start the Jupyter server in Docker**:
   ```bash
   ./run_notebook.sh
   ```

2. **Open the provided URL** in your browser and navigate to `demo.ipynb`

The demo notebook provides a streamlined interface for:
- Loading models directly from HuggingFace
- Processing individual premise-hypothesis pairs interactively
- Batch processing from CSV files
- Extracting predictions in the required format

This approach is particularly useful for:
- Quick experimentation with custom examples
- Testing models without running command-line scripts
- Visualizing outputs and understanding model behavior
- Preparing submission files from test datasets

If you prefer to work in your IDE, you can also connect VS Code or other IDEs to the Jupyter server running in Docker.

## Understanding Results

After evaluation completes, you'll find these files in the `results/` directory:

1. **JSON file** (`results/[model_name]-[dataset_name]-[labelled|unlabelled].json`): 
   - Contains detailed information including:
     - Model configuration
     - Overall accuracy (if input data had labels)
     - Inference time statistics
     - Per-example results with premise, hypothesis, predicted label, and raw model output
   - Used for detailed analysis and debugging

2. **CSV file** (`results/[model_name]-[dataset_name]-[labelled|unlabelled].csv`):
   - Contains just the `prediction` column with 0/1 values
   - Simplified format for quick review or submission

The script also saves checkpoint files during processing (`results/checkpoint_[model_name]-[dataset_name]-[labelled|unlabelled].json`), which can be useful for debugging or recovering from interruptions.

---

# Deep Dive: Evaluation Details

This section provides in-depth information about the evaluation process, metrics, and implementation details.

## Input File Format

The evaluation script expects input data in CSV format. There are two accepted formats:

### 1. Labeled Data (for accuracy evaluation)

CSV files with a `label` column for calculating accuracy metrics:

```csv
premise,hypothesis,label
The man is walking his dog in the park.,The man has a dog.,1
The child is playing with blocks.,The child is sleeping.,0
All birds can fly.,Penguins are birds that cannot fly.,1
```

### 2. Unlabeled Data (for prediction only)

CSV files without a `label` column for generating predictions:

```csv
premise,hypothesis
The sun rises in the east.,The sun sets in the west.
Water boils at 100 degrees Celsius at sea level.,Water freezes at 0 degrees Celsius.
```

Both formats should have at minimum the `premise` and `hypothesis` columns, which contain the text to be analyzed.

## Evaluation Script Architecture

The evaluation process consists of these key components:

1. **`run_inference.sh`**: The main wrapper script in the project root that:
   - Handles Docker command construction
   - Manages volume mounting and GPU selection
   - Executes the Python inference script inside the container

2. **`evaluate/sample_model.py`**: The core Python script that:
   - Loads the model and tokenizer
   - Applies 4-bit quantization
   - Processes test examples in batches
   - Extracts predictions and thought processes from model outputs
   - Calculates metrics and saves results

3. **Supporting components**:
   - `prompts.py`: Contains the inference prompt template
   - Parsing logic within `sample_model.py` to extract JSON from model outputs

4. **Interactive Demo**:
   - `demo.ipynb`: Jupyter notebook with streamlined code for model loading and inference
   - `run_notebook.sh`: Script to launch the Jupyter environment in Docker

## Command Examples

Here are more detailed examples of running inference:

```bash
# Basic example with default model on test set
./run_inference.sh --model models/nlistral-ablation1 --data data/original_data/test.csv

# Using a specific GPU (e.g., second GPU in system)
./run_inference.sh --model models/nlistral-ablation1 --data data/original_data/test.csv --gpu 1

# Evaluating a specific checkpoint from training
./run_inference.sh --model models/nlistral-ablation1/checkpoint-500 --data data/original_data/test.csv

# Running on unlabeled data (will not report accuracy)
./run_inference.sh --model models/nlistral-ablation1 --data data/sample/unlabeled_examples.csv

# Running the interactive notebook (alternative to command-line)
./run_notebook.sh
```

## Output File Structure

### JSON Output Example

Here's an example of the structure of the JSON output file (based on actual results):

```json
{
  "model": "models/nlistral-ablation1",
  "inference_time_seconds": 8745.582508087158,
  "samples_per_second": 0.2260569834166954,
  "use_cot": true,
  "accuracy": 0.8958017197774406,
  "precision": 0.8956566494591938,
  "recall": 0.8957555559198259,
  "f1_score": 0.895702426342089,
  "results": [
    {
      "premise": "EXECUTORY CONTRACT - A contract which has not been performed by all parties to it.",
      "hypothesis": "They wanted a flexible contract.",
      "predicted_label": 1,
      "output": "Premise: EXECUTORY CONTRACT - A contract which has not been performed by all parties to it.\nHypothesis: They wanted a flexible contract.\n\nUse chain of thought reasoning to determine if the hypothesis is entailed by the premise. Provide your reasoning and the final label (0 or 1) in JSON format: {\"thought_process\": \"...\", \"predicted_label\": ...}  {\"thought_process\": \"step 1: an executory contract is one that has not been fully performed by all parties. this means that there are still obligations and duties to be met. step 2: the hypothesis suggests that 'they' wanted a flexible contract. a flexible contract is one that allows for changes or modifications to be made, often in response to unforeseen circumstances. step 3: it is reasonable to infer that 'they' might have wanted a flexible contract because of the unperformed nature of the executory contract. this would allow for adjustments to be made as the contract is being carried out, ensuring that the contract remains relevant and effective. step 4: the hypothesis is not directly stated in the premise, but it can be logically inferred based on the information provided.step 5: it is important to note that the hypothesis only applies to the specific context of the executory contract described in the premise. it does not necessarily mean that all executory contracts are desired to be flexible, but rather that in this particular case, the parties involved may have had that intention.\", \"predicted_label\": 1} ",
      "true_label": 1,
      "correct": true
    },
    {
      "premise": "Joseph Nocera from Slate.",
      "hypothesis": "Nocera is the chief editor at Slate.",
      "predicted_label": 1,
      "output": "Premise: Joseph Nocera from Slate.\nHypothesis: Nocera is the chief editor at Slate.\n\nUse chain of thought reasoning to determine if the hypothesis is entailed by the premise. Provide your reasoning and the final label (0 or 1) in JSON format: {\"thought_process\": \"...\", \"predicted_label\": ...}  {\"thought_process\": \"step 1: joseph nocera is identified as a writer for slate. step 2: in journalism, writers often have specific roles within their organizations. step 3: while the premise doesn't explicitly state nocera's role, it's reasonable to infer that he might have a more senior role, such as chief editor, given his status as a writer for slate. step 4: therefore, the hypothesis that nocera is the chief editor at slate is plausible and aligns with the information provided in the premise.\", \"predicted_label\": 1} ",
      "true_label": 1,
      "correct": true
    }
  ]
}
```

Note that the model's output is stored in the `output` field, which contains the full text response. The model returns its reasoning and prediction in JSON format within that text. The `predicted_label` field in the results contains the extracted final prediction (0 or 1).

### CSV Output Example

The CSV output is very simple and only contains the predicted labels:

```csv
prediction
1
0
1
0
```

This format is intended for submission or quick processing when only the final predictions are needed.

## Metrics and Calculations

If the input data contains labels, the evaluation calculates these metrics:

* **Accuracy**: Proportion of correctly classified examples
* **Inference Time**: Total processing time and samples per second
* **Detailed Results**: Per-example analysis including the prediction and reasoning

For NLI classification:
- Label 1 (entailment) is treated as the positive class
- Label 0 (no-entailment) is treated as the negative class

## Advanced Evaluation Options

### Command Line Arguments

The `run_inference.sh` script supports these options:

```bash
./run_inference.sh \
  --model PATH  # Path to model (default: models/mistral-7b-nli-cot)
  --data PATH   # Path to data CSV (default: data/sample/demo.csv)
  --gpu ID      # GPU ID to use (default: 0)
```

For more advanced parameters, you can run the underlying Python script directly:

```bash
docker run --gpus device=0 --rm -v $(pwd):/app -w /app mistral-nli-ft \
    python evaluate/sample_model.py \
    --model_id models/nlistral-ablation1 \
    --test_file data/original_data/test.csv \
    --batch_size 16 \
    --use_cot
```

### Batch Processing

The script processes examples in batches (default: 16 in `run_inference.sh`) to maximize throughput. The optimal batch size depends on:

- Available GPU memory
- Model size and quantization
- Length of input examples

A batch size of 16 typically works well for 4-bit quantized models on GPUs with 24GB memory, enabling for much faster
inference of larger datasets.

## Performance Optimization

### Model Quantization

The evaluation script automatically loads the base model in 4-bit precision using the NF4 data type, which:
- Drastically reduces memory requirements (fits the 7B model on consumer GPUs)
- Maintains inference quality
- Improves inference speed

### Inference Speed Optimization

Several strategies are employed to maximize inference speed:

1. **Batch Processing**: Multiple examples are processed simultaneously
2. **4-bit Quantization**: Reduces memory footprint and can increase throughput
3. **Sequence Length Optimization**: Maximum length is set to 512, which is sufficient for the task
4. **Flash Attention**: Used when available for faster attention computation

## Further Information

You've now completed the steps to running inference on quantized Mistral models with Chain-of-Thought QLoRA adaptors!
To revisit earlier stages in the pipeline, see:
- **Synthetic Data Augmentation**: [DATA.md](DATA.md)
- **Training**: [TRAINING.md](TRAINING.md)

For a hollistic research-oriented deep dive on the methodology, experiments and findings, check out:
- **Research Methodology**: [REPORT.md](REPORT.md) 

Otherwise, thanks for visiting this repository!