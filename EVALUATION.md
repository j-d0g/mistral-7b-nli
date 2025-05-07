# Evaluating the Mistral-7B NLI Model

This document provides instructions for evaluating NLI models on test datasets, whether you've trained your own models or downloaded pre-trained ones.

## Table of Contents

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Evaluating a Model](#evaluating-a-model)
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
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv

# Specifying GPU to use
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv --gpu 1
```

### Option 2: Evaluating a Downloaded Model

You can also download our pre-trained models and evaluate them:

```bash
# First download a model (one-time operation)
docker run --rm -v $(pwd):/app -w /app --env-file .env mistral-nli-ft python3 models/download_model.py --model mistral-thinking-ablation0

# Then evaluate it
./run_inference.sh --model models/mistral-thinking-ablation0 --data data/original_data/test.csv
```

## Understanding Results

After evaluation completes, you'll find these files in the `results/` directory:

1. **JSON file** (`results/[model_name]-[dataset_name]-[timestamp].json`): 
   - Contains detailed information including:
     - Model configuration
     - Overall accuracy (if input data had labels)
     - Inference time statistics
     - Per-example results with premise, hypothesis, predicted label, and raw model output
   - Used for detailed analysis and debugging

2. **CSV file** (`results/[model_name]-[dataset_name]-[timestamp].csv`):
   - Contains just the `predicted_label` column with 0/1 values
   - Simplified format for quick review or submission

The script also saves checkpoint files during processing (`results/checkpoint_[model_name]-[dataset_name]-[timestamp].json`), which can be useful for debugging or recovering from interruptions.

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

## Command Examples

Here are more detailed examples of running inference:

```bash
# Basic example with default model on test set
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv

# Using a specific GPU (e.g., second GPU in system)
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/original_data/test.csv --gpu 1

# Evaluating a specific checkpoint from training
./run_inference.sh --model models/mistral-thinking-ablation1-best/checkpoint-500 --data data/original_data/test.csv

# Running on unlabeled data (will not report accuracy)
./run_inference.sh --model models/mistral-thinking-ablation1-best --data data/sample/unlabeled_examples.csv
```

## Output File Structure

### JSON Output Example

Here's an example of the structure of the JSON output file (based on actual results):

```json
{
  "model": "models/mistral-thinking-ablation1-best",
  "inference_time_seconds": 83.56,
  "samples_per_second": 0.23,
  "results": [
    {
      "premise": "Nearby is the new, architecturally inventive Supreme Court Building.",
      "hypothesis": "The Supreme Court Building was built in the early 1980s.",
      "predicted_label": 1,
      "output": "Premise: Nearby is the new, architecturally inventive Supreme Court Building.\nHypothesis: The Supreme Court Building was built in the early 1980s.\n\nUse chain of thought reasoning to determine if the hypothesis is entailed by the premise. Provide your reasoning and the final label (0 or 1) in JSON format: {\"thought_process\": \"...\", \"predicted_label\": ...}  {\"thought_process\": \"step 1: the premise describes the supreme court building as 'new', indicating it was built recently. step 2: the hypothesis suggests the building was built in the early 1980s, which is consistent with the premise's description of the building as 'new'. step 3: therefore, the hypothesis is entailed by the premise.\", \"predicted_label\": 1}"
    },
    {
      "premise": "Two examples of palatial townhouses funded by pastel (blue dye) are the Hotel d'Assezat (Place d'Assezat), now home to an art gallery that includes 30 works by Bonnard, and the Hotel de Bernuy (1 Rue Gambetta).",
      "hypothesis": "The art gallery has only ten works by Bonnard.",
      "predicted_label": 0,
      "output": "Premise: Two examples of palatial townhouses funded by pastel (blue dye) are the Hotel d'Assezat (Place d'Assezat), now home to an art gallery that includes 30 works by Bonnard, and the Hotel de Bernuy (1 Rue Gambetta).\nHypothesis: The art gallery has only ten works by Bonnard.\n\nUse chain of thought reasoning to determine if the hypothesis is entailed by the premise. Provide your reasoning and the final label (0 or 1) in JSON format: {\"thought_process\": \"...\", \"predicted_label\": ...}  {\"thought_process\": \"step 1: in the premise, it is mentioned that there are 30 works by bonnard in the art gallery located in the hotel d'assezat. this information is factual and consistent. step 2: the hypothesis states that there are only ten works by bonnard in the art gallery. this contradicts the information provided in the premise. step 3: based on the logical reasoning, the hypothesis cannot be inferred from the premise as it contradicts the factual information provided.\", \"predicted_label\": 0}"
    }
  ]
}
```

Note that the model's output is stored in the `output` field, which contains the full text response. The model returns its reasoning and prediction in JSON format within that text. The `predicted_label` field in the results contains the extracted final prediction (0 or 1).

### CSV Output Example

The CSV output is very simple and only contains the predicted labels:

```csv
predicted_label
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
    --model_id models/mistral-thinking-ablation1-best \
    --test_file data/original_data/test.csv \
    --batch_size 16 \
    --use_cot
```

### Batch Processing

The script processes examples in batches (default: 16 in `run_inference.sh`) to maximize throughput. The optimal batch size depends on:

- Available GPU memory
- Model size and quantization
- Length of input examples

A batch size of 16-32 typically works well for 4-bit quantized models on GPUs with 24GB memory.

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

- **Synthetic Data Augmentation**: See [DATA.md](DATA.md)
- **Training**: See [TRAINING.md](TRAINING.md)
- **Research Methodology**: See [REPORT.md](REPORT.md) 