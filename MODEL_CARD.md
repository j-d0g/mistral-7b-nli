---
license: apache-2.0
tags:
- natural-language-inference
- nli
- chain-of-thought
- cot
- mistral
- fine-tuning
- peft
- lora
---

# Model Card for Mistral-7B Fine-tuned for NLI with Chain-of-Thought Reasoning

<!-- Provide a quick summary of what the model is/does. -->

This model is a fine-tuned version of Mistral-7B-v0.3 for Natural Language Inference (NLI) that generates Chain-of-Thought (CoT) reasoning alongside classification labels (0 for no-entailment, 1 for entailment).

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model was developed to improve both classification accuracy and reasoning transparency for Natural Language Inference. It uses Parameter-Efficient Fine-Tuning (PEFT) with QLoRA to enhance Mistral-7B's performance on determining whether a hypothesis can be inferred from a premise. The model produces detailed reasoning chains explaining its decision-making process before providing a final classification.

<div align="center">
  <img src="metrics/model_architecture.png" alt="Model Architecture" width="600"/>
  <p><em>Figure 1: Overall architecture showing QLoRA's approach to parameter-efficient fine-tuning with 4-bit quantized base model weights and trainable low-rank adapters.</em></p>
</div>

- **Developed by:** Jordan Tran, The University of Manchester
- **Language(s):** English
- **Model type:** Natural Language Inference with Chain-of-Thought
- **Model architecture:** Mistral-7B with LoRA adapters
- **Finetuned from model:** mistralai/Mistral-7B-v0.3

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/mistralai/Mistral-7B-v0.3
- **Paper or documentation:** https://arxiv.org/abs/2310.06825

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model was trained on a custom dataset of over 35,000 NLI examples with premise-hypothesis pairs and generated Chain-of-Thought reasoning. This training data was created through our novel Reflection-CoT pipeline:

<div align="center">
  <img src="metrics/data_pipeline.png" alt="Data Pipeline" width="800"/>
  <p><em>Figure 2: The Reflection-CoT pipeline for data augmentation, showing the process from initial CoT generation to error identification and reflection-based correction.</em></p>
</div>

The pipeline consisted of three key stages:
1. **Initial CoT Generation**: All premise-hypothesis pairs processed by Mistral-7B to generate reasoning and labels
2. **Error Identification**: Examples where the initial prediction mismatched the dataset's ground-truth label were flagged (approximately 24.26%)
3. **Reflection-CoT Generation**: For flagged examples, a stronger model (Mistral-Nemo-12B) was used to analyze the flawed reasoning and generate corrections

This approach ensured comprehensive coverage of challenging cases while preserving the model's natural reasoning patterns for examples it already handled correctly.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->

We conducted several ablation studies to identify optimal training parameters. The repository hosts the best checkpoint from each ablation study:

| Parameter | Ablation0_Best | Ablation1_Best | Ablation2_Best |
|-----------|----------------|----------------|----------------|
| Batch Size | 8 per device | 16 per device | 16 per device |
| Gradient Accumulation | 2 steps | 2 steps | 4 steps |
| Effective Batch Size | 16 | 32 | 64 |
| Learning Rate | 2e-4 | 2e-4 | 5e-5 |
| LoRA Rank | 16 | 16 | 32 |
| LoRA Alpha | 32 | 32 | 64 |
| Training Duration | 2 epochs | 2 epochs | 5 epochs |
| Gradient Checkpointing | Disabled | Enabled | Enabled |
| Warmup Ratio | 0.03 | 0.03 | 0.05 |
| Gradient Clipping | None | None | 1.0 |
| Primary Focus | Small batch | Medium batch with optimization | Large model with stability |

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

- **Training Hardware:** 1~4x NVIDIA RTX 4090 GPUs
- **Base Model Size:** 7B parameters
- **LoRA Adapter Size:** 591~646 MB depending on configuration
- **Training Time:** 5-12 hours depending on configuration
- **Total Training Time:** 50 hours across 25+ ablation studies

<div align="center">
  <img src="metrics/training_dynamics.png" alt="Training Dynamics" width="700"/>
  <p><em>Figure 3: Training dynamics for different model configurations, showing validation loss and accuracy during fine-tuning. Note how Ablation2, with its lower learning rate and larger batch size, demonstrates more stable convergence.</em></p>
</div>

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The model was evaluated on a test set derived from our original development and training data, comprising approximately 1,972 examples (5% of the total dataset) with balanced label distribution.

### Results

The table below shows the performance of our best model configurations on the test set:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Ablation0_Best | 89.3% | 89.2% | 89.4% | 89.3% |
| Ablation1_Best | 89.6% | 89.6% | 89.6% | 89.6% |
| Ablation2_Best | 89.33% | 89.38% | 89.27% | 89.30% |

When compared to baseline models:

| Metric | Base Mistral-7B | Mistral-7B-Instruct* | Fine-tuned Model | Relative Improvement** |
|--------|----------------|-------------------|------------------|----------------------|
| Accuracy | 53.77% | 76.0% | 89.58% | +66.60% |
| Precision | 60.49% | 89.7% | 89.57% | +48.07% |
| Recall | 52.32% | 57.2% | 89.58% | +71.21% |
| F1 Score | 41.51% | 69.8% | 89.57% | +115.78% |

*Mistral-7B-Instruct with extensive prompt engineering and single-shot examples vs zero-shot fine-tuned model  
**Relative improvement calculated against the Base Mistral-7B model

One of our most significant findings was the improvement in reasoning quality across different chain lengths:

<div align="center">
  <img src="metrics/token_accuracy_comparison_combined.png" alt="Comparison of Token Length vs. Accuracy" width="800"/>
  <p><em>Figure 4: Side-by-side comparison of original model (left) and fine-tuned model (right) accuracy across token length ranges. The fine-tuned model shows significant improvements in medium-to-long reasoning chains (101-300 tokens), with a dramatic +23.41% improvement in the 201-300 token range where accuracy jumps from 68.99% to 92.40%.</em></p>
</div>

Our fine-tuned model showed remarkable improvements in handling medium-to-long reasoning chains (101-300 tokens):

| Token Range | Mistral-7B-Instruct Accuracy | Fine-tuned Model Accuracy | Improvement |
|-------------|-------------------------|---------------------------|-------------|
| 0-100 | 86.06% | 83.87% | -2.19% |
| 101-200 | 79.34% | 90.12% | +10.78% |
| 201-300 | 68.99% | 92.40% | +23.41% |
| 301+ | 56.79% | 60.87% | +4.08% |

These improvements demonstrate that our fine-tuning process successfully addressed the declining performance in longer reasoning chains that was present in the original model, with the most dramatic improvements in the critical 201-300 token range.

## Technical Specifications

### Hardware

- **Training Hardware:** 1~4x NVIDIA RTX 4090 GPUs
- **Inference Requirements:** 
  - Minimum: 16GB VRAM with 4-bit quantization
  - Recommended: 24GB+ VRAM for optimal performance

### Software

- **Framework:** PyTorch with Transformers library
- **Quantization:** 4-bit with NF4 type and double quantization
- **Optimizer:** paged_adamw_8bit with weight decay 0.01
- **Libraries:** PEFT, bitsandbytes, transformers, accelerate

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

### Domain Specificity and Model Limitations

1. **Domain Specificity:** The model is trained on a specific set of NLI examples and may not generalize to all domains or specialized knowledge areas
2. **Reasoning Patterns:** The model may develop specific reasoning patterns that don't represent the full spectrum of logical analysis
3. **Chain Length Sensitivity:** Very long reasoning chains (301+ tokens) still show decreased performance compared to medium-length chains

### Reasoning Length and Prediction Bias

An important phenomenon observed during training relates to reasoning length and prediction tendencies:

<div align="center">
  <img src="metrics/prediction_distribution.png" alt="Prediction Distribution by Token Length" width="700"/>
  <p><em>Figure 5: Analysis of prediction distribution across token length ranges, demonstrating that longer reasoning chains correlate with higher rates of no-entailment predictions.</em></p>
</div>

As reasoning chain length increases, models become more likely to predict "no-entailment" and less likely to predict "entailment." This creates a bias pattern where longer reasoning tends to be more critical and hesitant to assert entailment relationships.

This pattern appears to be a fundamental aspect of how language models reason - as they generate more text, they consider more potential contradictions or edge cases, naturally leading to more conservative predictions. Our fine-tuning process specifically addressed this bias, resulting in more balanced predictions across different reasoning lengths.

### Labeller Bias and Dataset Subjectivity

Our analysis revealed a significant finding: there appears to be substantial labeller bias in some portions of the dataset. Manual verification of examples where model predictions disagreed with dataset labels showed that many "no-entailment" examples in the dataset could reasonably be classified as "entailment" based on logical analysis.

This observation is reflected in the stark precision-recall imbalance we observed in our initial experiments (approximately 90% precision but only 50% recall). This suggests that in some cases, the disagreement between model predictions and dataset labels may not be due to model errors, but rather due to subjective judgments or inconsistencies in the original dataset annotation process.

The binary entailment/non-entailment distinction can be highly subjective in borderline cases, and our Reflection-CoT mechanism was specifically designed to handle these challenging examples by providing explicit reasoning paths that align with the dataset's ground truth labels, even in cases where the logical connection might be debatable. This approach acknowledges the presence of labeller bias while providing a structured framework to align model behavior with the expected outputs for the task.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

### Usage Example

```python
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig 
from dotenv import load_dotenv

# --- Configuration ---
HUB_REPO_ID = "jd0g/nlistral-7b-qlora"
ADAPTER_NAME_ON_HUB = "nlistral-7b-qlora-ablation1-best" # This is the subfolder name on the Hub
GPU_ID = 0
# ---

load_dotenv()
HF_TOKEN = "YOUR HF TOKEN HERE"

# Download adapter_config.json from the subfolder on the Hub
adapter_config = PeftConfig.from_pretrained(HUB_REPO_ID, subfolder=ADAPTER_NAME_ON_HUB, token=HF_TOKEN)
BASE_MODEL_ID = adapter_config.base_model_name_or_path


#  Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load the base model with quantization
print(f"Loading base model '{BASE_MODEL_ID}' with 4-bit quantization on GPU {GPU_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map={"": GPU_ID},
    trust_remote_code=True,
    token=HF_TOKEN
)
print("Base model loaded.")

# Load the tokenizer directly from the adapter's location on the Hub
print(f"Loading tokenizer from Hub: '{HUB_REPO_ID}' (subfolder: '{ADAPTER_NAME_ON_HUB}')...")
tokenizer = AutoTokenizer.from_pretrained(
    HUB_REPO_ID,
    subfolder=ADAPTER_NAME_ON_HUB, # Specify the subfolder
    use_fast=True,
    padding_side="left",
    token=HF_TOKEN
)
print("Tokenizer loaded.")

# Resize token embeddings if necessary
if len(tokenizer) != base_model.config.vocab_size:
    print(f"Resizing token embeddings of base model from {base_model.config.vocab_size} to {len(tokenizer)}.")
    base_model.resize_token_embeddings(len(tokenizer))

# Apply the PEFT adapter to the base model, loading adapter weights from the Hub
print(f"Applying PEFT adapter '{ADAPTER_NAME_ON_HUB}' from Hub to the base model...")
model = PeftModel.from_pretrained(
    base_model,
    HUB_REPO_ID,
    subfolder=ADAPTER_NAME_ON_HUB,
    token=HF_TOKEN
)
model.eval() # Set the model to evaluation mode
print("PEFT adapter applied. Model is ready.")

# Set pad_token if not already set
if tokenizer.pad_token is None:
    print("Tokenizer pad_token is None. Setting it to eos_token.")
    tokenizer.pad_token = tokenizer.eos_token

# 'model' and 'tokenizer' are now ready for use.
ADAPTER_PATH = f"{HUB_REPO_ID}/{ADAPTER_NAME_ON_HUB}"

print(f"\nModel and tokenizer for '{ADAPTER_PATH}' loaded successfully from Hugging Face Hub.")
```

This example demonstrates the essential steps:

1. Loading the model with 4-bit quantization
2. Creating a prompt in the expected format
3. Running inference to generate a prediction
4. Extracting the label and reasoning from the model's output

For batch processing and more advanced features, please refer to the full evaluation script in the repository.

### Interactive Demo

We provide an interactive Jupyter notebook demo that allows exploring the model in a more user-friendly way. The notebook features:

1. Load and explore sample data
2. Run inference on examples with clear visualization
3. Analyze Chain-of-Thought reasoning step-by-step
4. Process multiple examples in batches

Here's a streamlined version of the code that you can use to quickly test the model:

```python
import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Define the prompt template
FINETUNE_PROMPT = """Premise: {premise}
Hypothesis: {hypothesis}

Use chain of thought reasoning to determine if the hypothesis is entailed by the premise. Provide your reasoning and the final label (0 or 1) in JSON format: {{"thought_process": "...", "predicted_label": ...}}"""

premise = "The cat is on the rug."
hypothesis = "A feline is on the mat."

formatted_prompt = FINETUNE_PROMPT.format(
    premise=premise,
    hypothesis=hypothesis
)

input_text = f"[INST] {formatted_prompt} [/INST]"


inputs = tokenizer(input_text, return_tensors="pt").to(next(model.parameters()).device)

# Generate
with torch.no_grad():
    output_sequences = model.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

# Decode and print only the generated part
generated_text = tokenizer.decode(output_sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
generated_text
```

**Example Output:**
```JSON
{"thought_process": "step 1: the premise states that the cat is on the rug. a rug is a type of mat, often used for floor coverings. step 2: the hypothesis suggests that a feline is on the mat. since a mat can refer to a rug, and the cat is on the rug, it can be inferred that a feline is on the mat. step 3: based on the logical reasoning and lack of contradictory facts, the hypothesis can be inferred from the premise.", "predicted_label": 1}
```


### Project Context

This work was completed as part of the COMP34812 Natural Language Understanding coursework at The University of Manchester. The model demonstrates how Chain-of-Thought reasoning can be integrated with fine-tuning approaches to improve both performance and explainability in NLI tasks.

*This model card was created as part of the COMP34812 Natural Language Understanding coursework at The University of Manchester.*