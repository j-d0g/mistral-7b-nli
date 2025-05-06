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

# Mistral-7B Fine-tuned for NLI with Chain-of-Thought Reasoning

<div align="center">
  <img src="metrics/model_architecture.png" alt="Model Architecture" width="600"/>
</div>

This repository contains the best fine-tuned versions of `mistralai/Mistral-7B-v0.3` for Natural Language Inference (NLI), specifically trained to generate Chain-of-Thought (CoT) reasoning alongside the classification label (0 for no-entailment, 1 for entailment).

This work was completed as part of a university assignment at The University of Manchester, focusing on improving both classification accuracy and reasoning transparency.

## Model Variants

This repository hosts the best checkpoint from each ablation study, representing different training configurations:

*   **`Ablation0_Best`**: The optimized small batch configuration.
    *   **Configuration Details**:
        * Base model: `mistralai/Mistral-7B-v0.3`
        * Effective batch size: 16 (8 per device × 2 gradient accumulation steps)
        * Learning rate: 2e-4 with cosine scheduler and 3% warmup ratio
        * Training: 2 epochs 
        * LoRA Config: r=16, alpha=32, dropout=0.05
        * Target modules: q_proj, k_proj, v_proj, o_proj
        * Sequence length: 512 tokens
        * 4-bit quantization with NF4 and double quantization

*   **`Ablation1_Best`**: The optimized medium batch configuration.
    *   **Configuration Details**:
        * Base model: `mistralai/Mistral-7B-v0.3`
        * Effective batch size: 32 (16 per device × 2 gradient accumulation steps)
        * Learning rate: 2e-4 with cosine scheduler and 3% warmup ratio
        * LoRA Config: r=16, alpha=32, dropout=0.05
        * Gradient checkpointing enabled for memory efficiency
        * Optimized for performance with tuned warmup ratio

*   **`Ablation2_Best`**: The refined large model configuration.
    *   **Configuration Details**:
        * Base model: `mistralai/Mistral-7B-v0.3`
        * Effective batch size: 64 (16 per device × 4 gradient accumulation steps)
        * Ultra-low learning rate: 5e-5 with cosine scheduler
        * Extended training: 5 epochs
        * Moderate warmup ratio: 5%
        * LoRA Config: r=32, alpha=64, dropout=0.05
        * Stability measures: gradient clipping at 1.0

## Training Methodology

### Data Preparation

The data preparation involved a three-stage process:

<div align="center">
  <img src="metrics/data_pipeline.png" alt="Data Pipeline" width="800"/>
</div>

1. **Base Data Preparation**: Split into train (90%), validation (5%), and test (5%) sets
2. **Thought Generation**: Generated reasoning chains for each premise-hypothesis pair
3. **Reflection Generation**: Enhanced reasoning via self-critique and reflection

<div align="center">
  <img src="metrics/token_count_distribution.png" alt="Token Count Distribution" width="700"/>
  <p><em>Distribution of token counts across premises, hypotheses, and reasoning chains in the training data</em></p>
</div>

### Training Process

<div align="center">
  <img src="metrics/training_dynamics.png" alt="Training Dynamics" width="700"/>
  <p><em>Training dynamics for different model configurations, showing validation loss and accuracy during fine-tuning. Note how Ablation2, with its lower learning rate and larger batch size, demonstrates more stable convergence.</em></p>
</div>

## Configuration Differences

The models represent different training approaches with increasing complexity:

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

## Performance Results

### Classification Performance on Test Set

On the test set we derived from our original dev and train, we report the following performance across the best configurations across our ablation studies.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Ablation0_Best | 89.3% | 89.2% | 89.4% | 89.3% |
| Ablation1_Best | 89.6% | 89.6% | 89.6% | 89.6% |
| Ablation2_Best | 89.33% | 89.38% | 89.27% | 89.30% |

### Reasoning Quality Assessment

<div align="center">
  <img src="metrics/token_accuracy_comparison.png" alt="Comparison of Instruct vs Fine-tuned Model Accuracy by Token Length" width="700"/>
</div>

Our fine-tuned model demonstrates excellent reasoning capabilities across various token length ranges, with a notable improvement in medium-length reasoning chains (101-300 tokens) compared to the Mistral-7B-Instruct model with prompt engineering.

<div align="center">
  <img src="metrics/original_token_vs_accuracy.png" alt="Mistral-7B-Instruct Token Length vs Accuracy" width="700"/>
</div>

**Comparative analysis across token ranges:**

| Token Range | Mistral-7B-Instruct Accuracy | Fine-tuned Model Accuracy | Improvement |
|-------------|-------------------------|---------------------------|-------------|
| 0-100 | 86.44% | 83.87% | -2.57% |
| 101-200 | 80.14% | 90.12% | +9.98% |
| 201-300 | 69.50% | 92.40% | +22.90% |
| 301+ | 57.16% | 60.87% | +3.71% |

**Key findings from reasoning quality analysis:**

- **Mistral-7B-Instruct** showed declining accuracy with increasing token length (86.4% → 57.2%)
- **Fine-tuned model** achieves peak accuracy with medium-length reasoning (90.1% → 92.4%)
- **Maximum improvement** observed in long reasoning chains (201-300 tokens) with +22.90 percentage points
- Performance on very long reasoning (301+ tokens) remains challenging, but still shows improvement

This indicates that the fine-tuning process successfully addressed the quality degradation in longer reasoning chains that was present in the Mistral-7B-Instruct model. The most dramatic improvements were observed in the critical medium-to-long token ranges that represent the majority of examples in practical use.

**Comparison with base Mistral-7B model:**
It's important to note that the base Mistral-7B model (without instruction tuning) performed significantly worse overall, with only 53.77% accuracy on the NLI task. The fine-tuning process represents a 35.81 percentage point improvement over this baseline.

## Technical Implementation Details

All models use Parameter-Efficient Fine-Tuning (PEFT) with QLoRA:

- **Quantization**: 4-bit with NF4 type and double quantization
- **Optimizer**: paged_adamw_8bit with weight decay 0.01
- **Loss**: Standard autoregressive language modeling loss
- **Scheduler**: Cosine with warmup
- **Target Modules**: Query, Key, Value, and Output projections

This configuration balances efficiency and performance, allowing fine-tuning of the 7B parameter model on consumer hardware while maintaining high quality outputs.

## Loading a Model

You can load any of the models using the `peft` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

repo_id = "jd0g/Mistral-v0.3-Thinking_NLI"
# Choose the desired model variant
model_variant = "Ablation2_Best"  # Options: "Ablation0_Best", "Ablation1_Best", "Ablation2_Best"

# Load the base model with 4-bit quantization
base_model_name = "mistralai/Mistral-7B-v0.3"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Load the PeftModel (adapter)
model = PeftModel.from_pretrained(base_model, repo_id, subfolder=model_variant)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Example inference with proper formatting
premise = "All birds can fly."
hypothesis = "Penguins can fly."
prompt = f"""
Premise: {premise}
Hypothesis: {hypothesis}

Determine if the hypothesis can be inferred from the premise. Write out your thought process step by step, then provide your final answer (1 for entailment, 0 for no entailment). Respond in JSON format with 'thought_process' and 'predicted_label' keys. 
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

These models are designed for NLI tasks where understanding the reasoning process (via CoT) is valuable. The main differences between the variants:

- **Ablation0_Best**: Good overall balance of performance and reasoning with a small batch approach
- **Ablation1_Best**: Enhanced performance with medium batch size and memory optimization
- **Ablation2_Best**: Highest capacity and stability, best for complex reasoning tasks

All models output both detailed reasoning and a final classification label in JSON format.

## Limitations and Ethical Considerations

### Known Limitations

1. **Domain Specificity**: The models are trained on a limited set of NLI examples and may not generalize to all domains
2. **Reasoning Patterns**: The models may develop specific reasoning patterns that don't represent the full spectrum of logical analysis
3. **Chain Length Sensitivity**: As shown in our token vs accuracy analysis, very long reasoning chains (301+ tokens) still show decreased performance compared to medium-length chains

### Reasoning Length and Prediction Bias

An important phenomenon we observed during training relates to reasoning length and prediction tendencies:

<div align="center">
  <img src="metrics/prediction_distribution.png" alt="Prediction Distribution by Token Length" width="700"/>
</div>

In our analysis of the Mistral-7B-Instruct model outputs, we found that as reasoning chain length increases, models become more likely to predict "no-entailment" and less likely to predict "entailment." This creates a bias pattern where longer reasoning tends to be more critical and hesitant to assert entailment relationships. 

This pattern appears to be a fundamental aspect of how language models reason - as they generate more text, they consider more potential contradictions or edge cases, naturally leading to more conservative predictions. Interestingly, this effect can be more pronounced in more sophisticated reasoning models.

Our fine-tuning process specifically addressed this bias, resulting in more balanced predictions across different reasoning lengths. Users should still be aware that very long reasoning chains (beyond the 300-token range) may exhibit some residual bias toward no-entailment predictions.

### Ethical Considerations

1. **Bias**: The models may inherit biases present in the training data
2. **Reasoning Transparency**: While CoT improves explainability, models may still occasionally rationalize incorrect conclusions

---

*This model card was created as part of an open-ended research project at The University of Manchester.*