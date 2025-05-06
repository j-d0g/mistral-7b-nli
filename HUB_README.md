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

This repository contains the best fine-tuned versions of `mistralai/Mistral-7B-v0.3` for Natural Language Inference (NLI), specifically trained to generate Chain-of-Thought (CoT) reasoning alongside the classification label (0 for no-entailment, 1 for entailment).

The fine-tuning process and evaluation are detailed in the original project repository (consider adding link here if public).

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

## Configuration Differences

The models represent different training approaches with increasing complexity:

| Parameter | Ablation 0 Best | Ablation 1 Best | Ablation 2 Best |
|-----------|----------------|----------------|----------------|
| Batch Size | 8 per device | 16 per device | 16 per device |
| Gradient Accumulation | 2 steps | 2 steps | 4 steps |
| Effective Batch Size | 16 | 32 | 64 |
| Learning Rate | 2e-4 | 2e-4 | 5e-5 |
| LoRA Rank | 16 | 16 | 32 |
| LoRA Alpha | 32 | 32 | 64 |
| Training Duration | 2 epochs | 2 epochs | 5 epochs |
| Gradient Checkpointing | Disabled | Enabled | Enabled |
| Primary Focus | Small batch | Medium batch with optimization | Large model with stability |

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

- **Ablation 0 Best**: Good overall balance of performance and reasoning with a small batch approach
- **Ablation 1 Best**: Enhanced performance with medium batch size and memory optimization
- **Ablation 2 Best**: Highest capacity and stability, best for complex reasoning tasks

All models output both detailed reasoning and a final classification label in JSON format.

*(Add more details about Usage, Limitations, Bias, Training Procedure, Evaluation Results as available)*