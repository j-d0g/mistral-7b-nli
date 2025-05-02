---
license: apache-2.0 # Or choose appropriate license
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

This repository contains fine-tuned versions of `mistralai/Mistral-7B-v0.3` for Natural Language Inference (NLI), specifically trained to generate Chain-of-Thought (CoT) reasoning alongside the classification label (0 for no-entailment, 1 for entailment).

The fine-tuning process and evaluation are detailed in the original project repository (consider adding link here if public).

## Model Checkpoints & Ablations

This repository hosts checkpoints from different fine-tuning runs (ablations), organized into subdirectories:

*   **`Mistral_Thinking_Abl0.1`**: Checkpoints from the initial fine-tuning run (using combined correct + reflected data).
    *   `checkpoint-2225`: Final checkpoint from this run.
    *   **Configuration Details**:
        * Base model: `mistralai/Mistral-7B-v0.3`
        * Effective batch size: 32 (16 per device × 2 gradient accumulation steps)
        * Learning rate: 2e-4 with cosine scheduler and 3% warmup ratio
        * Training: 3 epochs with early stopping (eval_loss)
        * LoRA Config: r=16, alpha=32, dropout=0.05
        * Target modules: q_proj, k_proj, v_proj, o_proj
        * Sequence length: 512 tokens
        * 4-bit quantization with NF4 and double quantization
        * WandB logging enabled

*   **`Mistral_Thinking_Abl0.2`**: Checkpoints from a follow-up run based on Ablation 0. This run aimed to address token repetition issues (potentially related to EOS/padding) and was trained for significantly more epochs.
    *   `checkpoint-1750`: Checkpoint saved during this extended run.
    *   **Configuration Details**:
        * Base model: `mistralai/Mistral-7B-v0.3`
        * Effective batch size: 16 (8 per device × 2 gradient accumulation steps)
        * Learning rate: 2e-4 with cosine scheduler
        * Extended training duration (approximately 2× longer than Abl0.1)
        * LoRA Config: Same as Abl0.1 (r=16, alpha=32, dropout=0.05)
        * Resume from: `models/mistral-7b-nli-cot/checkpoint-2225`
        * Gradient checkpointing: Disabled
        * WandB run ID: jnz6en9a

*   **`Mistral_Thinking_Abl1`**: Checkpoints from an ablation focused on addressing potentially conflicting reasoning paths identified during data generation, where the base model showed bias towards "no entailment".
    *   `checkpoint-1250`: Final checkpoint from this run.
    *   **Configuration Details**:
        * Used filtered dataset with only correct examples from original generation
        * Similar configuration to Abl0.2 but with different dataset strategy
        * Focused on eliminating conflicting reasoning patterns
        * Results in cleaner reasoning paths but potentially narrower reasoning diversity

## Configuration Differences

The ablations differ in several key aspects:

| Parameter | Ablation 0.1 | Ablation 0.2 | Ablation 1 |
|-----------|--------------|--------------|------------|
| Batch Size | 16 per device | 8 per device | 8 per device |
| Gradient Accumulation | 2 steps | 2 steps | 2 steps |
| Effective Batch Size | 32 | 16 | 16 |
| Dataset | Combined (correct + reflected) | Combined (refined) | Correct examples only |
| Training Duration | Early stopped | Extended (2×) | Standard |
| Checkpoint Resumption | None | From Abl0.1 (ckpt-2225) | None |
| Primary Focus | Initial run | Fixing token repetition | Addressing reasoning conflicts |

## Technical Implementation Details

All models use Parameter-Efficient Fine-Tuning (PEFT) with QLoRA:

- **Quantization**: 4-bit with NF4 type and double quantization
- **Optimizer**: paged_adamw_8bit with weight decay 0.01
- **Loss**: Standard autoregressive language modeling loss
- **Scheduler**: Cosine with 3% warmup ratio
- **PEFT Config**: LoRA with r=16, alpha=32, dropout=0.05
- **Target Modules**: Query, Key, Value, and Output projections

This configuration balances efficiency and performance, allowing fine-tuning of the 7B parameter model on consumer hardware while maintaining high quality outputs.

## Loading a Specific Checkpoint

You can load a specific checkpoint using the `subfolder` argument with the `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

repo_id = "jd0g/Mistral-v0.3-Thinking_NLI"
# Choose the desired ablation and checkpoint subdirectory
# Example for Abl0.1 final checkpoint:
subfolder_path_abl01 = "Mistral_Thinking_Abl0.1/checkpoint-2225"
# Example for Abl1 final checkpoint:
subfolder_path_abl1 = "Mistral_Thinking_Abl1/checkpoint-1250"
# Example for Abl0.2 checkpoint:
subfolder_path_abl02 = "Mistral_Thinking_Abl0.2/checkpoint-1750"

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
model = PeftModel.from_pretrained(base_model, repo_id, subfolder=subfolder_path_abl01) # Or use subfolder_path_abl1, subfolder_path_abl02

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Example inference with proper formatting
premise = "All birds can fly."
hypothesis = "Penguins can fly."
prompt = f"""[INST] Premise: {premise}
Hypothesis: {hypothesis}

Determine if the hypothesis can be inferred from the premise. Write out your thought process step by step, then provide your final answer (1 for entailment, 0 for no entailment). Respond in JSON format with 'thought_process' and 'predicted_label' keys. [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Intended Use

These models are designed for NLI tasks where understanding the reasoning process (via CoT) is valuable. The main differences between the ablations:

- **Ablation 0.1**: Good general-purpose model with balance of performance and reasoning
- **Ablation 0.2**: Improved token generation with fewer repetition artifacts
- **Ablation 1**: Cleaner reasoning patterns but potentially less diverse thinking

All models output both detailed reasoning and a final classification label in JSON format.

*(Add more details about Usage, Limitations, Bias, Training Procedure, Evaluation Results as available)* 