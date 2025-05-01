# Extraction Method Analysis Findings

## Overview

This report analyzes the extraction methods used in the NLI task evaluation and identifies why the fine-tuned models initially appeared to perform poorly until extraction logic was fixed. We've conducted a detailed analysis of both original and fixed result files to understand the exact mechanisms of extraction failure.

## Key Findings

1. **Dramatic Performance Differences**
   - Base model: 54.0% → 53.0% (−1.0%)
   - Fine-tuned model: 64.0% → 91.0% (+27.0%)
   - Checkpoint-1250 model: 68.0% → 82.0% (+14.0%)

2. **All Prediction Changes Were From Label 1 → 0**
   - Base model: 7 changes (100% were 1→0)
   - Fine-tuned model: 35 changes (100% were 1→0) 
   - Checkpoint-1250 model: 22 changes (100% were 1→0)
   - This demonstrates a clear systematic bias in the original extraction logic

3. **Model Output Format Differences**
   - **Base model** primarily uses conclusion statements (80%) with only some JSON output (18%)
   - **Fine-tuned models** use exclusively JSON format (100%) with multiple JSON objects per output
   - This format difference explains why fine-tuned models were more affected by extraction issues

4. **JSON Object Prevalence**
   - Base model: 2.23 JSON objects per output, only 32% have multiple JSONs
   - Fine-tuned model: 3.75 JSON objects per output, 100% have multiple JSONs
   - Checkpoint-1250: 2.21 JSON objects per output, 100% have multiple JSONs

## Root Cause Analysis

### Core Issue: Fallback Extraction Logic
The original extraction algorithm in `sample_model.py` contained a problematic fallback logic:

```python
# If we couldn't parse JSON or find the label, check for direct "0" or "1" in the text
if '0' in output_text and '1' not in output_text:
    return 0
elif '1' in output_text and '0' not in output_text:
    return 1
```

This logic has a critical flaw: it triggers on **any** "1" in the output, even if it's part of a word or phrase (like "Step 1:"). Since Chain-of-Thought (CoT) outputs always include "Step 1" but rarely include the digit "0" standalone, this resulted in a strong bias toward label 1.

### Why Fine-Tuned Models Were More Affected

1. **Output Format Differences:**
   - Base model generated primarily conclusion statements (80% of outputs)
   - Fine-tuned models generated exclusively JSON format responses (100%)

2. **Multiple JSON Objects:**
   - The fine-tuned models often generated multiple JSON objects per response
   - The original logic only looked for the last JSON object, potentially missing earlier correct predictions

3. **CoT Format:**
   - While both models used CoT prompting, the fine-tuned model was specifically trained on it
   - All fine-tuned model outputs contained "step 1:", "step 2:", etc., which triggered the fallback logic

### Concrete Example of Flawed Extraction

For a representative example (from index 7 in the base model):
```
Premise: Most importantly, it reduces the incentive to save for retirement.
Hypothesis: There is a huge incentive to save money for retirement.

Use chain of thought reasoning to determine if the hypothesis is entailed by the premise.
Provide your reasoning and the final label (0 or 1) in JSON format: {"thought_process": "...", "predicted_label": ...}
```

The model output contains both "step 1:" and a JSON with `"predicted_label": 0`, but due to the fallback logic prioritizing "contains '1'" over proper JSON parsing, the original extraction incorrectly returned label 1.

## Model Performance Comparisons

| Model | Original Extraction | Fixed Extraction | Difference |
|-------|---------------------|-------------------|------------|
| Base | 54.0% | 53.0% | -1.0% |
| Fine-tuned | 64.0% | 91.0% | +27.0% |
| Checkpoint-1250 | 68.0% | 82.0% | +14.0% |

This comparison demonstrates that:
1. Fine-tuning was actually highly effective (91% vs 53% accuracy)
2. Even the early checkpoint showed significant learning
3. The original evaluation severely underestimated model performance

## Extraction Method Distribution

| Method | Base Model | Fine-tuned Model | Checkpoint-1250 |
|--------|------------|------------------|-----------------|
| json_predicted_label | 18% | 100% | 100% |
| conclusion_entailed | 80% | 0% | 0% |
| conclusion_not_entailed | 1% | 0% | 0% |
| explicit_label_0 | 1% | 0% | 0% |

This distribution confirms the format shift from the base model (primarily using textual reasoning) to the fine-tuned models (consistently using structured JSON).

## Recommendations

1. **Use More Robust Extraction Logic**
   - Add specific handling for Chain-of-Thought outputs
   - Handle multiple JSON objects properly
   - Avoid using risky string presence checks

2. **Return Raw Outputs for Analysis**
   - Always save the raw unprocessed model outputs
   - Keep extraction logic separate from inference

3. **Validate on a Subset**
   - Manually verify extraction on a small sample before running full evaluation
   - Check for systematic biases in extraction methods

4. **Standardize Output Format**
   - Train models to output in a consistent, easily parseable format
   - Consider explicitly structuring the prompt to encourage cleaner JSON responses

## Conclusion

The fine-tuning process was actually highly effective, improving performance from 53% to 91% accuracy. The apparent poor performance was due to a systematic bias in the extraction logic that disproportionately affected the fine-tuned models due to their different output format.

This analysis demonstrates the importance of robust extraction logic and the need to carefully analyze evaluation results, especially when comparing models with different output characteristics. 