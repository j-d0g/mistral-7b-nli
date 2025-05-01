# Extraction Method Analysis

## Overview

We've analyzed the extraction methods used across different models to understand how our improved prediction extraction algorithm works with different model outputs. This helps explain why the original extraction logic was failing and why our fix led to dramatically different results for the fine-tuned model vs. the base model.

## Key Findings

1. **Extraction Method Differences:**
   - Base model: Used a variety of extraction methods (conclusion statements: 81%, JSON: 18%)
   - Fine-tuned model: Used exclusively JSON format (100% json_predicted_label)
   - Checkpoint-1250: Used exclusively JSON format (100% json_predicted_label)

2. **Direction of Changes:**
   - Across all models, 100% of prediction changes were from 1 → 0
   - No instances of changes from 0 → 1 were found
   - This confirms a strong bias toward predicting "1" in the original extraction logic

3. **Proportion of Changes:**
   - Base model: Only 7% of predictions changed (7/100)
   - Fine-tuned model: 35% of predictions changed (35/100) 
   - Checkpoint-1250: 22% of predictions changed (22/100)
   - This shows the fine-tuned models were much more affected by the extraction bias

## Detailed Extraction Method Breakdown

### Base Model Sample

```
=== Extraction Methods Used ===
conclusion_entailed      :    80 (80.00%)
json_predicted_label     :    18 (18.00%)
conclusion_not_entailed  :     1 (1.00%)
explicit_label_0         :     1 (1.00%)

=== Methods for Label 0 ===
json_predicted_label     :     7 (77.78%)
conclusion_not_entailed  :     1 (11.11%)
explicit_label_0         :     1 (11.11%)

=== Methods for Label 1 ===
conclusion_entailed      :    80 (87.91%)
json_predicted_label     :    11 (12.09%)
```

### Fine-tuned Model Sample

```
=== Extraction Methods Used ===
json_predicted_label     :   100 (100.00%)

=== Methods for Label 0 ===
json_predicted_label     :    47 (100.00%)

=== Methods for Label 1 ===
json_predicted_label     :    53 (100.00%)
```

### Checkpoint-1250 Sample

```
=== Extraction Methods Used ===
json_predicted_label     :   100 (100.00%)

=== Methods for Label 0 ===
json_predicted_label     :    52 (100.00%)

=== Methods for Label 1 ===
json_predicted_label     :    48 (100.00%)
```

## Explanation for Performance Differences

1. **Base Model Format:** 
   - The base model mostly generates text with conclusion statements (81%)
   - Only 18% of outputs use JSON format
   - This mixed format was less affected by the original extraction bias

2. **Fine-tuned Models Format:**
   - Fine-tuned models consistently generate proper JSON format (100%)
   - The original extractor's fallback logic misinterpreted "step 1" as label "1"
   - Since fine-tuned models consistently used CoT with "step 1", they were systematically misinterpreted

3. **Original Extraction Logic Bias:**
   - When no JSON was found, it looked for "contains 1" and "not contains 0"
   - This incorrectly matched "step 1" in chain-of-thought reasoning
   - Explains why 100% of changes were from 1 → 0

4. **Performance Impact:**
   - Base model performance hardly changed (54% → 53%) because its outputs were less affected by the bias
   - Fine-tuned model showed dramatic improvement (64% → 91%) because its outputs were being systematically misinterpreted

## Recommendation

The full test script for checkpoint-1250 has been created (`run_checkpoint_fulltest.sh`). This script:

1. Runs inference on the full test set using checkpoint-1250
2. Applies the improved extraction logic with tracking
3. Saves both the raw and fixed results

This will provide a comprehensive evaluation of the current checkpoint and help track the fine-tuning progress. 