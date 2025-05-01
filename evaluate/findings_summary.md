# Mistral-7B NLI Fine-tuning Analysis

## Summary of Findings

After extensive investigation, we identified several critical issues with the fine-tuning and evaluation pipeline:

1. **Extraction Logic Bias**
   - The original prediction extraction function was heavily biased toward label "1"
   - It used a fallback strategy that checked for "contains 1" and "not contains 0", which incorrectly matched "step 1" in the chain-of-thought reasoning
   - This caused artificially low performance in both base and fine-tuned models
   
2. **Model Output Issues**
   - The fine-tuned model generated multiple JSON objects in a single response
   - The model failed to properly terminate outputs, likely due to training configuration issues
   - Key finding: Setting `pad_token = eos_token` in the training script caused the model to never see the EOS token during training
   
3. **Training Configuration**
   - Fixed the `tokenizer.pad_token` configuration to use a dedicated pad token ('[PAD]') instead of reusing the EOS token
   - This fix should improve the model's ability to generate properly terminated text
   
4. **Actual Performance**
   - After fixing extraction logic:
     - Base model: 53% accuracy
     - Fine-tuned model: 91% accuracy
     - Checkpoint-1250 model: 82% accuracy
   - The fine-tuned model shows a 38% improvement over the base model
   - Even the early checkpoint (1250) shows significant learning (+29% over base)

## Diagnostic Scripts

### 1. Improved Prediction Extraction (`fix_predictions.py`)

Created a robust extraction function to correctly parse model outputs, overcoming issues with multiple JSONs, malformed outputs, and biased fallback logic:

```python
def improved_extract_prediction(output_text, use_cot=False):
    """More robust function to extract the prediction (0 or 1) from model output."""
    
    output_text = output_text.strip()
    
    # Look for JSON objects - prioritize the one with "predicted_label"
    json_texts = []
    depth = 0
    start_idx = -1
    
    # Find all potential JSON objects
    for i, char in enumerate(output_text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx != -1:
                json_texts.append(output_text[start_idx:i+1])
    
    # Try each JSON object, prioritizing those with the expected keys
    for json_text in json_texts:
        try:
            response_dict = json.loads(json_text)
            
            # Check for predicted_label first, then label
            if 'predicted_label' in response_dict:
                label_value = response_dict['predicted_label']
            elif 'label' in response_dict:
                label_value = response_dict['label']
            else:
                continue  # Skip this JSON if it doesn't have either key
            
            # Handle string or number
            if isinstance(label_value, str):
                # For strings, do targeted extraction
                if label_value == "0" or label_value == "no entailment" or label_value.lower() == "false":
                    return 0
                elif label_value == "1" or label_value == "entailment" or label_value.lower() == "true":
                    return 1
                try:
                    return int(label_value)
                except:
                    # If we can't parse directly, continue to the next JSON
                    continue
            else:
                # For numbers, convert to int
                return int(label_value)
        except:
            # If this JSON fails, try the next one
            continue
    
    # Remaining fallback strategies...
```

### 2. Run Comparison (`compare_runs.py`)

Created a script to compare results between different runs to identify inconsistencies:

```python
def compare_results(sample_file, full_file):
    """
    Compares predictions and outputs for samples present in both result files.
    """
    # Load the result files
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    with open(full_file, 'r') as f:
        full_data = json.load(f)
    
    # Extract sample and full results
    sample_results = sample_data.get('results', [])
    full_results = full_data.get('results', [])
    
    # Map sample results by example ID
    sample_by_id = {result.get('id', i): result for i, result in enumerate(sample_results)}
    
    # Find matching examples
    matches = []
    for i, full_result in enumerate(full_results):
        full_id = full_result.get('id', i)
        if full_id in sample_by_id:
            matches.append((sample_by_id[full_id], full_result))
    
    # Compare matches
    predictions_differ = 0
    outputs_differ = 0
    
    for sample_result, full_result in matches:
        if sample_result.get('predicted_label') != full_result.get('predicted_label'):
            predictions_differ += 1
        if sample_result.get('output') != full_result.get('output'):
            outputs_differ += 1
    
    # Print statistics
    print(f"Found {len(matches)} matching examples")
    print(f"Prediction differences: {predictions_differ} ({predictions_differ/len(matches)*100:.2f}%)")
    print(f"Output differences: {outputs_differ} ({outputs_differ/len(matches)*100:.2f}%)")
```

### 3. Results Comparison (`compare_fixed_results.py`)

Created a script to compare the fixed results between different models:

```python
def compare_files(base_file, finetuned_file):
    """Compare base and fine-tuned model results."""
    base_data = load_results(base_file)
    finetuned_data = load_results(finetuned_file)
    
    # Get overall accuracy
    base_acc = base_data.get('accuracy', 'N/A')
    finetuned_acc = finetuned_data.get('accuracy', 'N/A')
    
    print(f"\n=== Overall Accuracy Comparison ===")
    print(f"Base model:     {base_acc:.4f}")
    print(f"Fine-tuned:     {finetuned_acc:.4f}")
    print(f"Improvement:    {(finetuned_acc - base_acc):.4f}\n")
    
    # Compare predictions on same examples
    base_results = {i: r for i, r in enumerate(base_data.get('results', []))}
    ft_results = {i: r for i, r in enumerate(finetuned_data.get('results', []))}
    
    # Find common indices
    common_indices = set(base_results.keys()) & set(ft_results.keys())
    
    # Analyze differences
    differences = Counter()
    
    for idx in common_indices:
        base_result = base_results[idx]
        ft_result = ft_results[idx]
        
        base_correct = base_result.get('correct', False)
        ft_correct = ft_result.get('correct', False)
        
        if base_correct and not ft_correct:
            differences['base_right_ft_wrong'] += 1
        elif not base_correct and ft_correct:
            differences['base_wrong_ft_right'] += 1
        elif base_correct and ft_correct:
            differences['both_right'] += 1
        else:
            differences['both_wrong'] += 1
```

### 4. Training Script Fix

Modified the tokenizer configuration in `scripts/train_sft.py` to use a separate pad token instead of reusing the EOS token:

```python
# Original code - problematic
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# Fixed code
if tokenizer.pad_token is None:
    logger.info("Adding new pad token '[PAD]'")
    # Add the token. `special_tokens_map.json` in the save dir will be updated.
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
    # Keep the pad_token_id consistent if the tokenizer already assigned one.
    # Otherwise, use the one from the added token.
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
```

## Key Performance Results

| Model | Original Accuracy | Fixed Accuracy | Improvement |
|-------|------------------|----------------|-------------|
| Base Model | ~54% | 53% | -1% |
| Fine-tuned Model | ~64% | 91% | +27% |
| Checkpoint-1250 | N/A | 82% | N/A |

## Recommendations

1. **Continue the training** with the fixed pad token configuration
2. **Always use a separate validation set** during training to detect issues early
3. **Check model output patterns** before relying on automated metrics
4. **Implement more robust prediction extraction** for evaluation
5. **Avoid setting pad_token = eos_token** when fine-tuning LLMs

The significant improvement (91% vs 53% accuracy) demonstrates that the fine-tuning was actually effective, but the evaluation was flawed due to extraction issues. 