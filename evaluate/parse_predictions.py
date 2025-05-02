#!/usr/bin/env python3
"""
Unified NLI Prediction Parser

This script provides robust parsing of NLI model outputs to extract structured predictions.
It consolidates functionality from multiple parsing approaches:

1. Standard parsing - Extract predictions using robust, balanced extraction logic
2. Detailed tracking - Track which extraction methods are used for detailed analysis
3. Strict academic - Use rigid academic-style extraction (JSON-only, no fallbacks)

Usage:
    python parse_predictions.py input_file.json [--output output_file.json] [OPTIONS]

Options:
    --tracking      Enable detailed extraction method tracking
    --strict        Use strict academic evaluation (JSON-only, no fallbacks)
    --summarize     Only print summary for already processed files
    --use_cot       Enable CoT-specific extraction (default: True)
"""

import json
import sys
import os
import re
import argparse
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, precision_recall_fscore_support

def standard_extract_prediction(output_text, use_cot=False):
    """
    Standard prediction extraction with robust handling of different formats.
    Returns only the prediction (0 or 1).
    """
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
    
    # If no valid JSON with label found, check for explicit label statements in text
    lower_text = output_text.lower()
    if "final label: 0" in lower_text or "final prediction: 0" in lower_text:
        return 0
    if "final label: 1" in lower_text or "final prediction: 1" in lower_text:
        return 1
        
    # Only if nothing better is found, check for conclusion statements
    if "not entailed" in lower_text or "no entailment" in lower_text:
        return 0
    if "is entailed" in lower_text or "entailment" in lower_text:
        return 1
    
    # For CoT reasoning, try to extract from "step 3" conclusion
    if use_cot and "step 3" in lower_text:
        step3_text = lower_text.split("step 3")[1].split("step 4")[0].split("\n\n")[0]
        if "not entailed" in step3_text or "no entailment" in step3_text:
            return 0
        if "is entailed" in step3_text or "entailment" in step3_text:
            return 1
    
    # If all else fails, return -1 to signal extraction failure
    return -1  # This will help us identify problem cases

def tracking_extract_prediction(output_text, use_cot=False):
    """
    Extract prediction with method tracking for detailed analysis.
    Returns a tuple of (prediction, method_used).
    """
    output_text = output_text.strip()
    method_used = "unknown"
    
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
                method_used = "json_predicted_label"
            elif 'label' in response_dict:
                label_value = response_dict['label']
                method_used = "json_label"
            else:
                continue  # Skip this JSON if it doesn't have either key
            
            # Handle string or number
            if isinstance(label_value, str):
                # For strings, do targeted extraction
                if label_value == "0" or label_value == "no entailment" or label_value.lower() == "false":
                    return 0, method_used
                elif label_value == "1" or label_value == "entailment" or label_value.lower() == "true":
                    return 1, method_used
                try:
                    return int(label_value), method_used
                except:
                    # If we can't parse directly, continue to the next JSON
                    continue
            else:
                # For numbers, convert to int
                return int(label_value), method_used
        except:
            # If this JSON fails, try the next one
            continue
    
    # If no valid JSON with label found, check for explicit label statements in text
    lower_text = output_text.lower()
    if "final label: 0" in lower_text or "final prediction: 0" in lower_text:
        method_used = "explicit_label_0"
        return 0, method_used
    if "final label: 1" in lower_text or "final prediction: 1" in lower_text:
        method_used = "explicit_label_1"
        return 1, method_used
        
    # Only if nothing better is found, check for conclusion statements
    if "not entailed" in lower_text or "no entailment" in lower_text:
        method_used = "conclusion_not_entailed"
        return 0, method_used
    if "is entailed" in lower_text or "entailment" in lower_text:
        method_used = "conclusion_entailed"
        return 1, method_used
    
    # For CoT reasoning, try to extract from "step 3" conclusion
    if use_cot and "step 3" in lower_text:
        step3_text = lower_text.split("step 3")[1].split("step 4")[0].split("\n\n")[0]
        if "not entailed" in step3_text or "no entailment" in step3_text:
            method_used = "cot_step3_not_entailed"
            return 0, method_used
        if "is entailed" in step3_text or "entailment" in step3_text:
            method_used = "cot_step3_entailed"
            return 1, method_used
    
    # If all else fails, return -1 to signal extraction failure
    method_used = "extraction_failure"
    return -1, method_used

def strict_extract_prediction(output_text):
    """
    Strict academic extraction using only valid JSON with explicit label field.
    Returns (prediction, method, success_flag)
    """
    # Look for JSON objects
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
    
    # Try each JSON object
    for json_text in json_texts:
        try:
            parsed = json.loads(json_text)
            
            # Check for predicted_label field
            if 'predicted_label' in parsed:
                label_value = parsed['predicted_label']
                
                # Handle different value formats
                if isinstance(label_value, (int, float)):
                    # Numeric value
                    if label_value == 0 or label_value == 1:
                        return int(label_value), "json_predicted_label", True
                elif isinstance(label_value, str):
                    # String value
                    if label_value in ["0", "no entailment", "false"]:
                        return 0, "json_predicted_label", True
                    elif label_value in ["1", "entailment", "true"]:
                        return 1, "json_predicted_label", True
                    
                    # Try to convert string to int
                    try:
                        value = int(label_value)
                        if value in [0, 1]:
                            return value, "json_predicted_label", True
                    except:
                        pass
            
            # Check for label field as fallback
            elif 'label' in parsed:
                label_value = parsed['label']
                
                # Handle different value formats
                if isinstance(label_value, (int, float)):
                    # Numeric value
                    if label_value == 0 or label_value == 1:
                        return int(label_value), "json_label", True
                elif isinstance(label_value, str):
                    # String value
                    if label_value in ["0", "no entailment", "false"]:
                        return 0, "json_label", True
                    elif label_value in ["1", "entailment", "true"]:
                        return 1, "json_label", True
                    
                    # Try to convert string to int
                    try:
                        value = int(label_value)
                        if value in [0, 1]:
                            return value, "json_label", True
                    except:
                        pass
        except:
            # Invalid JSON, skip
            continue
    
    # If no valid JSON with clear label found, return failure
    return None, "extraction_failure", False

def reanalyze_results(input_file, output_file=None, use_cot=True, tracking=False, strict=False):
    """Re-analyze an existing results file with different extraction options."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load input file {input_file}: {e}")
        return False

    if 'results' not in data:
        print(f"Error: Input file {input_file} does not have a 'results' field")
        return False

    # Extract only the metadata we want to keep
    results = data.pop('results')  # Remove results temporarily
    metadata = {k: v for k, v in data.items() if k in ['model', 'inference_time_seconds', 'samples_per_second', 'use_cot']}
    
    # Build a fresh data structure with just the metadata
    data = metadata
    
    # Get the analysis method based on flags - only use one flag at a time
    if strict:
        extract_fn = strict_extract_prediction
        tracking = False  # Disable tracking if strict is enabled
    elif tracking:
        extract_fn = tracking_extract_prediction
    else:
        extract_fn = standard_extract_prediction
    
    # Process and update each result
    y_true = []
    y_pred = []
    method_counts = Counter()
    method_counts_by_label = defaultdict(Counter)
    strict_failures = 0
    prediction_changes = 0
    original_correct = 0
    new_correct = 0
    label_0_to_1 = 0
    label_1_to_0 = 0
    successful_extractions = 0
    
    # Process each result
    for i, result in enumerate(results):
        # Skip entries without outputs
        if 'output' not in result:
            continue
        
        # Get the original prediction
        original_prediction = result.get('predicted_label', -1)
        
        # Has true label?
        has_true_label = 'true_label' in result
        if has_true_label:
            true_label = result['true_label']
            y_true.append(true_label)
            if original_prediction == true_label:
                original_correct += 1
        
        # Extract prediction using the selected method
        if strict:
            # Strict extraction returns (prediction, method, success)
            new_prediction, method, success = extract_fn(result['output'])
            if not success:
                strict_failures += 1
                print(f"Strict extraction failure for example {i}")
                method = "extraction_failure"
                if has_true_label:
                    y_pred.append(-1)  # Mark as extraction failure
            else:
                successful_extractions += 1
                if has_true_label:
                    y_pred.append(new_prediction)
                    if new_prediction == true_label:
                        new_correct += 1
            
            # Record method usage
            method_counts[method] += 1
            if has_true_label:
                method_counts_by_label[true_label][method] += 1
            
            # Check for prediction changes
            if original_prediction != new_prediction and new_prediction != -1:
                prediction_changes += 1
                print(f"Example {i}: Changed prediction from {original_prediction} to {new_prediction}, true label: {true_label if has_true_label else 'unknown'}")
                
                # Track direction of change
                if original_prediction == 0 and new_prediction == 1:
                    label_0_to_1 += 1
                elif original_prediction == 1 and new_prediction == 0:
                    label_1_to_0 += 1
            
            # Update the result with the new prediction
            if success:
                result['predicted_label'] = new_prediction
                result['extraction_method'] = method
            else:
                # For strict mode, we keep the original prediction on extraction failure
                result['extraction_method'] = "extraction_failure"
        
        elif tracking:
            # Tracking extraction returns (prediction, method)
            new_prediction, method = extract_fn(result['output'], use_cot)
            
            # Record method usage
            method_counts[method] += 1
            if has_true_label:
                y_pred.append(new_prediction if new_prediction != -1 else original_prediction)
                method_counts_by_label[true_label][method] += 1
                if new_prediction == true_label:
                    new_correct += 1
            
            # Update the result with the extraction method
            result['extraction_method'] = method
            
            # Update prediction if extraction succeeded
            if new_prediction != -1:
                successful_extractions += 1
                result['predicted_label'] = new_prediction
        
        else:
            # Standard extraction returns just the prediction
            new_prediction = extract_fn(result['output'], use_cot)
            
            if new_prediction != -1:
                successful_extractions += 1
                result['predicted_label'] = new_prediction
                if has_true_label:
                    y_pred.append(new_prediction)
                    if new_prediction == true_label:
                        new_correct += 1
            else:
                # Keep original prediction on extraction failure
                if has_true_label:
                    y_pred.append(original_prediction)
    
    # Calculate metrics for successful extractions
    metrics = {}
    total_samples = len(results)
    
    if len(y_true) > 0 and len(y_pred) > 0:
        # Filter out failed extractions
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
        valid_y_true = [y_true[i] for i in valid_indices]
        valid_y_pred = [y_pred[i] for i in valid_indices]
        
        if len(valid_y_true) > 0:
            # Calculate 4 core metrics
            accuracy = new_correct / len(valid_y_true) if len(valid_y_true) > 0 else 0
            precision, recall, f1, _ = precision_recall_fscore_support(
                valid_y_true, valid_y_pred, average='macro'
            )
            
            # Add the 4 core metrics
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
    
    # Add metrics to top level (BEFORE results)
    if metrics:
        data.update(metrics)
    
    # Add results 
    data['results'] = results
    
    # Prepare extraction stats (to be added AFTER results)
    extraction_stats = {}
    if strict:
        # For strict mode, report strict accuracy on successful extractions
        strict_accuracy = new_correct / successful_extractions if successful_extractions > 0 else 0
        
        extraction_stats = {
            'total_samples': total_samples,
            'successful_extractions': successful_extractions,
            'extraction_failures': strict_failures,
            'extraction_failure_rate': strict_failures / total_samples if total_samples > 0 else 0,
            'prediction_changes': prediction_changes,
            'prediction_change_rate': prediction_changes / total_samples if total_samples > 0 else 0,
            'changed_0_to_1': label_0_to_1,
            'changed_1_to_0': label_1_to_0,
            'original_accuracy': original_correct / total_samples if total_samples > 0 else 0,
            'new_accuracy': new_correct / total_samples if total_samples > 0 else 0,
            'strict_accuracy': strict_accuracy,
            'extraction_methods': {method: count for method, count in method_counts.items()},
            'methods_by_label': {label: dict(counts) for label, counts in method_counts_by_label.items()}
        }
    else:
        # For standard or tracking mode
        extraction_stats = {
            'total_samples': total_samples,
            'successful_extractions': successful_extractions,
            'extraction_success_rate': successful_extractions / total_samples if total_samples > 0 else 0,
            'original_accuracy': original_correct / total_samples if total_samples > 0 else 0,
            'new_accuracy': new_correct / total_samples if total_samples > 0 else 0
        }
        
        if tracking:
            extraction_stats['extraction_methods'] = {method: count for method, count in method_counts.items()}
            extraction_stats['methods_by_label'] = {label: dict(counts) for label, counts in method_counts_by_label.items()}
    
    # Add extraction stats AFTER results
    data['extraction_stats'] = extraction_stats
    
    # Determine output file path
    if output_file is None:
        # Create default output file name
        base_path = os.path.splitext(input_file)[0]
        suffix = "_strict" if strict else "_tracked" if tracking else "_parsed"
        output_file = f"{base_path}{suffix}.json"
    
    # Save updated results
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated results saved to: {output_file}")
    except Exception as e:
        print(f"Error: Failed to save output file {output_file}: {e}")
        return False
    
    # Print summary
    print("\n=== Re-analysis Complete ===")
    print(f"Total samples: {total_samples}")
    print(f"Successful extractions: {successful_extractions}")
    if strict:
        print(f"Extraction failures: {strict_failures} ({strict_failures/total_samples*100:.2f}%)")
        print()
        print(f"Prediction changes: {prediction_changes} ({prediction_changes/total_samples*100:.2f}%)")
        print(f"Changes 0->1: {label_0_to_1} ({label_0_to_1/prediction_changes*100:.2f}% of changes)")
        print(f"Changes 1->0: {label_1_to_0} ({label_1_to_0/prediction_changes*100:.2f}% of changes)")
        print()
        print(f"Accuracy before: {original_correct/total_samples:.4f} ({original_correct}/{total_samples})")
        print(f"Accuracy after:  {new_correct/total_samples:.4f} ({new_correct}/{total_samples})")
        print(f"Strict accuracy (on successful extractions): {strict_accuracy:.4f} ({new_correct}/{successful_extractions})")
    else:
        print(f"Extraction failures: {total_samples - successful_extractions} ({(total_samples - successful_extractions)/total_samples*100:.2f}%)")
        if len(y_true) > 0:
            accuracy = new_correct/len(y_true)
            print(f"Accuracy: {accuracy:.4f} ({new_correct}/{len(y_true)})")
            if 'precision' in metrics:
                print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    # Print classification report for successful extractions if we have labels
    if len(y_true) > 0 and len(y_pred) > 0:
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
        valid_y_true = [y_true[i] for i in valid_indices]
        valid_y_pred = [y_pred[i] for i in valid_indices]
        
        if len(valid_y_true) > 0:
            print(f"\n=== Classification Metrics (on successful extractions) ===")
            print(classification_report(valid_y_true, valid_y_pred, labels=[0, 1], 
                                       target_names=["no entailment (0)", "entailment (1)"]))
    
    # Print extraction methods if tracking
    if tracking or strict:
        print("\n=== Extraction Methods Used ===")
        total = sum(method_counts.values())
        for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{method:20s}: {count:5d} ({count/total*100:.2f}%)")
        
        # Print methods by predicted label
        if 0 in method_counts_by_label:
            print("\n=== Methods for Label 0 ===")
            for method, count in sorted(method_counts_by_label[0].items(), key=lambda x: x[1], reverse=True):
                print(f"{method:20s}: {count:5d}")
        
        if 1 in method_counts_by_label:
            print("\n=== Methods for Label 1 ===")
            for method, count in sorted(method_counts_by_label[1].items(), key=lambda x: x[1], reverse=True):
                print(f"{method:20s}: {count:5d}")
    
    return True

def summarize_extraction_stats(input_file):
    """Print a summary of extraction statistics from an already processed file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        stats = data.get('extraction_stats', {})
        if not stats:
            print("No extraction statistics found in this file. Run with --tracking first.")
            return
            
        print(f"\n=== Extraction Methods Summary for {input_file} ===")
        
        methods = stats.get('methods', {})
        label_0_methods = stats.get('label_0_methods', {})
        label_1_methods = stats.get('label_1_methods', {})
        changes_1_to_0 = stats.get('changes_1_to_0', 0)
        changes_0_to_1 = stats.get('changes_0_to_1', 0)
        extraction_failures = stats.get('extraction_failures', 0)
        
        total_samples = sum(methods.values()) if methods else 0
        
        print(f"Total samples: {total_samples}")
        print(f"Changes 1->0: {changes_1_to_0}")
        print(f"Changes 0->1: {changes_0_to_1}")
        print(f"Extraction failures: {extraction_failures} ({stats.get('failure_rate', 0)*100:.2f}%)")
        
        print("\n=== Extraction Methods Used ===")
        for method, count in sorted(methods.items(), key=lambda x: x[1], reverse=True):
            percent = count/total_samples*100 if total_samples > 0 else 0
            print(f"{method:<25}: {count:5} ({percent:.2f}%)")
        
        print("\n=== Methods for Label 0 ===")
        total_0s = sum(label_0_methods.values()) if label_0_methods else 0
        for method, count in sorted(label_0_methods.items(), key=lambda x: x[1], reverse=True):
            percent = count/total_0s*100 if total_0s > 0 else 0
            print(f"{method:<25}: {count:5} ({percent:.2f}%)")
        
        print("\n=== Methods for Label 1 ===")
        total_1s = sum(label_1_methods.values()) if label_1_methods else 0
        for method, count in sorted(label_1_methods.items(), key=lambda x: x[1], reverse=True):
            percent = count/total_1s*100 if total_1s > 0 else 0
            print(f"{method:<25}: {count:5} ({percent:.2f}%)")
            
    except Exception as e:
        print(f"Error loading or processing file: {e}")

def compare_models(file_paths):
    """Print a comparison of extraction results across multiple models."""
    if len(file_paths) < 2:
        print("Need at least 2 files to compare.")
        return
    
    all_stats = {}
    
    for file_path in file_paths:
        model_name = os.path.basename(file_path).replace('.json', '')
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract relevant stats
            accuracy = data.get('accuracy', 0)
            strict_accuracy = data.get('strict_accuracy', None)
            stats = data.get('extraction_stats', {})
            class_metrics = data.get('classification_metrics', {})
            f1_macro = class_metrics.get('macro avg', {}).get('f1-score', None)
            f1_weighted = class_metrics.get('weighted avg', {}).get('f1-score', None)
            precision_1 = class_metrics.get('entailment (1)', {}).get('precision', None)
            recall_1 = class_metrics.get('entailment (1)', {}).get('recall', None)
            f1_1 = class_metrics.get('entailment (1)', {}).get('f1-score', None)

            model_data = {
                'accuracy': accuracy,
                'extraction_methods': stats.get('methods', {}),
                'label_0_methods': stats.get('label_0_methods', {}),
                'label_1_methods': stats.get('label_1_methods', {}),
                'changes_1_to_0': stats.get('changes_1_to_0', 0),
                'changes_0_to_1': stats.get('changes_0_to_1', 0),
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'precision_1': precision_1,
                'recall_1': recall_1,
                'f1_1': f1_1
            }
            if strict_accuracy is not None:
                model_data['strict_accuracy'] = strict_accuracy
                model_data['extraction_failures'] = stats.get('extraction_failures', 0)
                model_data['failure_rate'] = stats.get('failure_rate', 0)

            all_stats[model_name] = model_data
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_stats:
        print("No valid stats found in the provided files.")
        return
    
    # Print comparison
    model_names = list(all_stats.keys())
    
    print("\n=== Model Comparison ===")
    
    # Accuracy and F1 comparison
    print("\nAccuracy & F1 Comparison:")
    
    has_strict = any('strict_accuracy' in all_stats[m] for m in model_names)
    has_f1 = any(all_stats[m].get('f1_macro') is not None for m in model_names)

    headers = ["Model", "Accuracy"]
    if has_strict: headers.append("StrictAcc")
    if has_f1: headers.extend(["F1-Macro", "F1-Wgt", "F1 (1)"])
    if has_strict: headers.append("FailRate")
    
    col_widths = [30, 10]
    if has_strict: col_widths.append(10)
    if has_f1: col_widths.extend([10, 10, 10])
    if has_strict: col_widths.append(10)

    header_fmt = "".join([f"{{{i}:<{w}}}" for i, w in enumerate(col_widths)])
    print(header_fmt.format(*headers))
    print("-" * sum(col_widths))

    for model in model_names:
        stats = all_stats[model]
        cols = [model]
        cols.append(f"{stats.get('accuracy', 0)*100:.1f}%")
        
        if has_strict:
            strict_acc = stats.get('strict_accuracy')
            cols.append(f"{strict_acc*100:.1f}%" if strict_acc is not None else "N/A")
            
        if has_f1:
            f1_macro = stats.get('f1_macro')
            f1_wgt = stats.get('f1_weighted')
            f1_1 = stats.get('f1_1')
            cols.append(f"{f1_macro:.3f}" if f1_macro is not None else "N/A")
            cols.append(f"{f1_wgt:.3f}" if f1_wgt is not None else "N/A")
            cols.append(f"{f1_1:.3f}" if f1_1 is not None else "N/A")

        if has_strict:
            fail_rate = stats.get('failure_rate')
            cols.append(f"{fail_rate*100:.1f}%" if fail_rate is not None else "N/A")
            
        print(header_fmt.format(*cols))

    # Print prediction changes
    if all('changes_1_to_0' in all_stats[m] for m in model_names):
        print("\nPrediction Changes:")
        headers = ["Model", "1->0 Changes", "0->1 Changes"]
        print(f"{headers[0]:<30} {headers[1]:<15} {headers[2]:<15}")
        print("-" * 65)
        
        for model in model_names:
            stats = all_stats[model]
            changes_1_0 = stats.get('changes_1_to_0', 0)
            changes_0_1 = stats.get('changes_0_to_1', 0)
            print(f"{model:<30} {changes_1_0:<15} {changes_0_1:<15}")
    
    # Print extraction methods if available for all
    if all('extraction_methods' in all_stats[m] for m in model_names):
        # Collect all unique method names
        all_methods = set()
        for model in model_names:
            all_methods.update(all_stats[model]['extraction_methods'].keys())
        
        print("\nExtraction Methods Comparison:")
        # Print header
        print(f"{'Method':<25}", end="")
        for model in model_names:
            print(f" {model:<15}", end="")
        print()
        print("-" * (25 + 15 * len(model_names)))
        
        # Print each method
        for method in sorted(all_methods):
            print(f"{method:<25}", end="")
            for model in model_names:
                methods = all_stats[model]['extraction_methods']
                count = methods.get(method, 0)
                total = sum(methods.values()) if methods else 1
                percent = count / total * 100 if total > 0 else 0
                print(f" {count:3d} ({percent:5.1f}%)", end="")
            print()

def main():
    parser = argparse.ArgumentParser(description="Unified NLI model output parser")
    parser.add_argument("files", nargs="+", help="Input JSON result file(s) to process")
    parser.add_argument("--output", type=str, help="Path to save updated results (optional)")
    parser.add_argument("--tracking", action="store_true", help="Enable detailed extraction method tracking")
    parser.add_argument("--strict", action="store_true", help="Use strict academic evaluation (JSON-only)")
    parser.add_argument("--use_cot", action="store_true", default=True, help="Enable CoT-specific extraction (default: True)")
    parser.add_argument("--summarize", action="store_true", help="Only print summary for already processed files")
    parser.add_argument("--compare", action="store_true", help="Compare results across multiple files")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple result files
        compare_models(args.files)
        return
    
    # Process each file
    for input_file in args.files:
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
            continue
    
        if args.summarize:
            # Only print summary
            summarize_extraction_stats(input_file)
        else:
            # Generate output file name if not specified
            output_file = args.output
            if not output_file and len(args.files) == 1:
                # For single file, create default output name
                base_dir = os.path.dirname(input_file)
                base_name = os.path.basename(input_file).replace('.json', '')
                suffix = '_strict' if args.strict else '_parsed'
                output_file = os.path.join(base_dir, f"{base_name}{suffix}.json")
                print(f"Will save updated results to: {output_file}")
    
            # Process the file
            print(f"Processing: {input_file}")
            reanalyze_results(
                input_file, 
                output_file if len(args.files) == 1 else None,  # Only use output for single file
                args.use_cot,
                args.tracking,
                args.strict
            )

if __name__ == "__main__":
    main() 