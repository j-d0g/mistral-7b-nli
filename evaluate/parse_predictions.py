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
    """
    Re-analyze model outputs with selected extraction method and save updated results.
    
    Parameters:
        input_file: Path to input JSON results file
        output_file: Optional path to save updated results
        use_cot: Enable CoT-specific extraction
        tracking: Enable detailed method tracking
        strict: Use strict academic evaluation
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        return
        
    results = data.get('results', [])
    print(f"Loaded {len(results)} results from {input_file}")
    
    # Track statistics
    extraction_failures = 0
    prediction_changes = 0
    correct_before = 0
    correct_after = 0
    
    # Tracking-specific counters
    if tracking or strict:
        extraction_methods = Counter()
        label_0_methods = Counter()
        label_1_methods = Counter()
        changes_1_to_0 = 0
        changes_0_to_1 = 0
    
    # Process each result
    for idx, result in enumerate(results):
        original_pred = result.get('predicted_label', -999)
        output_text = result.get('output', '')
        true_label = result.get('true_label', -999)
        
        # Skip if missing essential data
        if output_text == '' or true_label == -999:
            print(f"Warning: Missing data in result {idx}")
            continue
            
        # Count correct predictions with original extraction method
        if original_pred == true_label:
            correct_before += 1
            
        # Apply selected extraction method
        try:
            if strict:
                # Use strict academic extraction
                new_pred, method_used, success = strict_extract_prediction(output_text)
                if not success:
                    extraction_failures += 1
                    print(f"Strict extraction failure for example {idx}")
                    # In strict mode, extraction failures are counted as incorrect
                    new_pred = None
                
                if tracking or strict:
                    extraction_methods[method_used] += 1
                    
            elif tracking:
                # Use tracking extraction
                new_pred, method_used = tracking_extract_prediction(output_text, use_cot)
                
            if new_pred == -1:
                extraction_failures += 1
                print(f"Extraction failure for example {idx}, keeping original prediction")
                    new_pred = original_pred
                
                extraction_methods[method_used] += 1
                
                # Track methods by label
                if new_pred == 0:
                    label_0_methods[method_used] += 1
                elif new_pred == 1:
                    label_1_methods[method_used] += 1
                
            else:
                # Use standard extraction
                new_pred = standard_extract_prediction(output_text, use_cot)
                
                if new_pred == -1:
                    extraction_failures += 1
                    print(f"Extraction failure for example {idx}, keeping original prediction")
                    new_pred = original_pred
            
            # Track prediction changes
            if new_pred != original_pred:
                prediction_changes += 1
                
                # Track direction of change for tracking mode
                if tracking or strict:
                    if original_pred == 1 and new_pred == 0:
                        changes_1_to_0 += 1
                    elif original_pred == 0 and new_pred == 1:
                        changes_0_to_1 += 1
                
                print(f"Example {idx}: Changed prediction from {original_pred} to {new_pred}, true label: {true_label}")
                
            # Update the result
            if new_pred is not None:  # Skip None predictions from strict mode
            result['predicted_label'] = new_pred
            result['correct'] = (new_pred == true_label)
            
                # Add method information for tracking mode
                if tracking or strict:
                    result['extraction_method'] = method_used
            
            # Count correct predictions with new extraction method
            if new_pred == true_label:
                correct_after += 1
                
        except Exception as e:
            print(f"Error processing result {idx}: {e}", file=sys.stderr)
    
    # Calculate accuracy based on mode
    total_samples = len(results)
    
    if strict:
        # In strict mode, we only count examples with successful extraction
        successful_extractions = total_samples - extraction_failures
        new_accuracy = correct_after / total_samples if total_samples > 0 else 0
        strict_accuracy = correct_after / successful_extractions if successful_extractions > 0 else 0
        data['strict_accuracy'] = strict_accuracy
    else:
        # Regular accuracy calculation
    new_accuracy = correct_after / total_samples if total_samples > 0 else 0
    
    # Update accuracy in the main data
    data['accuracy'] = new_accuracy
    
    # Add extraction method statistics for tracking mode
    if tracking or strict:
        data['extraction_stats'] = {
            'methods': dict(extraction_methods),
            'label_0_methods': dict(label_0_methods),
            'label_1_methods': dict(label_1_methods),
            'changes_1_to_0': changes_1_to_0,
            'changes_0_to_1': changes_0_to_1,
            'extraction_failures': extraction_failures,
            'failure_rate': extraction_failures / total_samples if total_samples > 0 else 0
        }
    
    # Save updated results if output file specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Updated results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving output file: {e}", file=sys.stderr)
    
    # Report statistics
    print("\n=== Re-analysis Complete ===")
    print(f"Total samples: {total_samples}")
    print(f"Extraction failures: {extraction_failures} ({extraction_failures/total_samples*100:.2f}%)")
    print(f"Prediction changes: {prediction_changes} ({prediction_changes/total_samples*100:.2f}%)")
    
    if tracking or strict:
        print(f"Changes 1->0: {changes_1_to_0} ({changes_1_to_0/prediction_changes*100:.2f}% of changes)" if prediction_changes > 0 else "Changes 1->0: 0 (0%)")
        print(f"Changes 0->1: {changes_0_to_1} ({changes_0_to_1/prediction_changes*100:.2f}% of changes)" if prediction_changes > 0 else "Changes 0->1: 0 (0%)")
    
    print(f"\nAccuracy before: {correct_before/total_samples:.4f} ({correct_before}/{total_samples})")
    print(f"Accuracy after:  {correct_after/total_samples:.4f} ({correct_after}/{total_samples})")
    
    if strict:
        print(f"Strict accuracy: {strict_accuracy:.4f} (counting only successful extractions)")
    
    if tracking or strict:
        print("\n=== Extraction Methods Used ===")
        for method, count in extraction_methods.most_common():
            percent = count/total_samples*100
            print(f"{method:<25}: {count:5} ({percent:.2f}%)")
        
        print("\n=== Methods for Label 0 ===")
        total_0s = sum(label_0_methods.values())
        for method, count in label_0_methods.most_common():
            percent = count/total_0s*100 if total_0s > 0 else 0
            print(f"{method:<25}: {count:5} ({percent:.2f}%)")
        
        print("\n=== Methods for Label 1 ===")
        total_1s = sum(label_1_methods.values())
        for method, count in label_1_methods.most_common():
            percent = count/total_1s*100 if total_1s > 0 else 0
            print(f"{method:<25}: {count:5} ({percent:.2f}%)")
    
    return {
        'total': total_samples,
        'original_correct': correct_before,
        'improved_correct': correct_after,
        'accuracy_before': correct_before/total_samples if total_samples > 0 else 0,
        'accuracy_after': correct_after/total_samples if total_samples > 0 else 0,
        'changes': prediction_changes,
        'extraction_failures': extraction_failures
    }

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
            
            if strict_accuracy is not None:
                # This is a strict evaluation file
                all_stats[model_name] = {
                    'accuracy': accuracy,
                    'strict_accuracy': strict_accuracy,
                    'extraction_methods': stats.get('methods', {}),
                    'label_0_methods': stats.get('label_0_methods', {}),
                    'label_1_methods': stats.get('label_1_methods', {}),
                    'changes_1_to_0': stats.get('changes_1_to_0', 0),
                    'changes_0_to_1': stats.get('changes_0_to_1', 0),
                    'extraction_failures': stats.get('extraction_failures', 0),
                    'failure_rate': stats.get('failure_rate', 0)
                }
            else:
                # Standard evaluation file
                all_stats[model_name] = {
                    'accuracy': accuracy,
                    'extraction_methods': stats.get('methods', {}),
                    'label_0_methods': stats.get('label_0_methods', {}),
                    'label_1_methods': stats.get('label_1_methods', {}),
                    'changes_1_to_0': stats.get('changes_1_to_0', 0),
                    'changes_0_to_1': stats.get('changes_0_to_1', 0)
                }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not all_stats:
        print("No valid stats found in the provided files.")
        return
    
    # Print comparison
    model_names = list(all_stats.keys())
    
    print("\n=== Model Comparison ===")
    
    # Accuracy comparison
    print("\nAccuracy Comparison:")
    
    if any('strict_accuracy' in all_stats[m] for m in model_names):
        headers = ["Model", "Standard", "Strict", "Failure Rate"]
        print(f"{headers[0]:<30} {headers[1]:<10} {headers[2]:<10} {headers[3]:<15}")
        print("-" * 70)
        
        for model in model_names:
            stats = all_stats[model]
            standard = stats.get('accuracy', 0) * 100
            strict = stats.get('strict_accuracy', 0) * 100 if 'strict_accuracy' in stats else 'N/A'
            failure = stats.get('failure_rate', 0) * 100 if 'failure_rate' in stats else 'N/A'
            
            strict_str = f"{strict:.1f}%" if isinstance(strict, float) else strict
            failure_str = f"{failure:.1f}%" if isinstance(failure, float) else failure
            
            print(f"{model:<30} {standard:.1f}%{' ':5} {strict_str:<10} {failure_str:<15}")
    else:
        headers = ["Model", "Accuracy"]
        print(f"{headers[0]:<30} {headers[1]:<10}")
        print("-" * 45)
        
        for model in model_names:
            stats = all_stats[model]
            accuracy = stats.get('accuracy', 0) * 100
            print(f"{model:<30} {accuracy:.1f}%")
    
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