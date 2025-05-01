#!/usr/bin/env python3
import json
import sys
import argparse
from collections import Counter

def load_results(file_path):
    """Load results from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return None

def compare_files(base_file, finetuned_file):
    """Compare base and fine-tuned model results."""
    base_data = load_results(base_file)
    finetuned_data = load_results(finetuned_file)
    
    if not base_data or not finetuned_data:
        return
    
    # Get overall accuracy
    base_acc = base_data.get('accuracy', 'N/A')
    finetuned_acc = finetuned_data.get('accuracy', 'N/A')
    
    print(f"\n=== Overall Accuracy Comparison ===")
    print(f"Base model:     {base_acc:.4f}")
    print(f"Fine-tuned:     {finetuned_acc:.4f}")
    print(f"Improvement:    {(finetuned_acc - base_acc):.4f}\n")
    
    # Compare predictions on same examples
    base_results_list = base_data.get('results', [])
    ft_results_list = finetuned_data.get('results', [])
    
    # Create dictionaries with index as key
    base_results = {i: r for i, r in enumerate(base_results_list)}
    ft_results = {i: r for i, r in enumerate(ft_results_list)}
    
    # Find common indices - should be all indices if sample sizes match
    common_indices = set(base_results.keys()) & set(ft_results.keys())
    print(f"Found {len(common_indices)} examples in both result sets\n")
    
    # Analyze differences
    differences = Counter()
    examples = {
        'base_right_ft_wrong': [],
        'base_wrong_ft_right': [],
        'both_right': [],
        'both_wrong': []
    }
    
    for idx in common_indices:
        base_result = base_results[idx]
        ft_result = ft_results[idx]
        
        base_correct = base_result.get('correct', False)
        ft_correct = ft_result.get('correct', False)
        
        if base_correct and not ft_correct:
            differences['base_right_ft_wrong'] += 1
            examples['base_right_ft_wrong'].append(idx)
        elif not base_correct and ft_correct:
            differences['base_wrong_ft_right'] += 1
            examples['base_wrong_ft_right'].append(idx)
        elif base_correct and ft_correct:
            differences['both_right'] += 1
            examples['both_right'].append(idx)
        else:
            differences['both_wrong'] += 1
            examples['both_wrong'].append(idx)
    
    print(f"=== Prediction Differences ===")
    print(f"Base right, fine-tuned wrong: {differences['base_right_ft_wrong']}")
    print(f"Base wrong, fine-tuned right: {differences['base_wrong_ft_right']}")
    print(f"Both right: {differences['both_right']}")
    print(f"Both wrong: {differences['both_wrong']}")
    
    # Compute McNemar's test statistics
    print(f"\n=== McNemar's Test (A/B Testing) ===")
    print(f"  | FT Right | FT Wrong |")
    print(f"--+----------+----------+")
    print(f"B Right | {differences['both_right']:8} | {differences['base_right_ft_wrong']:8} |")
    print(f"B Wrong | {differences['base_wrong_ft_right']:8} | {differences['both_wrong']:8} |")
    
    # Print some example differences
    print(f"\n=== Example Output Differences ===")
    
    # Show base wrong, fine-tuned right examples
    if examples['base_wrong_ft_right']:
        print("\nBase wrong, fine-tuned right examples:")
        for i, idx in enumerate(examples['base_wrong_ft_right'][:3]):  # Show up to 3 examples
            base = base_results[idx]
            ft = ft_results[idx]
            print(f"\nExample {i+1} (Index: {idx}):")
            print(f"Premise: {base.get('premise', 'N/A')}")
            print(f"Hypothesis: {base.get('hypothesis', 'N/A')}")
            print(f"True label: {base.get('true_label', 'N/A')}")
            print(f"\nBase model output:")
            print(f"{base.get('output', 'N/A')[:300]}...")
            print(f"\nFine-tuned model output:")
            print(f"{ft.get('output', 'N/A')[:300]}...")
    
    # Show base right, fine-tuned wrong examples
    if examples['base_right_ft_wrong']:
        print("\nBase right, fine-tuned wrong examples:")
        for i, idx in enumerate(examples['base_right_ft_wrong'][:3]):  # Show up to 3 examples
            base = base_results[idx]
            ft = ft_results[idx]
            print(f"\nExample {i+1} (Index: {idx}):")
            print(f"Premise: {base.get('premise', 'N/A')}")
            print(f"Hypothesis: {base.get('hypothesis', 'N/A')}")
            print(f"True label: {base.get('true_label', 'N/A')}")
            print(f"\nBase model output:")
            print(f"{base.get('output', 'N/A')[:300]}...")
            print(f"\nFine-tuned model output:")
            print(f"{ft.get('output', 'N/A')[:300]}...")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fixed base and fine-tuned model results")
    parser.add_argument("base_file", help="Path to fixed base model results JSON")
    parser.add_argument("finetuned_file", help="Path to fixed fine-tuned model results JSON")
    
    args = parser.parse_args()
    compare_files(args.base_file, args.finetuned_file) 