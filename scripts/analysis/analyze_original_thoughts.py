#!/usr/bin/env python3
"""
Script to analyze the relationship between reasoning chain token length and accuracy
in the original thoughts generated during data preparation.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# Set style for plots
plt.style.use('ggplot')

# Custom color palettes with good contrast
COLORS = ["#7e57c2", "#3949ab", "#e91e63", "#009688"]
HIGHLIGHT_COLOR = "#ff6e40"
BACKGROUND_COLOR = '#f8f9fa'

def estimate_token_count(text):
    """Simple function to estimate token count."""
    if not text or not isinstance(text, str):
        return 0
    # Simple approximation: ~4 characters per token on average for English text
    return len(text) // 4

def load_jsonl_file(file_path):
    """Load data from a JSONL file (each line is a JSON object)."""
    data = []Learning From Mistakes
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file_path}: {e}")
                    continue
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

def analyze_token_vs_accuracy(data):
    """Analyze token length vs accuracy relationship."""
    # Define token ranges
    ranges = [
        (0, 100),     # 0-100 tokens
        (101, 200),   # 101-200 tokens
        (201, 300),   # 201-300 tokens
        (301, float('inf'))  # 301+ tokens
    ]
    
    range_labels = ['0-100', '101-200', '201-300', '301+']
    
    # Initialize counters for each range
    range_totals = [0] * len(ranges)
    range_correct = [0] * len(ranges)
    
    # Process each sample
    for item in data:
        # Extract thought process - might be under different field names in original thoughts
        thought_process = ""
        if "thought_process" in item:
            thought_process = item["thought_process"]
        elif "reasoning" in item:
            thought_process = item["reasoning"]
        
        # Calculate token count
        token_count = estimate_token_count(thought_process)
        
        # Skip if no thought process
        if token_count == 0:
            continue
            
        # Determine which range this falls into
        for i, (min_tokens, max_tokens) in enumerate(ranges):
            if min_tokens <= token_count <= max_tokens:
                range_totals[i] += 1
                
                # Check if prediction was correct
                correct = False
                if "predicted_label" in item and "true_label" in item:
                    correct = item["predicted_label"] == item["true_label"]
                elif "correct" in item:
                    correct = item["correct"]
                
                if correct:
                    range_correct[i] += 1
                break
    
    # Calculate accuracy for each range
    accuracies = []
    for correct, total in zip(range_correct, range_totals):
        if total > 0:
            accuracies.append((correct / total) * 100)
        else:
            accuracies.append(0)
    
    return range_labels, accuracies, range_totals

def create_token_vs_accuracy_visualization(range_labels, accuracies, counts, output_path):
    """Create visualization showing relationship between token length and accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Handle case where all counts are zero
    if sum(counts) == 0:
        ax.text(0.5, 0.5, "No data available for analysis", 
               ha='center', va='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
        plt.close()
        print(f"Visualization saved to {output_path}")
        return output_path
    
    # Normalize counts for plotting
    max_count = max(counts)
    normalized_counts = [count/max_count * 50 for count in counts]
    
    # Plot scatter points with size representing count
    for i, (token_range, accuracy, count, norm_count) in enumerate(zip(range_labels, accuracies, counts, normalized_counts)):
        ax.scatter(i, accuracy, s=norm_count*20, alpha=0.7, color=COLORS[i % len(COLORS)], 
                  edgecolor='black', linewidth=1)
        
        # Add count labels
        ax.text(i, accuracy+3, f'{count} examples', ha='center', va='center', fontsize=10)
    
    # Connect points with line
    ax.plot(range(len(range_labels)), accuracies, color='gray', linestyle='--', alpha=0.5)
    
    # Add annotations
    for i, (token_range, accuracy) in enumerate(zip(range_labels, accuracies)):
        ax.text(i, accuracy-4, f'{accuracy:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Customize chart
    ax.set_title('Reasoning Chain Length vs. Accuracy (Original Thoughts)', fontsize=16, pad=15)
    ax.set_xlabel('Token Range in Reasoning Chain', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels)
    ax.set_ylim(50, 100)  # Set y-axis limit to focus on the relevant range
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Analyze token length vs accuracy in original thoughts.')
    parser.add_argument('--train-file', type=str, default='data/original_thoughts/train_thoughts.json', 
                        help='Path to the training thoughts JSONL file')
    parser.add_argument('--dev-file', type=str, default='data/original_thoughts/dev_thoughts.json', 
                        help='Path to the dev thoughts JSONL file')
    parser.add_argument('--output-dir', type=str, default='metrics', 
                        help='Directory to save output files')
    parser.add_argument('--output-file', type=str, default='original_token_vs_accuracy.png', 
                        help='Name of the output visualization file')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training thoughts
    print(f"Loading training thoughts from {args.train_file}...")
    train_data = load_jsonl_file(args.train_file)
    print(f"Loaded {len(train_data)} training examples")
    
    # Load dev thoughts
    print(f"Loading dev thoughts from {args.dev_file}...")
    dev_data = load_jsonl_file(args.dev_file)
    print(f"Loaded {len(dev_data)} dev examples")
    
    # Merge data
    merged_data = train_data + dev_data
    print(f"Merged data contains {len(merged_data)} examples")
    
    # Analyze token length vs accuracy
    print("Analyzing token length vs accuracy relationship...")
    range_labels, accuracies, counts = analyze_token_vs_accuracy(merged_data)
    
    # Create output path
    output_path = os.path.join(args.output_dir, args.output_file)
    
    # Print summary
    print("\n===== Original Thoughts: Token Length vs Accuracy =====")
    total_correct = sum([counts[i] * (accuracies[i]/100) for i in range(len(counts))])
    total_examples = sum(counts)
    overall_accuracy = (total_correct / total_examples * 100) if total_examples > 0 else 0
    print(f"Overall accuracy: {overall_accuracy:.2f}% ({int(total_correct)}/{total_examples})")
    
    for i, (label, accuracy, count) in enumerate(zip(range_labels, accuracies, counts)):
        print(f"{label} tokens: {accuracy:.2f}% accuracy ({count} examples)")
    
    # Create visualization
    create_token_vs_accuracy_visualization(range_labels, accuracies, counts, output_path)
    
    # Save data to json for reference
    data_output_path = os.path.join(args.output_dir, "original_token_vs_accuracy.json")
    with open(data_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "ranges": range_labels,
            "accuracies": accuracies,
            "counts": counts,
            "overall_accuracy": overall_accuracy
        }, f, indent=2)
    print(f"Data saved to {data_output_path}")

if __name__ == "__main__":
    main() 