#!/usr/bin/env python3
"""
Script to analyze the relationship between reasoning chain token length and accuracy.
This script processes evaluation results and generates token_vs_accuracy visualization.
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

def extract_thought_process(output):
    """Extract thought_process from model output."""
    if not output or not isinstance(output, str):
        return ""
        
    try:
        # Try to find JSON content in the output
        match = re.search(r'\{.*"thought_process":\s*"(.*?)",.*\}', output, re.DOTALL)
        if match:
            return match.group(1)
        return ""
    except Exception as e:
        print(f"Error extracting thought_process: {e}")
        return ""

def load_results(file_path):
    """Load results from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading results file: {e}")
        return None

def analyze_token_vs_accuracy(results):
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
    for sample in results:
        # Extract thought process
        thought_process = ""
        if "thought_process" in sample:
            thought_process = sample["thought_process"]
        elif "output" in sample:
            thought_process = extract_thought_process(sample["output"])
        
        # Calculate token count
        token_count = estimate_token_count(thought_process)
        
        # Determine which range this falls into
        for i, (min_tokens, max_tokens) in enumerate(ranges):
            if min_tokens <= token_count <= max_tokens:
                range_totals[i] += 1
                if sample.get("correct", False):
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

def create_token_vs_accuracy_visualization(range_labels, accuracies, counts, output_dir="metrics"):
    """Create visualization showing relationship between token length and accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Normalize counts for plotting
    max_count = max(counts) if counts else 1
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
    ax.set_title('Reasoning Chain Length vs. Accuracy', fontsize=16, pad=15)
    ax.set_xlabel('Token Range in Reasoning Chain', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(range(len(range_labels)))
    ax.set_xticklabels(range_labels)
    ax.set_ylim(50, 100)  # Set y-axis limit to focus on the relevant range
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Final adjustments
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save figure
    output_path = os.path.join(output_dir, "token_vs_accuracy.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path

def update_metrics_json(range_labels, accuracies, counts, output_dir="metrics"):
    """Update the card_metrics.json file with token vs accuracy data."""
    metrics_path = os.path.join(output_dir, "card_metrics.json")
    
    try:
        # Load existing metrics
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        # Add token vs accuracy data
        if "token_vs_accuracy" not in metrics:
            metrics["token_vs_accuracy"] = {}
        
        metrics["token_vs_accuracy"] = {
            "ranges": range_labels,
            "accuracies": accuracies,
            "counts": counts
        }
        
        # Save updated metrics
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Updated metrics in {metrics_path}")
    except Exception as e:
        print(f"Error updating metrics file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Analyze token length vs accuracy relationship.')
    parser.add_argument('--results-file', type=str, default='results/nlistral-ablation1-test-labelled.json', 
                        help='Path to the results JSON file')
    parser.add_argument('--output-dir', type=str, default='metrics', 
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    data = load_results(args.results_file)
    
    if not data or "results" not in data:
        print("Error: No results found in the file.")
        return
    
    # Get overall accuracy from the results file
    overall_accuracy = data.get("accuracy", 0) * 100
    print(f"Overall accuracy in results: {overall_accuracy:.2f}%")
    
    # Analyze token length vs accuracy
    print("Analyzing token length vs accuracy relationship...")
    range_labels, accuracies, counts = analyze_token_vs_accuracy(data["results"])
    
    # Print summary
    print("\n===== Token Length vs Accuracy =====")
    for i, (label, accuracy, count) in enumerate(zip(range_labels, accuracies, counts)):
        print(f"{label} tokens: {accuracy:.2f}% accuracy ({count} examples)")
    
    # Create visualization
    create_token_vs_accuracy_visualization(range_labels, accuracies, counts, args.output_dir)
    
    # Update metrics JSON
    update_metrics_json(range_labels, accuracies, counts, args.output_dir)

if __name__ == "__main__":
    main() 