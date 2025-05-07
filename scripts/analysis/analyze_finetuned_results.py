#!/usr/bin/env python3
"""
Script to analyze the relationship between token length and accuracy in the fine-tuned model's test results.
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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

def extract_thought_process(output_text):
    """Extract the thought process from the model output."""
    if not output_text:
        return ""
        
    # Try to find JSON in the output
    json_match = re.search(r'\{.*"thought_process":\s*"(.*?)".*\}', output_text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # Look for other patterns that might contain the reasoning
    reasoning_match = re.search(r'step 1:(.*?)(?:predicted_label|$)', output_text, re.DOTALL)
    if reasoning_match:
        return reasoning_match.group(1)
    
    # If all else fails, return the full output
    return output_text

def analyze_token_vs_accuracy(results):
    """Analyze token length vs accuracy relationship in the fine-tuned model results."""
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
    
    # Process each result
    for result in results:
        # Extract thought process
        thought_process = extract_thought_process(result.get("output", ""))
        
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
                if result.get("correct", False):
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
    ax.set_title('Reasoning Chain Length vs. Accuracy (Fine-tuned Model)', fontsize=16, pad=15)
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
    # Load test results
    results_file = 'results/nlistral-ablation1-test-labelled.json'
    output_dir = 'metrics'
    output_file = 'token_vs_accuracy.png'
    
    print(f"Loading fine-tuned model results from {results_file}...")
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Make sure 'results' key exists
    if 'results' not in data:
        print(f"Error: 'results' key not found in {results_file}")
        return
    
    results = data['results']
    print(f"Loaded {len(results)} test examples")
    
    # Analyze token length vs accuracy
    print("Analyzing token length vs accuracy relationship...")
    range_labels, accuracies, counts = analyze_token_vs_accuracy(results)
    
    # Create output path
    output_path = os.path.join(output_dir, output_file)
    
    # Print summary
    print("\n===== Fine-tuned Model: Token Length vs Accuracy =====")
    correct_sum = sum(counts[i] * (accuracies[i]/100) for i in range(len(counts)))
    total_examples = sum(counts)
    overall_accuracy = (correct_sum / total_examples * 100) if total_examples > 0 else 0
    print(f"Overall accuracy: {overall_accuracy:.2f}% ({int(correct_sum)}/{total_examples})")
    
    for i, (label, accuracy, count) in enumerate(zip(range_labels, accuracies, counts)):
        print(f"{label} tokens: {accuracy:.2f}% accuracy ({count} examples)")
    
    # Create visualization
    create_token_vs_accuracy_visualization(range_labels, accuracies, counts, output_path)
    
    # Save data to json for reference
    data_output_path = os.path.join(output_dir, "finetuned_token_vs_accuracy.json")
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