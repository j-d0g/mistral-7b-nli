#!/usr/bin/env python3
"""
Script to analyze the relationship between token length and accuracy in the dataset.
Generates a visualization showing how accuracy varies with different token length ranges.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set style for plots
plt.style.use('ggplot')
sns.set_style("whitegrid")

# Custom color palettes with good contrast
MISTRAL_COLORS = ["#7e57c2", "#3949ab", "#e91e63", "#009688"]
HIGHLIGHT_COLOR = "#ff6e40"
BACKGROUND_COLOR = '#f8f9fa'

def estimate_token_count(text):
    """Simple function to estimate token count."""
    if not text or not isinstance(text, str):
        return 0
    # Simple approximation: ~4 characters per token on average for English text
    return len(text) // 4

def extract_json_from_text(text):
    """Extract JSON object from within a string that contains JSON."""
    if not text or not isinstance(text, str):
        return {}
    
    try:
        # Try to find JSON content after [/INST]
        if '[/INST]' in text:
            content_after_inst = text.split('[/INST]')[1].strip()
            # Find the JSON object within the text (anything between { and })
            match = re.search(r'\{.*\}', content_after_inst)
            if match:
                json_str = match.group(0)
                # Parse the JSON content
                return json.loads(json_str)
        return {}
    except Exception as e:
        # If any error occurs, return empty dict
        return {}

def process_jsonl_file(file_path):
    """Process a JSONL file and extract data with token counts."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return data
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                
                # Handle nested JSON inside 'text' field
                if 'text' in record and isinstance(record['text'], str):
                    json_content = extract_json_from_text(record['text'])
                    # Merge the extracted JSON into the main record
                    if json_content:
                        for key, value in json_content.items():
                            record[key] = value
                
                # Extract thought process and calculate token count
                if 'thought_process' in record:
                    thought_process = str(record['thought_process'])
                    record['token_count'] = estimate_token_count(thought_process)
                
                data.append(record)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {file_path}")
                continue
    return data

def get_token_bucket(token_count):
    """Assign a token count to a bucket."""
    if token_count <= 100:
        return "0-100"
    elif token_count <= 200:
        return "101-200"
    elif token_count <= 300:
        return "201-300"
    else:
        return "301+"

def calculate_accuracy_by_token_length(data):
    """Calculate accuracy grouped by token length buckets."""
    # Group data by token length buckets
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for item in data:
        # Skip items without token count or labels
        if 'token_count' not in item:
            continue
            
        # Get the token bucket
        bucket = get_token_bucket(item['token_count'])
        
        # Check if prediction matches ground truth
        predicted_label = None
        true_label = None
        
        if 'predicted_label' in item:
            predicted_label = item['predicted_label']
        
        if 'true_label' in item:
            true_label = item['true_label']
        elif 'label' in item:
            true_label = item['label']
            
        # Skip items without both predicted and true labels
        if predicted_label is None or true_label is None:
            continue
            
        # Convert labels to same format (string or int)
        if isinstance(predicted_label, str) and predicted_label.isdigit():
            predicted_label = int(predicted_label)
        if isinstance(true_label, str) and true_label.isdigit():
            true_label = int(true_label)
            
        # Check if prediction is correct
        is_correct = predicted_label == true_label
        
        # Update bucket statistics
        buckets[bucket]["total"] += 1
        if is_correct:
            buckets[bucket]["correct"] += 1
    
    # Calculate accuracy for each bucket
    accuracy_by_bucket = {}
    for bucket, stats in buckets.items():
        if stats["total"] > 0:
            accuracy_by_bucket[bucket] = {
                "accuracy": (stats["correct"] / stats["total"]) * 100,
                "count": stats["total"]
            }
    
    return accuracy_by_bucket

def create_token_vs_accuracy_visualization(accuracy_data, output_path):
    """Create visualization showing relationship between token length and accuracy."""
    # Extract data from accuracy_by_bucket
    buckets = []
    accuracies = []
    counts = []
    
    # Sort buckets in ascending order
    bucket_order = {"0-100": 0, "101-200": 1, "201-300": 2, "301+": 3}
    sorted_buckets = sorted(accuracy_data.items(), key=lambda x: bucket_order.get(x[0], 999))
    
    for bucket, data in sorted_buckets:
        buckets.append(bucket)
        accuracies.append(data["accuracy"])
        counts.append(data["count"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Normalize counts for plotting
    max_count = max(counts)
    normalized_counts = [count/max_count * 50 for count in counts]
    
    # Plot scatter points with size representing count
    for i, (bucket, accuracy, count, norm_count) in enumerate(zip(buckets, accuracies, counts, normalized_counts)):
        ax.scatter(i, accuracy, s=norm_count*20, alpha=0.7, color=MISTRAL_COLORS[i % len(MISTRAL_COLORS)], 
                  edgecolor='black', linewidth=1)
        
        # Add count labels
        ax.text(i, accuracy+3, f'{count} examples', ha='center', va='center', fontsize=10)
    
    # Connect points with line
    ax.plot(range(len(buckets)), accuracies, color='gray', linestyle='--', alpha=0.5)
    
    # Add annotations
    for i, (bucket, accuracy) in enumerate(zip(buckets, accuracies)):
        ax.text(i, accuracy-4, f'{accuracy:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Customize chart
    ax.set_title('Reasoning Chain Length vs. Accuracy', fontsize=16, pad=15)
    ax.set_xlabel('Token Range in Reasoning Chain', fontsize=12, labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets)
    
    # Set y-axis limits with some padding
    min_acc = min(accuracies) - 5 if accuracies else 70
    max_acc = max(accuracies) + 5 if accuracies else 100
    ax.set_ylim(max(0, min(min_acc, 70)), min(100, max(max_acc, 100)))
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Final adjustments
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze token length vs accuracy relationship.')
    parser.add_argument('--train-file', type=str, default='data/finetune/train_ft_final.jsonl', help='Path to the training data JSONL')
    parser.add_argument('--dev-file', type=str, default='data/finetune/dev_ft.jsonl', help='Path to the dev data JSONL')
    parser.add_argument('--test-file', type=str, default='data/finetune/test_ft.jsonl', help='Path to the test data JSONL')
    parser.add_argument('--output-dir', type=str, default='metrics', help='Directory to save metrics output')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print(f"Loading training data from {args.train_file}...")
    train_data = process_jsonl_file(args.train_file)
    
    print(f"Loading dev data from {args.dev_file}...")
    dev_data = process_jsonl_file(args.dev_file)
    
    print(f"Loading test data from {args.test_file}...")
    test_data = process_jsonl_file(args.test_file)
    
    # Combine datasets to get more robust statistics
    all_data = train_data + dev_data + test_data
    
    # Calculate accuracy by token length bucket
    print("Calculating accuracy by token length...")
    accuracy_by_bucket = calculate_accuracy_by_token_length(all_data)
    
    # Print results
    print("\n===== Token Length vs. Accuracy =====")
    for bucket in sorted(accuracy_by_bucket.keys(), key=lambda x: (0 if x == "0-100" else 
                                                                 1 if x == "101-200" else 
                                                                 2 if x == "201-300" else 3)):
        data = accuracy_by_bucket[bucket]
        print(f"Bucket {bucket}: {data['accuracy']:.2f}% accuracy ({data['count']} examples)")
    
    # Create visualization
    output_path = os.path.join(args.output_dir, "token_vs_accuracy.png")
    create_token_vs_accuracy_visualization(accuracy_by_bucket, output_path)
    print(f"\nVisualization saved to {output_path}")

if __name__ == "__main__":
    main() 