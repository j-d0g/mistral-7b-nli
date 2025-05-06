#!/usr/bin/env python3
"""
Script to analyze the NLI dataset and model outputs to generate metrics
for Hugging Face model and dataset cards.

This script:
1. Analyzes dataset statistics (token lengths, distributions)
2. Calculates model performance metrics
3. Prepares visualization data
4. Outputs a JSON file with all metrics

Usage:
    python generate_card_metrics.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import sys

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_analysis import (
    count_tokens, calculate_statistics, calculate_token_bucket_stats, 
    get_token_bucket, generate_summary
)

# Directories
DATA_DIR = "data"
FINETUNE_DIR = os.path.join(DATA_DIR, "finetune")
ORIGINAL_DIR = os.path.join(DATA_DIR, "original_data")
THOUGHTS_DIR = os.path.join(DATA_DIR, "original_thoughts")
REFLECTED_DIR = os.path.join(DATA_DIR, "reflected_thoughts")
SCORED_DIR = os.path.join(DATA_DIR, "scored_thoughts")
OUTPUT_DIR = "metrics"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def load_json(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def analyze_dataset_statistics():
    """Analyze dataset statistics and return metrics."""
    print("Analyzing dataset statistics...")
    
    # Load datasets
    train_data = load_jsonl(os.path.join(FINETUNE_DIR, "train_ft.jsonl"))
    dev_data = load_jsonl(os.path.join(FINETUNE_DIR, "dev_ft.jsonl"))
    test_data = load_jsonl(os.path.join(FINETUNE_DIR, "test_ft.jsonl")) if os.path.exists(os.path.join(FINETUNE_DIR, "test_ft.jsonl")) else []
    
    # If test set doesn't exist, check for other files that might contain test data
    if not test_data:
        for filename in os.listdir(FINETUNE_DIR):
            if "test" in filename and filename.endswith(".jsonl"):
                test_data = load_jsonl(os.path.join(FINETUNE_DIR, filename))
                break
    
    metrics = {
        "dataset": {
            "total_examples": len(train_data) + len(dev_data) + len(test_data),
            "train_examples": len(train_data),
            "dev_examples": len(dev_data),
            "test_examples": len(test_data),
            "train_percentage": round(len(train_data) / (len(train_data) + len(dev_data) + len(test_data)) * 100, 1) if (len(train_data) + len(dev_data) + len(test_data)) > 0 else 0,
            "dev_percentage": round(len(dev_data) / (len(train_data) + len(dev_data) + len(test_data)) * 100, 1) if (len(train_data) + len(dev_data) + len(test_data)) > 0 else 0,
            "test_percentage": round(len(test_data) / (len(train_data) + len(dev_data) + len(test_data)) * 100, 1) if (len(train_data) + len(dev_data) + len(test_data)) > 0 else 0,
        }
    }
    
    # Label distribution
    train_labels = [ex.get("predicted_label", None) for ex in train_data if "predicted_label" in ex]
    label_counts = Counter(train_labels)
    
    metrics["dataset"]["label_distribution"] = {
        "entailment": label_counts.get(1, 0),
        "no_entailment": label_counts.get(0, 0),
        "entailment_percentage": round(label_counts.get(1, 0) / len(train_labels) * 100, 1) if train_labels else 0,
        "no_entailment_percentage": round(label_counts.get(0, 0) / len(train_labels) * 100, 1) if train_labels else 0,
    }
    
    # Token length analysis
    premise_tokens = []
    hypothesis_tokens = []
    thought_tokens = []
    reflection_tokens = []
    
    # Analyze token lengths
    for dataset in [train_data, dev_data, test_data]:
        for example in dataset:
            if "premise" in example:
                premise_tokens.append(count_tokens(example["premise"]))
            if "hypothesis" in example:
                hypothesis_tokens.append(count_tokens(example["hypothesis"]))
            if "thought_process" in example:
                thought_tokens.append(count_tokens(example["thought_process"]))
            if "reflection" in example:
                reflection_tokens.append(count_tokens(example["reflection"]))
    
    # Calculate token statistics
    metrics["tokens"] = {
        "premise": {
            "avg": round(np.mean(premise_tokens), 1) if premise_tokens else 0,
            "min": min(premise_tokens) if premise_tokens else 0,
            "max": max(premise_tokens) if premise_tokens else 0,
            "quartiles": [int(q) for q in np.percentile(premise_tokens, [25, 50, 75])] if premise_tokens else [0, 0, 0],
        },
        "hypothesis": {
            "avg": round(np.mean(hypothesis_tokens), 1) if hypothesis_tokens else 0,
            "min": min(hypothesis_tokens) if hypothesis_tokens else 0,
            "max": max(hypothesis_tokens) if hypothesis_tokens else 0,
            "quartiles": [int(q) for q in np.percentile(hypothesis_tokens, [25, 50, 75])] if hypothesis_tokens else [0, 0, 0],
        },
        "thought_process": {
            "avg": round(np.mean(thought_tokens), 1) if thought_tokens else 0,
            "min": min(thought_tokens) if thought_tokens else 0,
            "max": max(thought_tokens) if thought_tokens else 0,
            "quartiles": [int(q) for q in np.percentile(thought_tokens, [25, 50, 75])] if thought_tokens else [0, 0, 0],
        },
        "reflection": {
            "avg": round(np.mean(reflection_tokens), 1) if reflection_tokens else 0,
            "min": min(reflection_tokens) if reflection_tokens else 0,
            "max": max(reflection_tokens) if reflection_tokens else 0,
            "quartiles": [int(q) for q in np.percentile(reflection_tokens, [25, 50, 75])] if reflection_tokens else [0, 0, 0],
        }
    }
    
    return metrics

def analyze_model_performance():
    """Analyze model performance from scored thoughts."""
    print("Analyzing model performance...")
    
    metrics = {"models": {}}
    
    # Check for scored thoughts in the scored_thoughts directory
    if not os.path.exists(SCORED_DIR):
        print(f"Warning: {SCORED_DIR} directory not found.")
        return metrics
    
    # Find all model result files
    for filename in os.listdir(SCORED_DIR):
        if filename.endswith("_results.json"):
            model_name = filename.replace("_results.json", "")
            results_path = os.path.join(SCORED_DIR, filename)
            
            # Load results file
            results = load_json(results_path)
            
            if not results:
                continue
                
            # Try to extract performance metrics
            model_metrics = {}
            
            # Look for summary/stats in different possible formats
            if "stats" in results:
                stats = results["stats"]
                model_metrics = {
                    "accuracy": round(stats.get("accuracy", 0), 1),
                    "precision": round(stats.get("precision", 0), 1),
                    "recall": round(stats.get("recall", 0), 1),
                    "f1_score": round(stats.get("f1_score", 0), 1),
                }
            elif "summary" in results and "stats" in results["summary"]:
                stats = results["summary"]["stats"]
                model_metrics = {
                    "accuracy": round(stats.get("accuracy", 0), 1),
                    "precision": round(stats.get("precision", 0), 1),
                    "recall": round(stats.get("recall", 0), 1),
                    "f1_score": round(stats.get("f1_score", 0), 1),
                }
            elif "output_count" in results and "correct_count" in results:
                # Calculate metrics from raw counts
                output_count = results.get("output_count", 0)
                correct_count = results.get("correct_count", 0)
                tp = results.get("true_positives", 0)
                fp = results.get("false_positives", 0)
                fn = results.get("false_negatives", 0)
                
                model_metrics = calculate_statistics({
                    "output_count": output_count,
                    "correct_count": correct_count,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn
                })
                
                model_metrics = {
                    "accuracy": round(model_metrics.get("accuracy", 0), 1),
                    "precision": round(model_metrics.get("precision", 0), 1),
                    "recall": round(model_metrics.get("recall", 0), 1),
                    "f1_score": round(model_metrics.get("f1_score", 0), 1),
                }
            
            # Add token lengths if available
            if "token_counts" in results:
                token_counts = results["token_counts"]
                if token_counts:
                    model_metrics["avg_tokens"] = round(np.mean(token_counts), 1)
                    model_metrics["token_quartiles"] = [int(q) for q in np.percentile(token_counts, [25, 50, 75])]
            
            # If we found metrics, add to our results
            if model_metrics:
                metrics["models"][model_name] = model_metrics
    
    return metrics

def create_visualizations(metrics):
    """Create visualizations for metrics and save to files."""
    print("Creating visualizations (placeholders)...")
    
    # Create placeholder image files
    visualizations = {
        "dataset_distribution": os.path.join(OUTPUT_DIR, "dataset_distribution.png"),
        "token_lengths": os.path.join(OUTPUT_DIR, "token_lengths.png"),
        "model_performance": os.path.join(OUTPUT_DIR, "model_performance.png"),
        "token_vs_accuracy": os.path.join(OUTPUT_DIR, "token_vs_accuracy.png"),
        "reasoning_quality": os.path.join(OUTPUT_DIR, "reasoning_quality.png"),
    }
    
    # Create a simple placeholder fig to save
    for name, path in visualizations.items():
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"Placeholder for {name}", 
                 ha='center', va='center', fontsize=20, transform=plt.gca().transAxes)
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    
    metrics["visualizations"] = {name: os.path.basename(path) for name, path in visualizations.items()}
    
    return metrics

def main():
    """Main function to generate all metrics."""
    print("Generating metrics for Hugging Face cards...")
    
    # Analyze dataset statistics
    dataset_metrics = analyze_dataset_statistics()
    
    # Analyze model performance
    model_metrics = analyze_model_performance()
    
    # Combine all metrics
    metrics = {**dataset_metrics, **model_metrics}
    
    # Create visualizations
    metrics = create_visualizations(metrics)
    
    # Write all metrics to a JSON file
    output_path = os.path.join(OUTPUT_DIR, "card_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main() 