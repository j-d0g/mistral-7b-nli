#!/usr/bin/env python3
"""
Generate visualizations and analysis from NLI experiment results.

This script produces various plots, tables, and analyses from the results of
thought generation experiments. It's designed to be easily portable to a
Jupyter notebook.

Usage:
    python scripts/analyze_results.py --results-json path/to/results.json [--output-dir path/to/output]

Author: Jordan
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Dict, List, Any, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project utilities
from utils.data_analysis import (
    count_tokens,
    calculate_statistics,
    calculate_token_bucket_stats,
    get_token_bucket
)

# Set prettier defaults for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
sns.set_context("talk")
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


def load_results(json_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load results from a JSON file.
    
    Args:
        json_path: Path to the JSON results file
        
    Returns:
        Tuple containing:
        - Dictionary with experiment metadata
        - List of result dictionaries (one per example)
    """
    # Read the JSON file line by line (JSONL format)
    results = []
    with open(json_path, 'r') as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError:
                continue
    
    # Extract metadata (from summary file if available)
    metadata = {}
    summary_path = f"{json_path}_summary.txt"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_text = f.read()
            # Parse basic metrics from summary
            for line in summary_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return metadata, results


def prepare_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert results list to a pandas DataFrame for analysis.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame with processed results
    """
    # Extract relevant fields
    processed_results = []
    
    for result in results:
        # Skip malformed entries
        if not isinstance(result, dict):
            continue
            
        # Calculate token count if not present
        if 'thought_process' in result and 'token_count' not in result:
            result['token_count'] = count_tokens(result['thought_process'])
            
        # Determine correctness
        is_correct = False
        if 'predicted_label' in result and 'true_label' in result:
            is_correct = result['predicted_label'] == result['true_label']
            
        # Extract key fields
        entry = {
            'id': result.get('id', None),
            'premise': result.get('premise', ''),
            'hypothesis': result.get('hypothesis', ''),
            'true_label': result.get('true_label', None),
            'predicted_label': result.get('predicted_label', None),
            'is_correct': is_correct,
            'token_count': result.get('token_count', 0),
            'thought_process': result.get('thought_process', '')
        }
        processed_results.append(entry)
    
    return pd.DataFrame(processed_results)


def plot_accuracy_by_token_count(df: pd.DataFrame, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Plot accuracy by token count, using quartile bins.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot (if None, just returns figure)
        
    Returns:
        Matplotlib figure
    """
    # Calculate quartiles
    token_counts = df['token_count'].tolist()
    quartiles = np.percentile(token_counts, [25, 50, 75])
    quartiles = [int(q) for q in quartiles]
    
    # Add token bucket column
    df['token_bucket'] = df['token_count'].apply(lambda x: get_token_bucket(x, quartiles))
    
    # Calculate accuracy by bucket
    bucket_stats = df.groupby('token_bucket').agg(
        count=('is_correct', 'count'),
        correct=('is_correct', 'sum')
    ).reset_index()
    
    bucket_stats['accuracy'] = bucket_stats['correct'] / bucket_stats['count'] * 100
    
    # Sort buckets properly
    bucket_stats['sort_key'] = bucket_stats['token_bucket'].apply(
        lambda x: int(x.split('-')[0]) if '-' in x else int(x.split('+')[0])
    )
    bucket_stats = bucket_stats.sort_values('sort_key')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(bucket_stats['token_bucket'], bucket_stats['accuracy'], color=COLORS[0])
    
    # Add count annotations
    for bar, count, correct in zip(bars, bucket_stats['count'], bucket_stats['correct']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{int(correct)}/{int(count)}',
            ha='center', va='bottom', rotation=0
        )
    
    # Formatting
    ax.set_ylim(0, 100)
    ax.set_xlabel('Token Count Range')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Token Count Range')
    
    # Add quartile annotations
    plt.figtext(0.02, 0.02, f"Quartiles: Q1={quartiles[0]}, Q2={quartiles[1]}, Q3={quartiles[2]}",
                ha="left", fontsize=9)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'accuracy_by_token_count.png'), dpi=300, bbox_inches='tight')
    
    return fig


def plot_token_count_distribution(df: pd.DataFrame, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Plot the distribution of token counts.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot (if None, just returns figure)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot token count distribution
    sns.histplot(data=df, x='token_count', hue='is_correct', 
                 bins=20, kde=True, alpha=0.6, ax=ax,
                 palette=['#e74c3c', '#2ecc71'])
    
    # Add mean line
    plt.axvline(x=df['token_count'].mean(), color='blue', linestyle='--', 
                label=f'Mean ({df["token_count"].mean():.1f})')
    
    # Add median line
    plt.axvline(x=df['token_count'].median(), color='green', linestyle='-.',
                label=f'Median ({df["token_count"].median():.1f})')
    
    # Formatting
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Token Counts')
    ax.legend(title='Prediction')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'token_count_distribution.png'), dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(df: pd.DataFrame, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Plot a confusion matrix of predictions.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot (if None, just returns figure)
        
    Returns:
        Matplotlib figure
    """
    # Filter out rows with missing labels
    valid_df = df.dropna(subset=['true_label', 'predicted_label'])
    
    # Create confusion matrix
    cm = confusion_matrix(valid_df['true_label'], valid_df['predicted_label'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Contradiction (0)', 'Entailment (1)'])
    disp.plot(cmap='Blues', values_format='d', ax=ax)
    
    # Formatting
    ax.set_title('Confusion Matrix')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_summary(df: pd.DataFrame, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Plot summary metrics (accuracy, precision, recall, F1).
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot (if None, just returns figure)
        
    Returns:
        Matplotlib figure
    """
    # Calculate metrics
    valid_df = df.dropna(subset=['true_label', 'predicted_label'])
    
    # Convert labels to numeric to ensure proper calculation
    valid_df['true_label'] = valid_df['true_label'].astype(int)
    valid_df['predicted_label'] = valid_df['predicted_label'].astype(int)
    
    # Calculate TP, FP, FN
    true_positives = sum((valid_df['predicted_label'] == 1) & (valid_df['true_label'] == 1))
    false_positives = sum((valid_df['predicted_label'] == 1) & (valid_df['true_label'] == 0))
    false_negatives = sum((valid_df['predicted_label'] == 0) & (valid_df['true_label'] == 1))
    true_negatives = sum((valid_df['predicted_label'] == 0) & (valid_df['true_label'] == 0))
    
    # Calculate metrics
    accuracy = sum(valid_df['predicted_label'] == valid_df['true_label']) / len(valid_df) * 100
    precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics_df['Metric'], metrics_df['Value'], color=COLORS)
    
    # Add value annotations
    for bar, value in zip(bars, metrics_df['Value']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{value:.1f}%',
            ha='center', va='bottom'
        )
    
    # Formatting
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance Metrics')
    
    # Add raw counts as text
    plt.figtext(0.02, 0.02, 
                f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}, TN: {true_negatives}",
                ha="left", fontsize=9)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    
    return fig


def create_examples_table(df: pd.DataFrame, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create a table with interesting examples of correct and incorrect predictions.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save CSV (if None, just returns DataFrame)
        
    Returns:
        DataFrame with examples
    """
    # Ensure we have the needed columns
    if not all(col in df.columns for col in ['premise', 'hypothesis', 'true_label', 'predicted_label', 'is_correct']):
        return pd.DataFrame(columns=['Type', 'Premise', 'Hypothesis', 'True Label', 'Predicted Label'])
    
    # Filter out rows with missing values
    valid_df = df.dropna(subset=['premise', 'hypothesis', 'true_label', 'predicted_label'])
    
    # Get interesting examples
    correct_entailment = valid_df[(valid_df['is_correct'] == True) & (valid_df['true_label'] == 1)].sample(min(3, sum((valid_df['is_correct'] == True) & (valid_df['true_label'] == 1))))
    correct_contradiction = valid_df[(valid_df['is_correct'] == True) & (valid_df['true_label'] == 0)].sample(min(3, sum((valid_df['is_correct'] == True) & (valid_df['true_label'] == 0))))
    incorrect_entailment = valid_df[(valid_df['is_correct'] == False) & (valid_df['true_label'] == 1)].sample(min(3, sum((valid_df['is_correct'] == False) & (valid_df['true_label'] == 1))))
    incorrect_contradiction = valid_df[(valid_df['is_correct'] == False) & (valid_df['true_label'] == 0)].sample(min(3, sum((valid_df['is_correct'] == False) & (valid_df['true_label'] == 0))))
    
    # Combine and format
    examples = pd.concat([correct_entailment, correct_contradiction, incorrect_entailment, incorrect_contradiction])
    
    # Add type column and reorder
    examples['Type'] = examples.apply(
        lambda row: f"{'Correct' if row['is_correct'] else 'Incorrect'} {'Entailment' if row['true_label'] == 1 else 'Contradiction'}",
        axis=1
    )
    
    formatted_examples = examples[['Type', 'premise', 'hypothesis', 'true_label', 'predicted_label']].rename(
        columns={'premise': 'Premise', 'hypothesis': 'Hypothesis', 
                 'true_label': 'True Label', 'predicted_label': 'Predicted Label'}
    )
    
    if output_dir:
        formatted_examples.to_csv(os.path.join(output_dir, 'example_predictions.csv'), index=False)
    
    return formatted_examples


def plot_thought_length_vs_accuracy(df: pd.DataFrame, output_dir: Optional[str] = None) -> plt.Figure:
    """
    Create a scatter plot of thought length vs. prediction accuracy.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plot (if None, just returns figure)
        
    Returns:
        Matplotlib figure
    """
    # Ensure we have the needed columns
    if not all(col in df.columns for col in ['token_count', 'is_correct']):
        return plt.figure()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create binned data for smoothed trend
    bins = np.linspace(df['token_count'].min(), df['token_count'].max(), 20)
    bin_means = []
    bin_centers = []
    
    for i in range(len(bins) - 1):
        bin_start, bin_end = bins[i], bins[i+1]
        bin_df = df[(df['token_count'] >= bin_start) & (df['token_count'] < bin_end)]
        
        if len(bin_df) > 0:
            bin_accuracy = bin_df['is_correct'].mean() * 100
            bin_means.append(bin_accuracy)
            bin_centers.append((bin_start + bin_end) / 2)
    
    # Scatter plot for individual examples
    ax.scatter(df['token_count'], df['is_correct'].astype(int) * 100, 
               alpha=0.2, color=COLORS[0], label='Individual examples')
    
    # Line plot for binned trend
    if bin_means:
        ax.plot(bin_centers, bin_means, color=COLORS[1], linewidth=3, label='Binned accuracy trend')
    
    # Add linear regression line
    m, b = np.polyfit(df['token_count'], df['is_correct'].astype(int) * 100, 1)
    x_line = np.linspace(df['token_count'].min(), df['token_count'].max(), 100)
    y_line = m * x_line + b
    ax.plot(x_line, y_line, color=COLORS[2], linestyle='--', 
            label=f'Linear trend (slope: {m:.4f})')
    
    # Formatting
    ax.set_xlabel('Token Count')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(-5, 105)
    ax.set_title('Relationship Between Thought Length and Prediction Accuracy')
    ax.legend()
    
    # Add correlation coefficient
    corr = df['token_count'].corr(df['is_correct'].astype(int))
    plt.figtext(0.02, 0.02, f"Correlation coefficient: {corr:.4f}", ha="left", fontsize=9)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'thought_length_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    
    return fig


def create_sample_thoughts_table(df: pd.DataFrame, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create a table with sample thought processes from different length quartiles.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save CSV (if None, just returns DataFrame)
        
    Returns:
        DataFrame with sample thoughts
    """
    # Ensure we have the needed columns
    if not all(col in df.columns for col in ['token_count', 'thought_process']):
        return pd.DataFrame(columns=['Quartile', 'Token Count', 'Thought Process', 'Is Correct'])
    
    # Calculate quartiles
    token_counts = df['token_count'].tolist()
    quartiles = np.percentile(token_counts, [25, 50, 75])
    quartiles = [int(q) for q in quartiles]
    
    # Add token bucket column
    df['token_bucket'] = df['token_count'].apply(lambda x: get_token_bucket(x, quartiles))
    
    # Get samples from each quartile
    samples = []
    for bucket in sorted(df['token_bucket'].unique(), 
                         key=lambda x: int(x.split('-')[0]) if '-' in x else int(x.split('+')[0])):
        # Get one correct and one incorrect example if possible
        bucket_df = df[df['token_bucket'] == bucket]
        
        correct = bucket_df[bucket_df['is_correct'] == True].sample(min(1, sum(bucket_df['is_correct'] == True)))
        incorrect = bucket_df[bucket_df['is_correct'] == False].sample(min(1, sum(bucket_df['is_correct'] == False)))
        
        samples.append(pd.concat([correct, incorrect]))
    
    # Combine all samples
    all_samples = pd.concat(samples)
    
    # Format the table
    formatted_samples = pd.DataFrame({
        'Quartile': all_samples['token_bucket'],
        'Token Count': all_samples['token_count'],
        'Thought Process': all_samples['thought_process'],
        'Is Correct': all_samples['is_correct']
    })
    
    # Sort by token count
    formatted_samples = formatted_samples.sort_values('Token Count')
    
    if output_dir:
        formatted_samples.to_csv(os.path.join(output_dir, 'sample_thoughts.csv'), index=False)
    
    return formatted_samples


def analyze_common_phrases(df: pd.DataFrame, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze common phrases in correct vs. incorrect thought processes.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save CSV (if None, just returns DataFrame)
        
    Returns:
        DataFrame with phrase analysis
    """
    import re
    from collections import Counter
    
    # Ensure we have the needed columns
    if not all(col in df.columns for col in ['thought_process', 'is_correct']):
        return pd.DataFrame(columns=['Phrase', 'Correct Count', 'Incorrect Count', 'Correct %', 'Incorrect %'])
    
    # Function to extract phrases
    def extract_phrases(text, length=3):
        """Extract n-grams from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [' '.join(words[i:i+length]) for i in range(len(words) - length + 1)]
    
    # Collect phrases
    correct_phrases = []
    incorrect_phrases = []
    
    for _, row in df.iterrows():
        if pd.isna(row['thought_process']):
            continue
            
        phrases = extract_phrases(row['thought_process'])
        
        if row['is_correct']:
            correct_phrases.extend(phrases)
        else:
            incorrect_phrases.extend(phrases)
    
    # Count frequencies
    correct_counter = Counter(correct_phrases)
    incorrect_counter = Counter(incorrect_phrases)
    
    # Get total counts
    total_correct = sum(correct_counter.values())
    total_incorrect = sum(incorrect_counter.values())
    
    # Combine and calculate relative frequencies
    all_phrases = set(list(correct_counter.keys()) + list(incorrect_counter.keys()))
    
    phrase_data = []
    for phrase in all_phrases:
        correct_count = correct_counter.get(phrase, 0)
        incorrect_count = incorrect_counter.get(phrase, 0)
        
        correct_pct = correct_count / total_correct * 100 if total_correct > 0 else 0
        incorrect_pct = incorrect_count / total_incorrect * 100 if total_incorrect > 0 else 0
        
        phrase_data.append({
            'Phrase': phrase,
            'Correct Count': correct_count,
            'Incorrect Count': incorrect_count,
            'Correct %': correct_pct,
            'Incorrect %': incorrect_pct,
            'Difference': correct_pct - incorrect_pct
        })
    
    # Convert to DataFrame and sort
    phrase_df = pd.DataFrame(phrase_data)
    
    # Filter to keep only relatively common phrases
    min_count = (total_correct + total_incorrect) * 0.01
    phrase_df = phrase_df[(phrase_df['Correct Count'] > min_count) | 
                          (phrase_df['Incorrect Count'] > min_count)]
    
    # Sort by difference (most differentiating phrases first)
    phrase_df = phrase_df.sort_values('Difference', ascending=False)
    
    if output_dir:
        phrase_df.to_csv(os.path.join(output_dir, 'phrase_analysis.csv'), index=False)
    
    return phrase_df


def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Generate visualizations and analysis from experiment results')
    parser.add_argument('--results-json', type=str, required=True, 
                        help='Path to the JSON results file')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Directory to save output files (default: analysis_output)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load results
    print(f"Loading results from {args.results_json}...")
    metadata, results = load_results(args.results_json)
    
    # Convert to DataFrame
    print("Processing results...")
    df = prepare_dataframe(results)
    
    # Generate and save visualizations
    print("Generating visualizations...")
    
    plot_accuracy_by_token_count(df, args.output_dir)
    plot_token_count_distribution(df, args.output_dir)
    plot_confusion_matrix(df, args.output_dir)
    plot_metrics_summary(df, args.output_dir)
    plot_thought_length_vs_accuracy(df, args.output_dir)
    
    # Generate and save tables
    print("Generating analysis tables...")
    
    examples_table = create_examples_table(df, args.output_dir)
    sample_thoughts = create_sample_thoughts_table(df, args.output_dir)
    phrase_analysis = analyze_common_phrases(df, args.output_dir)
    
    print(f"Analysis complete! Results saved to {args.output_dir}")
    
    # Return data for interactive use (useful for notebooks)
    return {
        'metadata': metadata,
        'dataframe': df,
        'examples': examples_table,
        'sample_thoughts': sample_thoughts,
        'phrase_analysis': phrase_analysis
    }


if __name__ == "__main__":
    main() 