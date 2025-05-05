"""
Utility functions for data analysis in the NLI project.
Functions for token counting, statistical calculations, and summary generation.
"""

import numpy as np
import re
from collections import defaultdict
from transformers import AutoTokenizer

# Load the Mistral tokenizer once
mistral_tokenizer = None

def get_tokenizer():
    """Get or initialize the Mistral tokenizer"""
    global mistral_tokenizer
    if mistral_tokenizer is None:
        try:
            mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        except Exception as e:
            print(f"Failed to load Mistral tokenizer: {e}. Falling back to approximation.")
    return mistral_tokenizer

def count_tokens(text):
    """
    Count tokens using the Mistral tokenizer.
    Falls back to approximation if tokenizer is not available.
    
    Args:
        text (str): The text to count tokens for
        
    Returns:
        int: Number of tokens
    """
    tokenizer = get_tokenizer()
    if tokenizer:
        # Use the actual tokenizer
        return len(tokenizer.encode(text))
    else:
        # Fallback to approximation
        words = re.findall(r'\w+', text)
        punctuation = re.findall(r'[^\w\s]', text)
        return len(words) + len(punctuation)

def get_token_bucket(token_count, quartiles):
    """
    Determine the bucket for a given token count based on quartiles.
    
    Args:
        token_count (int): Number of tokens
        quartiles (list): List of quartile values [Q1, Q2, Q3]
        
    Returns:
        str: Bucket label
    """
    if token_count <= quartiles[0]:
        return f"0-{quartiles[0]}"
    elif token_count <= quartiles[1]:
        return f"{quartiles[0]+1}-{quartiles[1]}"
    elif token_count <= quartiles[2]:
        return f"{quartiles[1]+1}-{quartiles[2]}"
    else:
        return f"{quartiles[2]+1}+"

def calculate_statistics(results, all_token_counts=None):
    """
    Calculate accuracy, precision, recall, F1, and token statistics.
    
    Args:
        results (dict): Dictionary with 'output_count', 'correct_count', 
                       'true_positives', 'false_positives', 'false_negatives'
        all_token_counts (list, optional): List of token counts for all examples
        
    Returns:
        dict: Dictionary containing all computed statistics
    """
    stats = {}
    
    # Extract counts from results
    total_output = results.get('output_count', 0)
    total_correct = results.get('correct_count', 0)
    total_tp = results.get('true_positives', 0)
    total_fp = results.get('false_positives', 0)
    total_fn = results.get('false_negatives', 0)
    
    # Calculate core metrics
    stats['accuracy'] = (total_correct / total_output * 100) if total_output > 0 else 0
    stats['precision'] = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0
    stats['recall'] = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0
    stats['f1_score'] = 2 * (stats['precision'] * stats['recall']) / (stats['precision'] + stats['recall']) if (stats['precision'] + stats['recall']) > 0 else 0
    
    # Token statistics if token counts are provided
    if all_token_counts:
        stats['avg_tokens'] = sum(all_token_counts) / len(all_token_counts) if all_token_counts else 0
        stats['min_tokens'] = min(all_token_counts) if all_token_counts else 0
        stats['max_tokens'] = max(all_token_counts) if all_token_counts else 0
        stats['quartiles'] = np.percentile(all_token_counts, [25, 50, 75]) if all_token_counts else [0, 0, 0]
        stats['quartiles'] = [int(q) for q in stats['quartiles']]
    
    return stats

def calculate_token_bucket_stats(results_dict, token_counts, quartiles):
    """
    Calculate statistics for each token count bucket.
    
    Args:
        results_dict (dict): Dictionary with results for each example
        token_counts (list): List of token counts
        quartiles (list): List of quartile values [Q1, Q2, Q3]
        
    Returns:
        dict: Dictionary with statistics for each bucket
    """
    token_buckets = defaultdict(lambda: {'count': 0, 'correct': 0})
    
    # Assign examples to buckets
    for index_str, data in results_dict.items():
        if 'token_count' in data:
            token_count = data['token_count']
            bucket = get_token_bucket(token_count, quartiles)
            token_buckets[bucket]['count'] += 1
            if data.get('correct', False):
                token_buckets[bucket]['correct'] += 1
    
    return token_buckets

def generate_summary(stats, token_buckets, processing_df_length, total_failed, 
                     model_info, system_prompt, multi_worker=False, 
                     worker_results=None, processing_time=None):
    """
    Generate a summary of the results.
    
    Args:
        stats (dict): Statistics from calculate_statistics
        token_buckets (dict): Token bucket statistics
        processing_df_length (int): Number of examples processed
        total_failed (int): Number of examples that failed
        model_info (str): Information about the model used
        system_prompt (str): System prompt used
        multi_worker (bool): Whether multi-worker mode was used
        worker_results (list, optional): Results from each worker
        processing_time (float, optional): Processing time in seconds
        
    Returns:
        str: Generated summary
    """
    summary = []
    
    # Header
    if multi_worker:
        worker_count = len(worker_results) if worker_results else 0
        summary.append(f"Results Summary (Parallel Processing - {worker_count} workers, {model_info}):")
    else:
        summary.append(f"Results Summary (Single Process - {model_info}):")
    
    # Basic stats
    summary.append(f"System Prompt: {system_prompt}")
    summary.append(f"Total examples attempted: {processing_df_length}")
    summary.append(f"Total examples processed successfully: {stats.get('output_count', 0)}")
    
    if multi_worker:
        summary.append(f"Total examples failed: {total_failed}")
        summary.append(f"Correct predictions (on successful): {stats.get('correct_count', 0)}\n")
    else:
        summary.append(f"Correct predictions: {stats.get('correct_count', 0)}\n")
    
    # Core metrics
    summary.append(f"Accuracy (on successful): {stats.get('accuracy', 0):.2f}%")
    summary.append(f"Precision: {stats.get('precision', 0):.2f}%")
    summary.append(f"Recall: {stats.get('recall', 0):.2f}%")
    summary.append(f"F1 Score: {stats.get('f1_score', 0):.2f}%")
    
    if processing_time:
        summary.append(f"Processing time: {processing_time:.2f} seconds")
    summary.append("")
    
    # Token stats
    summary.append("Token Count Statistics (Using Mistral Tokenizer):")
    summary.append(f"Average tokens: {stats.get('avg_tokens', 0):.2f}")
    summary.append(f"Min tokens: {stats.get('min_tokens', 0)}")
    summary.append(f"Max tokens: {stats.get('max_tokens', 0)}")
    
    if 'quartiles' in stats:
        quartiles = stats['quartiles']
        summary.append(f"Quartiles: {quartiles[0]}, {quartiles[1]}, {quartiles[2]}\n")
    else:
        summary.append("")
    
    # Token bucket accuracy
    summary.append("Token Count vs. Accuracy:")
    for bucket in sorted(token_buckets.keys(), key=lambda x: int(x.split('-')[0]) if '-' in x else int(x.split('+')[0])):
        bucket_count = token_buckets[bucket]['count']
        bucket_correct = token_buckets[bucket]['correct']
        bucket_accuracy = (bucket_correct / bucket_count * 100) if bucket_count > 0 else 0
        summary.append(f"Token Range {bucket}: {bucket_count} examples, {bucket_correct} correct, {bucket_accuracy:.2f}% accuracy")
    
    # Worker stats if applicable
    if multi_worker and worker_results:
        summary.append("\nPer-worker statistics:")
        for result in worker_results:
            worker_id = result.get('worker_id', 0)
            output_count = result.get('output_count', 0)
            failure_count = result.get('failure_count', 0)
            correct_count = result.get('correct_count', 0)
            worker_accuracy = (correct_count / output_count * 100) if output_count > 0 else 0
            summary.append(f"Worker {worker_id}: {output_count} successful, {failure_count} failed, {correct_count} correct ({worker_accuracy:.2f}% accuracy)")
    
    return "\n".join(summary)

def combine_worker_results(worker_results):
    """
    Combine results from multiple workers.
    
    Args:
        worker_results (list): List of dictionaries with worker results
        
    Returns:
        dict: Combined results
    """
    combined = {
        'output_count': sum(result.get('output_count', 0) for result in worker_results),
        'correct_count': sum(result.get('correct_count', 0) for result in worker_results),
        'failure_count': sum(result.get('failure_count', 0) for result in worker_results),
        'true_positives': sum(result.get('true_positives', 0) for result in worker_results),
        'false_positives': sum(result.get('false_positives', 0) for result in worker_results),
        'false_negatives': sum(result.get('false_negatives', 0) for result in worker_results)
    }
    
    return combined 