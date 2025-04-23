#!/usr/bin/env python3
import pandas as pd
import sys
import argparse
import os
import json
import time
import multiprocessing
from multiprocessing import Manager, Lock
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math
import logging
import threading
from collections import deque
import csv

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------

from service.prediction_service import predict_label
from llm.mistral import Mistral
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parallel_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate Chain-of-Thought augmentations for NLI examples using multiple workers.')
parser.add_argument('--input-csv', type=str, required=True, help='Path to the input CSV file containing premise, hypothesis, label, and id.')
parser.add_argument('--output-json', type=str, required=True, help='Path to the output JSON file where results will be appended.')
parser.add_argument('--failed-csv', type=str, default='failed_parallel_generation.csv', help='Path to save details of failed examples.')
parser.add_argument('--model-name', type=str, default='open-mistral-7b', help='Name of the Mistral model to use.')
parser.add_argument('--workers', type=int, default=4, help='Number of worker processes (max 4 recommended).')
parser.add_argument('--rate-limit', type=int, default=6, help='API rate limit per second.')
args = parser.parse_args()

# Define Prompt
json_schema = {
    "thought_process": "<deductive/common-sense reasoning steps>",
    "label": "<0 or 1>"
}

system_prompt = """You are an expert in natural language reasoning and inference. Your task is to analyze pairs of sentences and determine if the second sentence (hypothesis) can be logically inferred from the first sentence (premise).

For each example, I will provide the premise and hypothesis. Your response should be in the following JSON format:
{
    "thought_process": "Step 1. <Identify key information and relationships in the premise, considering logical connections, commonsense understanding, and factual consistency>. Step 2. <Analyze how the hypothesis relates to or contradicts the premise based on the information identified in Step 1. Evaluate if the hypothesis can be reasonably inferred from the premise>. Step 3. <Explain your final reasoning and conclusion on whether the hypothesis is entailed by the premise or not>",
    "label": "<0 for no entailment, 1 for entailment>"
}
Please provide a clear multi-step reasoning chain explaining how you arrived at your final answer, breaking it down into logical components. Ground your response in the given information, logical principles and common-sense reasoning.

Example:
Premise: The dog chased the cat up the tree. Hypothesis: The cat climbed the tree.

Your response:
{
  "thought_process": "Step 1: the premise indicates a scenario where a dog chases a cat, resulting in the cat moving up a tree. The movement 'up the tree' suggests a vertical ascent, typical of climbing behavior. It is common sense that a cat would climb a tree to escape a chasing dog, and there are no known facts that contradict the premise or hypothesis. Step 2: 'The cat climbed the tree' can be logically inferred from the premise because the action of climbing is a reasonable and necessary part of the cat moving 'up the tree' as described. Thus, the hypothesis logically follows from the premise. Step 3: Based on the logical reasoning, common sense, and lack of contradictory facts, the hypothesis can be inferred from the premise.",
  "label": 1
}
"""

class SimpleRateLimiter:
    """A simple rate limiter using a sliding window of timestamps"""
    
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
        
    def __enter__(self):
        """Block if rate limit is exceeded, then allow the call"""
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than the period
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            
            # If at limit, sleep until oldest call "expires"
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()  # Update now after sleeping
                    
                    # Clean up again after sleeping
                    while self.calls and now - self.calls[0] > self.period:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(now)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager"""
        pass

def write_failed_example(file_path, lock, example_data):
    """Append a failed example to the CSV file, ensuring thread/process safety."""
    with lock:
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'premise', 'hypothesis', 'label', 'error_info'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(example_data)

def process_chunk(chunk_df, api_key, model_name, json_schema, output_json, failed_csv_path, failed_lock, rate_limit_per_worker, worker_id, results_dict):
    """Process a chunk of examples using a single worker"""
    logger.info(f"Worker {worker_id} starting to process {len(chunk_df)} examples")
    
    # Create a rate limiter for this worker
    rate_limiter = SimpleRateLimiter(max_calls=rate_limit_per_worker, period=1)
    
    output_count = 0
    correct_count = 0
    failure_count = 0
    
    for index, row in chunk_df.iterrows():
        with rate_limiter:
            logger.info(f"Worker {worker_id} - Processing ID: {row['id']} (Index: {index})...")
            
            # Create a fresh LLM instance for each example to avoid context sharing
            llm = Mistral(api_key)
            try:
                response_json = predict_label(
                    id=row['id'],
                    sys=system_prompt,
                    premise=row['premise'],
                    hypothesis=row['hypothesis'],
                    true_label=row['label'],
                    llm=llm,
                    model_name=model_name,
                    json_format=json_schema,
                    json_filepath=output_json  # This will be used internally by predict_label
                )
                
                # Check for valid response
                if response_json and 'label' in response_json and response_json['label'] != -1:
                    # Append the result to our shared dictionary
                    results_dict[str(index)] = {
                        'response': response_json,
                        'correct': response_json['label'] == row['label'],
                        'id': row['id']
                    }
                    
                    output_count += 1
                    if response_json['label'] == row['label']:
                        correct_count += 1
                else:
                    logger.warning(f"Worker {worker_id} - Failed to process ID: {row['id']}. Response: {response_json}")
                    failure_count += 1
                    failed_data = {
                        'id': row['id'],
                        'premise': row['premise'],
                        'hypothesis': row['hypothesis'],
                        'label': row['label'],
                        'error_info': f"Invalid response: {response_json}"
                    }
                    write_failed_example(failed_csv_path, failed_lock, failed_data)

            except Exception as e:
                logger.error(f"Worker {worker_id} - Error processing ID: {row['id']}: {str(e)}")
                failure_count += 1
                failed_data = {
                    'id': row['id'],
                    'premise': row['premise'],
                    'hypothesis': row['hypothesis'],
                    'label': row['label'],
                    'error_info': str(e)
                }
                write_failed_example(failed_csv_path, failed_lock, failed_data)
    
    logger.info(f"Worker {worker_id} finished. Processed {output_count} examples, {correct_count} correct, {failure_count} failures.")
    return {
        'worker_id': worker_id,
        'output_count': output_count,
        'correct_count': correct_count,
        'failure_count': failure_count
    }

def main():
    # Load environment variables (for API key)
    load_dotenv()
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        logger.error("Error: MISTRAL_API_KEY not found in environment variables.")
        sys.exit(1)
    
    # Load dataset
    try:
        input_df = pd.read_csv(args.input_csv)
        # Ensure required columns exist
        required_cols = ['id', 'premise', 'hypothesis', 'label']
        if not all(col in input_df.columns for col in required_cols):
            raise ValueError(f"Input CSV must contain columns: {', '.join(required_cols)}")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {args.input_csv}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Error reading input CSV: {e}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(input_df)} examples from {args.input_csv}")
    
    # Prepare failed examples file (delete if exists)
    if os.path.exists(args.failed_csv):
        os.remove(args.failed_csv)
        logger.info(f"Removed existing failed examples file: {args.failed_csv}")
        
    # Calculate chunk size based on number of workers
    num_workers = min(args.workers, 4)  # Cap at 4 workers as recommended
    chunk_size = math.ceil(len(input_df) / num_workers)
    
    # Calculate rate limit per worker
    rate_limit_per_worker = max(1, args.rate_limit // num_workers)
    
    logger.info(f"Starting parallel processing with {num_workers} workers")
    logger.info(f"Rate limit set to {rate_limit_per_worker} requests/second per worker")
    
    # Create a shared manager and lock for writing to the failed CSV
    manager = Manager()
    results_dict = manager.dict()
    failed_lock = manager.Lock() # Use Manager Lock for ProcessPoolExecutor
    
    # Create chunks of data for each worker
    chunks = [input_df.iloc[i:i+chunk_size] for i in range(0, len(input_df), chunk_size)]
    
    # Process chunks in parallel
    start_time = time.time()
    worker_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Pass arguments directly to the submit call
        futures = []
        for i, chunk in enumerate(chunks):
            future = executor.submit(
                process_chunk,
                chunk_df=chunk,
                api_key=api_key,
                model_name=args.model_name,
                json_schema=json_schema,
                output_json=args.output_json,
                failed_csv_path=args.failed_csv,
                failed_lock=failed_lock,
                rate_limit_per_worker=rate_limit_per_worker,
                worker_id=i,
                results_dict=results_dict
            )
            futures.append(future)
            
        for future in futures:
            worker_results.append(future.result())
    
    # Write all results to the output file (ensure sorted by index for consistency if needed)
    sorted_indices = sorted(results_dict.keys(), key=int)
    with open(args.output_json, "a") as outfile:
        for index_str in sorted_indices:
            json.dump(results_dict[index_str]['response'], outfile)
            outfile.write('\n')
    
    # Calculate overall statistics
    total_output = sum(result['output_count'] for result in worker_results)
    total_correct = sum(result['correct_count'] for result in worker_results)
    total_failed = sum(result['failure_count'] for result in worker_results)
    accuracy = (total_correct / total_output * 100) if total_output > 0 else 0
    
    # Save summary to a file
    summary_file_path = f"{args.output_json}_summary.txt"
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Results Summary (Parallel Processing - {num_workers} workers):\n")
        summary_file.write(f"Total examples attempted: {len(input_df)}\n")
        summary_file.write(f"Total examples processed successfully: {total_output}\n")
        summary_file.write(f"Total examples failed: {total_failed}\n")
        summary_file.write(f"Correct predictions (on successful): {total_correct}\n")
        summary_file.write(f"Accuracy (on successful): {accuracy:.2f}%\n")
        summary_file.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
        
        # Add per-worker statistics
        summary_file.write("\nPer-worker statistics:\n")
        for result in worker_results:
            worker_accuracy = (result['correct_count'] / result['output_count'] * 100) if result['output_count'] > 0 else 0
            summary_file.write(f"Worker {result['worker_id']}: {result['output_count']} successful, {result['failure_count']} failed, {result['correct_count']} correct ({worker_accuracy:.2f}% accuracy)\n")
    
    logger.info(f"Finished processing. Results saved to {args.output_json}")
    logger.info(f"Failed examples saved to {args.failed_csv} ({total_failed} failures)")
    logger.info(f"Summary saved to {summary_file_path}")
    logger.info(f"Processed {total_output} examples successfully with accuracy {accuracy:.2f}%")
    logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main() 