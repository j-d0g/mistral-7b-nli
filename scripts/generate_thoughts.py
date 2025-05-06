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
import csv
import pathlib
import re
import numpy as np

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------

# --- Import project utilities ---
from utils.data_analysis import (
    count_tokens,
    get_token_bucket,
    calculate_statistics,
    calculate_token_bucket_stats,
    generate_summary,
    combine_worker_results
)
# -------------------------------

# --- Import get_prompt function ---
from utils.prompts import get_prompt
# --------------------

from service.prediction_service import predict_label
from dotenv import load_dotenv

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs', 'thoughts'), exist_ok=True)
log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'thoughts', 'generation.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate Chain-of-Thought augmentations for NLI examples using any supported model API.')
parser.add_argument('--input-csv', type=str, default='data/original_data/sample.csv', help='Path to the input CSV file containing premise, hypothesis, label, and id. Defaults to sample.csv.')
parser.add_argument('--output-json', type=str, help='Path to the output JSON file where results will be appended. If not specified, will be auto-generated based on model and input file.')
parser.add_argument('--failed-csv', type=str, help='Path to save details of failed examples. If not specified, will be auto-generated.')
parser.add_argument('--api', type=str, choices=['mistral', 'deepseek'], required=True, help='Which API to use (mistral or deepseek).')
parser.add_argument('--model-name', type=str, help='Name of the model to use. Defaults depend on the API selected.')
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes. Default is 1 (single process).')
parser.add_argument('--start-index', type=int, default=0, help='Start processing from this index in the input CSV.')
parser.add_argument('--end-index', type=int, default=None, help='Stop processing at this index (exclusive) in the input CSV.')
parser.add_argument('--system-prompt', type=str, default='initial_generation', choices=['initial_generation'], help='Which system prompt to use from prompts.py')
args = parser.parse_args()

# Set default model names based on API
if args.model_name is None:
    if args.api == 'mistral':
        args.model_name = 'open-mistral-7b'
    elif args.api == 'deepseek':
        args.model_name = 'deepseek-chat'  # Default DeepSeek model

# Generate default output paths if not provided
if not args.output_json:
    input_base = os.path.basename(args.input_csv).split('.')[0]
    # Use data directory structure
    args.output_json = os.path.join('data', 'original_thoughts', f"{args.model_name}_{input_base}_{args.system_prompt}_output.json")
    logger.info(f"Output file not specified, using: {args.output_json}")

if not args.failed_csv:
    # Use data directory structure
    args.failed_csv = os.path.join('data', 'original_thoughts', f"failed_{args.api}_{args.model_name}_{args.system_prompt}_generation.csv")
    logger.info(f"Failed examples file not specified, using: {args.failed_csv}")

def write_failed_example(file_path, lock, example_data):
    """Append a failed example to the CSV file, ensuring thread/process safety."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with lock:
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'premise', 'hypothesis', 'true_label', 'error_info'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(example_data)

def get_llm_instance(api_name, api_key, model_name=None):
    """Create and return an LLM instance based on the specified API"""
    if api_name == 'mistral':
        from llm.mistral import Mistral
        return Mistral(api_key)
    elif api_name == 'deepseek':
        from llm.deepseek_api import DeepSeekAPI
        # For DeepSeek, we need to pass the model name to get the appropriate model instance
        valid_deepseek_models = ['deepseek-chat', 'deepseek-reasoner']
        if model_name not in valid_deepseek_models and model_name is not None:
            logger.warning(f"Unknown DeepSeek model: {model_name}. Using default 'deepseek-chat'.")
            model_name = 'deepseek-chat'
        return DeepSeekAPI(api_key=api_key, model=model_name)
    else:
        raise ValueError(f"Unsupported API: {api_name}")

def process_chunk(chunk_df, api_name, api_key, model_name, output_json, failed_csv_path, failed_lock, worker_id, results_dict, prompt_type='initial_generation'):
    """Process a chunk of examples using a single worker"""
    logger.info(f"Worker {worker_id} starting to process {len(chunk_df)} examples")
    
    output_count = 0
    correct_count = 0
    failure_count = 0
    # Add tracking for TP, FP, FN for precision, recall, and F1
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    token_counts = []
    
    for index, row in chunk_df.iterrows():
        logger.info(f"Worker {worker_id} - Processing ID: {row['id']} (Index: {index})...")
        process_result = process_single_example(
            row, index, api_name, api_key, model_name,
            output_json, failed_csv_path, failed_lock, worker_id, results_dict,
            prompt_type
        )
        
        # Update counters
        if process_result['success']:
            output_count += 1
            if process_result.get('token_count'):
                token_counts.append(process_result['token_count'])
                
            if process_result['correct']:
                correct_count += 1
                # For true positives, both predicted and true labels must be 1
                if process_result.get('predicted_label', 0) == 1 and process_result.get('true_label', 0) == 1:
                    true_positives += 1
            else:
                # For false positives, predicted is 1 but true is 0
                if process_result.get('predicted_label', 0) == 1 and process_result.get('true_label', 0) == 0:
                    false_positives += 1
                # For false negatives, predicted is 0 but true is 1
                elif process_result.get('predicted_label', 0) == 0 and process_result.get('true_label', 0) == 1:
                    false_negatives += 1
        else:
            failure_count += 1
    
    logger.info(f"Worker {worker_id} finished. Processed {output_count} examples, {correct_count} correct, {failure_count} failures.")
    return {
        'worker_id': worker_id,
        'output_count': output_count,
        'correct_count': correct_count,
        'failure_count': failure_count,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'token_counts': token_counts
    }

def process_single_example(row, index, api_name, api_key, model_name, output_json, failed_csv_path, failed_lock, worker_id, results_dict, prompt_type='initial_generation'):
    """Process a single example and return success/failure status"""
    try:
        # Create a fresh LLM instance for each example to avoid context sharing
        llm = get_llm_instance(api_name, api_key, model_name)
        
        # Get the prompt and schema using get_prompt
        prompt, schema = get_prompt(prompt_type)
        
        # Format the regeneration prompt template if needed
        if prompt_type == 'regeneration':
            # Format the regeneration prompt template
            prompt = prompt.format(
                premise=row['premise'],
                hypothesis=row['hypothesis'],
                true_label=row['true_label']
            )
        
        response_json = predict_label(
            id=row['id'],
            sys=prompt,
            premise=row['premise'],
            hypothesis=row['hypothesis'],
            true_label=row['true_label'],
            llm=llm,
            model_name=model_name,
            json_format=schema,
            json_filepath=output_json  # This will be used internally by predict_label
        )
        
        # Check for valid response
        if response_json and 'predicted_label' in response_json and response_json['predicted_label'] != -1:
            # Estimate token count for the thought process
            token_count = count_tokens(response_json['thought_process'])
            
            # Append the result to our shared dictionary if in parallel mode
            if results_dict is not None:
                results_dict[str(index)] = {
                    'response': response_json,
                    'correct': response_json['predicted_label'] == row['true_label'],
                    'id': row['id'],
                    'predicted_label': response_json['predicted_label'],
                    'true_label': row['true_label'],
                    'token_count': token_count
                }
            
            return {
                'success': True, 
                'correct': response_json['predicted_label'] == row['true_label'],
                'predicted_label': response_json['predicted_label'],
                'true_label': row['true_label'],
                'token_count': token_count
            }
        else:
            logger.warning(f"Worker {worker_id} - Failed to process ID: {row['id']}. Response: {response_json}")
            failed_data = {
                'id': row['id'],
                'premise': row['premise'],
                'hypothesis': row['hypothesis'],
                'true_label': row['true_label'],
                'error_info': f"Invalid response: {response_json}"
            }
            write_failed_example(failed_csv_path, failed_lock, failed_data)
            return {'success': False, 'correct': False}

    except Exception as e:
        logger.error(f"Worker {worker_id} - Error processing ID: {row['id']}: {str(e)}")
        failed_data = {
            'id': row['id'],
            'premise': row['premise'],
            'hypothesis': row['hypothesis'],
            'true_label': row['true_label'],
            'error_info': str(e)
        }
        write_failed_example(failed_csv_path, failed_lock, failed_data)
        return {'success': False, 'correct': False}

def main():
    # Load environment variables (for API key)
    load_dotenv()
    
    # Get appropriate API key based on selected API
    api_key = None
    if args.api == 'mistral':
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            logger.error("Error: MISTRAL_API_KEY not found in environment variables.")
            sys.exit(1)
    elif args.api == 'deepseek':
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            logger.error("Error: DEEPSEEK_API_KEY not found in environment variables.")
            sys.exit(1)
    
    # Load dataset
    try:
        input_df = pd.read_csv(args.input_csv)
        
        # Add id column if it doesn't exist
        if 'id' not in input_df.columns:
            logger.info("'id' column not found in input CSV. Adding 'id' column based on index.")
            input_df['id'] = input_df.index
        
        # Ensure required columns exist
        required_cols = ['premise', 'hypothesis', 'label']
        if not all(col in input_df.columns for col in required_cols):
            raise ValueError(f"Input CSV must contain columns: {', '.join(required_cols)}")
        
        # Rename 'label' to 'true_label' for consistency
        if 'label' in input_df.columns:
            input_df['true_label'] = input_df['label']
        
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {args.input_csv}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Error reading input CSV: {e}")
        sys.exit(1)
    
    # Slice the dataframe based on start/end indices
    if args.end_index is None:
        args.end_index = len(input_df)
    processing_df = input_df.iloc[args.start_index:args.end_index]
    
    logger.info(f"Loaded {len(processing_df)} examples from {args.input_csv}")
    
    # Prepare failed examples file (delete if exists)
    if os.path.exists(args.failed_csv):
        os.remove(args.failed_csv)
        logger.info(f"Removed existing failed examples file: {args.failed_csv}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Single worker mode (simple sequential processing)
    if args.workers == 1:
        logger.info("Running in single worker mode")
        
        # Get the prompt and schema using get_prompt
        prompt, schema = get_prompt(args.system_prompt)
        
        # Process examples sequentially
        output_count = 0
        correct_count = 0
        # Add tracking for precision, recall, F1
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        # Track token counts
        token_counts = []
        
        with open(args.output_json, "a") as outfile:  # Open in append mode
            for index, row in processing_df.iterrows():
                logger.info(f"Processing ID: {row['id']} (Index: {index})...")
                
                # Create a fresh LLM instance for each example
                llm = get_llm_instance(args.api, api_key, args.model_name)
                
                try:
                    # Handle special formatting for regeneration prompt type
                    current_prompt = prompt
                    
                    if args.system_prompt == 'regeneration':
                        # Format the regeneration prompt
                        current_prompt = prompt.format(
                            premise=row['premise'],
                            hypothesis=row['hypothesis'],
                            true_label=row['true_label']
                        )
                    
                    response_json = predict_label(
                        id=row['id'],
                        sys=current_prompt,
                        premise=row['premise'],
                        hypothesis=row['hypothesis'],
                        true_label=row['true_label'],
                        llm=llm,
                        model_name=args.model_name,
                        json_format=schema,
                        json_filepath=args.output_json
                    )
                    
                    # Check for valid response and write to output file
                    if response_json and 'predicted_label' in response_json and response_json['predicted_label'] != -1:
                        output_count += 1
                        
                        # Calculate token count
                        token_count = count_tokens(response_json['thought_process'])
                        token_counts.append(token_count)
                        
                        if response_json['predicted_label'] == row['true_label']:
                            correct_count += 1
                            
                            # For true positives, both predicted and true labels must be 1
                            if response_json['predicted_label'] == 1 and row['true_label'] == 1:
                                true_positives += 1
                        else:
                            # For false positives, predicted is 1 but true is 0
                            if response_json['predicted_label'] == 1 and row['true_label'] == 0:
                                false_positives += 1
                            # For false negatives, predicted is 0 but true is 1
                            elif response_json['predicted_label'] == 0 and row['true_label'] == 1:
                                false_negatives += 1
                    else:
                        logger.warning(f"Failed to process ID: {row['id']}. Response: {response_json}")
                        failed_data = {
                            'id': row['id'],
                            'premise': row['premise'],
                            'hypothesis': row['hypothesis'],
                            'true_label': row['true_label'],
                            'error_info': f"Invalid response: {response_json}"
                        }
                        # Create a dummy lock for single-threaded mode
                        dummy_lock = threading.Lock()
                        write_failed_example(args.failed_csv, dummy_lock, failed_data)
                
                except Exception as e:
                    logger.error(f"Error processing ID: {row['id']}: {str(e)}")
                    failed_data = {
                        'id': row['id'],
                        'premise': row['premise'],
                        'hypothesis': row['hypothesis'],
                        'true_label': row['true_label'],
                        'error_info': str(e)
                    }
                    # Create a dummy lock for single-threaded mode
                    dummy_lock = threading.Lock()
                    write_failed_example(args.failed_csv, dummy_lock, failed_data)
        
        # Compile results
        results = {
            'output_count': output_count,
            'correct_count': correct_count,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        # Calculate statistics
        stats = calculate_statistics(results, token_counts)
        
        # Calculate token bucket statistics
        if token_counts:
            token_buckets = calculate_token_bucket_stats(
                {str(i): {'token_count': tc, 'correct': i < len(token_counts) and True} for i, tc in enumerate(token_counts)},
                token_counts,
                stats['quartiles']
            )
        else:
            token_buckets = {}
        
        # Generate summary
        summary = generate_summary(
            stats,
            token_buckets,
            len(processing_df),
            0,  # No failures in single worker mode (they're immediately handled)
            f"{args.api}/{args.model_name}",
            args.system_prompt,
            multi_worker=False
        )
        
        # Save summary to file
        summary_file_path = f"{args.output_json}_summary.txt"
        with open(summary_file_path, "w") as summary_file:
            summary_file.write(summary)
        
        logger.info(f"Summary saved to {summary_file_path}")
        logger.info(f"Finished processing. Successfully processed {output_count} examples with accuracy {stats['accuracy']:.2f}%")
        logger.info(f"Precision: {stats['precision']:.2f}%, Recall: {stats['recall']:.2f}%, F1 Score: {stats['f1_score']:.2f}%")
        logger.info(f"Average tokens: {stats['avg_tokens']:.2f}, Min: {stats['min_tokens']}, Max: {stats['max_tokens']}")
        
    # Multi-worker mode (parallel processing)
    else:
        logger.info(f"Running in parallel mode with {args.workers} workers")
        
        # Calculate chunk size based on number of workers
        num_workers = args.workers
        chunk_size = math.ceil(len(processing_df) / num_workers)
        
        # Create a shared manager and lock for writing to the failed CSV
        manager = Manager()
        results_dict = manager.dict()
        failed_lock = manager.Lock()
        
        # Create chunks of data for each worker
        chunks = [processing_df.iloc[i:i+chunk_size] for i in range(0, len(processing_df), chunk_size)]
        
        # Track token counts
        all_token_counts = []
        
        # Process chunks in parallel
        start_time = time.time()
        worker_results = []
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    process_chunk,
                    chunk_df=chunk,
                    api_name=args.api,
                    api_key=api_key,
                    model_name=args.model_name,
                    output_json=args.output_json,
                    failed_csv_path=args.failed_csv,
                    failed_lock=failed_lock,
                    worker_id=i,
                    results_dict=results_dict,
                    prompt_type=args.system_prompt
                )
                futures.append(future)
                
            for future in futures:
                result = future.result()
                worker_results.append(result)
                # Collect token counts from all workers
                if 'token_counts' in result:
                    all_token_counts.extend(result['token_counts'])
        
        # Collect and calculate statistics
        combined_results = combine_worker_results(worker_results)
        stats = calculate_statistics(combined_results, all_token_counts)
        
        # Calculate token bucket statistics
        token_buckets = calculate_token_bucket_stats(results_dict, all_token_counts, stats['quartiles'])
        
        # Generate summary
        processing_time = time.time() - start_time
        summary = generate_summary(
            stats,
            token_buckets,
            len(processing_df),
            combined_results['failure_count'],
            f"{args.api}/{args.model_name}",
            args.system_prompt,
            multi_worker=True,
            worker_results=worker_results,
            processing_time=processing_time
        )
        
        # Save summary to file
        summary_file_path = f"{args.output_json}_summary.txt"
        with open(summary_file_path, "w") as summary_file:
            summary_file.write(summary)
        
        logger.info(f"Finished processing. Results saved to {args.output_json}")
        logger.info(f"Failed examples saved to {args.failed_csv} ({combined_results['failure_count']} failures)")
        logger.info(f"Summary saved to {summary_file_path}")
        logger.info(f"Processed {combined_results['output_count']} examples successfully with accuracy {stats['accuracy']:.2f}%")
        logger.info(f"Precision: {stats['precision']:.2f}%, Recall: {stats['recall']:.2f}%, F1 Score: {stats['f1_score']:.2f}%")
        logger.info(f"Average tokens: {stats['avg_tokens']:.2f}, Min: {stats['min_tokens']}, Max: {stats['max_tokens']}")
        logger.info(f"Total processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()