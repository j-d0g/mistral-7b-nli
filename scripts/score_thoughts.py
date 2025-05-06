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
import glob
import pathlib

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------

# --- Import get_prompt function ---
from utils.prompts import get_prompt
# --------------------

from service.scoring_service import generate_score
from llm.mistral import Mistral
from dotenv import load_dotenv

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs', 'score'), exist_ok=True)
log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'score', 'scoring.log')

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
parser = argparse.ArgumentParser(description='Score and improve chain-of-thought reasoning for NLI examples using any supported model API.')
parser.add_argument('--input-json', type=str, required=True, help='Path to the input JSON file or directory containing JSONs with thought processes to score.')
parser.add_argument('--output-dir', type=str, help='Directory to save gold and low standard outputs. If not specified, will be auto-generated.')
parser.add_argument('--failed-csv', type=str, help='Path to save details of failed examples. If not specified, will be auto-generated.')
parser.add_argument('--api', type=str, choices=['mistral', 'deepseek'], required=True, help='Which API to use (mistral or deepseek).')
parser.add_argument('--model-name', type=str, help='Name of the model to use. Defaults depend on the API selected.')
parser.add_argument('--workers', type=int, default=1, help='Number of worker processes. Default is 1 (single process).')
parser.add_argument('--start-index', type=int, default=0, help='Start processing from this index in the input dataset.')
parser.add_argument('--end-index', type=int, default=None, help='Stop processing at this index (exclusive) in the input dataset.')
args = parser.parse_args()

# Set default model names based on API
if args.model_name is None:
    if args.api == 'mistral':
        args.model_name = 'open-mixtral-8x7b'
    elif args.api == 'deepseek':
        args.model_name = 'deepseek-chat'  # Default DeepSeek model

# Generate default output paths if not provided
if not args.output_dir:
    input_base = os.path.basename(args.input_json).split('.')[0]
    args.output_dir = os.path.join('logs', 'score', f"scored_{args.model_name}_{input_base}")
    logger.info(f"Output directory not specified, using: {args.output_dir}")

if not args.failed_csv:
    args.failed_csv = os.path.join('logs', 'score', f"failed_{args.api}_{args.model_name}_scoring.csv")
    logger.info(f"Failed examples file not specified, using: {args.failed_csv}")

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "gold_standard"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, "low_standard"), exist_ok=True)

def write_failed_example(file_path, lock, example_data):
    """Append a failed example to the CSV file, ensuring thread/process safety."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with lock:
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'premise', 'hypothesis', 'thought_process', 'predicted_label', 'true_label', 'error_info'])
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

def load_input_data(input_path):
    """Load input data from JSON file(s)"""
    if os.path.isdir(input_path):
        # If input is a directory, load all JSON files
        all_files = sorted(glob.glob(os.path.join(input_path, "*.json")))
        dfs = []
        for file in all_files:
            df = pd.read_json(file, lines=True)
            dfs.append(df)
        if not dfs:
            raise ValueError(f"No JSON files found in directory: {input_path}")
        df = pd.concat(dfs)
    else:
        # If input is a file, load it directly
        df = pd.read_json(input_path, lines=True)
    
    # Ensure required columns exist
    required_standard_cols = ['id', 'premise', 'hypothesis', 'true_label']
    if not all(col in df.columns for col in required_standard_cols):
        missing = [col for col in required_standard_cols if col not in df.columns]
        raise ValueError(f"Input JSON missing required columns: {', '.join(missing)}")
        
    # Check for thought process column - could be 'thought_process', 'improved_thought_process', or 'initial_thought'
    thought_cols = ['thought_process', 'improved_thought_process', 'initial_thought']
    thought_process_cols_found = [col for col in thought_cols if col in df.columns]
    if not thought_process_cols_found:
        raise ValueError(f"Input JSON missing thought process column. Need one of: {', '.join(thought_cols)}")
    
    # Check for prediction column - could be 'predicted_label', 'label', 'prediction', or 'initial_label'
    prediction_cols = ['predicted_label', 'label', 'prediction', 'initial_label']
    prediction_cols_found = [col for col in prediction_cols if col in df.columns]
    if not prediction_cols_found:
        raise ValueError(f"Input JSON missing prediction column. Need one of: {', '.join(prediction_cols)}")
    
    # Map the available columns to standard names for consistency
    # First, map thought_process (use the first found column in the preferred order)
    if 'thought_process' not in df.columns:
        for col in thought_cols:
            if col in df.columns:
                logger.info(f"Mapping '{col}' to standard field name 'thought_process'")
                df['thought_process'] = df[col]
                break
    
    # Then, map predicted_label (use the first found column in the preferred order)
    if 'predicted_label' not in df.columns:
        for col in prediction_cols:
            if col in df.columns:
                logger.info(f"Mapping '{col}' to standard field name 'predicted_label'")
                df['predicted_label'] = df[col]
                break
    
    # Sort and deduplicate by id
    df.drop_duplicates(subset=['id'], inplace=True)
    df.sort_values(by='id', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def process_chunk(chunk_df, api_name, api_key, model_name, output_dir, failed_csv_path, failed_lock, worker_id, results_dict):
    """Process a chunk of examples using a single worker"""
    logger.info(f"Worker {worker_id} starting to process {len(chunk_df)} examples")
    
    output_count = 0
    high_score_count = 0
    failure_count = 0
    
    # Get prompt and schema
    scoring_prompt, scoring_schema = get_prompt('scoring')
    
    for index, row in chunk_df.iterrows():
        logger.info(f"Worker {worker_id} - Processing ID: {row['id']} (Index: {index})...")
        process_result = process_single_example(
            row, index, api_name, api_key, model_name,
            output_dir, scoring_prompt, scoring_schema, 
            failed_csv_path, failed_lock, worker_id, results_dict
        )
        
        # Update counters
        if process_result['success']:
            output_count += 1
            if process_result.get('high_score', False):
                high_score_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Worker {worker_id} finished. Processed {output_count} examples, {high_score_count} high-quality, {failure_count} failures.")
    return {
        'worker_id': worker_id,
        'output_count': output_count,
        'high_score_count': high_score_count,
        'failure_count': failure_count
    }

def persist_benchmarks(id,
                       premise,
                       hypothesis,
                       thought_process,
                       predicted_label,
                       true_label,
                       score_response,
                       reprompts,
                       file_path
                       ):
    benchmarks = {
        "id": id,
        "premise": premise,
        "hypothesis": hypothesis,
        "thought_process": thought_process,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "score_json": score_response,
        "reprompt_counts": reprompts
    }

    with open(file_path, "a") as file:
        json.dump(benchmarks, file)
        file.write("\n")

def process_single_example(row, index, api_name, api_key, model_name, output_dir, 
                         scoring_prompt, scoring_schema, failed_csv_path, 
                         failed_lock, worker_id, results_dict):
    """Process a single example and return success/failure status"""
    try:
        # Create a fresh LLM instance for each example to avoid context sharing
        llm = get_llm_instance(api_name, api_key, model_name)
        
        # Use generate_score from scoring_service
        response_json = generate_score(
            id=row['id'],
            sys=scoring_prompt,
            premise=row['premise'],
            hypothesis=row['hypothesis'],
            thought_process=row['thought_process'],
            predicted_label=row['predicted_label'],
            true_label=row['true_label'],
            llm=llm,
            model_name=model_name,
            json_format=scoring_schema,
            json_filepath=output_dir  # Will be used internally by generate_score
        )
        
        # Check for valid response
        if response_json and 'score' in response_json:
            is_high_score = response_json['score'] >= 4
            
            # Append the result to our shared dictionary if in parallel mode
            if results_dict is not None:
                results_dict[str(index)] = {
                    'response': response_json,
                    'high_score': is_high_score,
                    'id': row['id']
                }
            
            return {
                'success': True,
                'high_score': is_high_score
            }
        else:
            logger.warning(f"Worker {worker_id} - Failed to process ID: {row['id']}. Response: {response_json}")
            failed_data = {
                'id': row['id'],
                'premise': row['premise'],
                'hypothesis': row['hypothesis'],
                'thought_process': row['thought_process'],
                'predicted_label': row['predicted_label'],
                'true_label': row['true_label'],
                'error_info': f"Invalid response: {response_json}"
            }
            write_failed_example(failed_csv_path, failed_lock, failed_data)
            return {'success': False, 'high_score': False}

    except Exception as e:
        logger.error(f"Worker {worker_id} - Error processing ID: {row['id']}: {str(e)}")
        failed_data = {
            'id': row['id'],
            'premise': row['premise'],
            'hypothesis': row['hypothesis'],
            'thought_process': row['thought_process'],
            'predicted_label': row['predicted_label'],
            'true_label': row['true_label'],
            'error_info': str(e)
        }
        write_failed_example(failed_csv_path, failed_lock, failed_data)
        return {'success': False, 'high_score': False}

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
        input_df = load_input_data(args.input_json)
        logger.info(f"Loaded {len(input_df)} examples from {args.input_json}")
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        sys.exit(1)
    
    # Slice the dataframe based on start/end indices
    if args.end_index is None:
        args.end_index = len(input_df)
    processing_df = input_df.iloc[args.start_index:args.end_index]
    
    logger.info(f"Processing {len(processing_df)} examples from index {args.start_index} to {args.end_index}")
    
    # Prepare failed examples file (delete if exists)
    if os.path.exists(args.failed_csv):
        os.remove(args.failed_csv)
        logger.info(f"Removed existing failed examples file: {args.failed_csv}")
    
    # Single worker mode (simple sequential processing)
    if args.workers == 1:
        logger.info("Running in single worker mode")
        
        # Get the prompt and schema
        scoring_prompt, scoring_schema = get_prompt('scoring')
        
        # Process examples sequentially
        output_count = 0
        high_score_count = 0
        
        for index, row in processing_df.iterrows():
            logger.info(f"Processing ID: {row['id']} (Index: {index})...")
            
            # Create a fresh LLM instance for each example
            llm = get_llm_instance(args.api, api_key, args.model_name)
            
            try:
                response_json = generate_score(
                    id=row['id'],
                    sys=scoring_prompt,
                    premise=row['premise'],
                    hypothesis=row['hypothesis'],
                    thought_process=row['thought_process'],
                    predicted_label=row['predicted_label'],
                    true_label=row['true_label'],
                    llm=llm,
                    model_name=args.model_name,
                    json_format=scoring_schema,
                    json_filepath=args.output_dir
                )
                
                # Check for valid response
                if response_json and 'score' in response_json:
                    output_count += 1
                    if response_json['score'] >= 4:
                        high_score_count += 1
                else:
                    logger.warning(f"Failed to process ID: {row['id']}. Response: {response_json}")
                    failed_data = {
                        'id': row['id'],
                        'premise': row['premise'],
                        'hypothesis': row['hypothesis'],
                        'thought_process': row['thought_process'],
                        'predicted_label': row['predicted_label'],
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
                    'thought_process': row['thought_process'],
                    'predicted_label': row['predicted_label'],
                    'true_label': row['true_label'],
                    'error_info': str(e)
                }
                # Create a dummy lock for single-threaded mode
                dummy_lock = threading.Lock()
                write_failed_example(args.failed_csv, dummy_lock, failed_data)
        
        # Calculate statistics
        if output_count > 0:
            high_score_percentage = (high_score_count / output_count) * 100
        else:
            high_score_percentage = 0
            
        # Print and save summary
        logger.info(f"Finished processing. Successfully processed {output_count} examples with {high_score_percentage:.2f}% high-quality thought processes")
        
        # Save summary to a file
        summary_file_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file_path, "w") as summary_file:
            summary_file.write(f"Results Summary (Single Process - {args.api}/{args.model_name}):\n")
            summary_file.write(f"Total examples attempted: {len(processing_df)}\n")
            summary_file.write(f"Total examples processed successfully: {output_count}\n")
            summary_file.write(f"High-quality thought processes (score >= 4): {high_score_count}\n")
            summary_file.write(f"High-quality percentage: {high_score_percentage:.2f}%\n")
        
        logger.info(f"Summary saved to {summary_file_path}")
        
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
                    output_dir=args.output_dir,
                    failed_csv_path=args.failed_csv,
                    failed_lock=failed_lock,
                    worker_id=i,
                    results_dict=results_dict
                )
                futures.append(future)
                
            for future in futures:
                worker_results.append(future.result())
        
        # Calculate overall statistics
        total_output = sum(result['output_count'] for result in worker_results)
        total_high_score = sum(result['high_score_count'] for result in worker_results)
        total_failed = sum(result['failure_count'] for result in worker_results)
        high_score_percentage = (total_high_score / total_output * 100) if total_output > 0 else 0
        
        # Save summary to a file
        summary_file_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_file_path, "w") as summary_file:
            summary_file.write(f"Results Summary (Parallel Processing - {num_workers} workers, {args.api}/{args.model_name}):\n")
            summary_file.write(f"Total examples attempted: {len(processing_df)}\n")
            summary_file.write(f"Total examples processed successfully: {total_output}\n")
            summary_file.write(f"Total examples failed: {total_failed}\n")
            summary_file.write(f"High-quality thought processes (score >= 4): {total_high_score}\n")
            summary_file.write(f"High-quality percentage: {high_score_percentage:.2f}%\n")
            summary_file.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
            
            # Add per-worker statistics
            summary_file.write("\nPer-worker statistics:\n")
            for result in worker_results:
                worker_high_score_pct = (result['high_score_count'] / result['output_count'] * 100) if result['output_count'] > 0 else 0
                summary_file.write(f"Worker {result['worker_id']}: {result['output_count']} successful, {result['failure_count']} failed, "
                                  f"{result['high_score_count']} high-quality ({worker_high_score_pct:.2f}%)\n")
        
        logger.info(f"Finished processing. Results saved to {args.output_dir}")
        logger.info(f"Failed examples saved to {args.failed_csv} ({total_failed} failures)")
        logger.info(f"Summary saved to {summary_file_path}")
        logger.info(f"Processed {total_output} examples successfully with {high_score_percentage:.2f}% high-quality thought processes")
        logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
