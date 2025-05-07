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
from pprint import pprint
from collections import OrderedDict

# --- Add project root to sys.path ---
# Ensure this path is correct relative to the script's location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------

# --- Import necessary components ---
from utils.prompts import get_prompt
from utils.json_helpers import clean_json # Assuming json_helpers exists
from models.response_models import NLIResponse # For potential reuse, though schema differs
from dotenv import load_dotenv
from service.reflection_service import generate_reflection

# --- Configure logging ---
# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs', 'reflections'), exist_ok=True)
log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'reflections', 'reflection_generation.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file), # Use the new log file location
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Functions (to be copied/adapted/imported) ---

# Placeholder: Function to get LLM instance (adapt from generate_thoughts.py)
def get_llm_instance(api_name, api_key, model_name=None):
    """Create and return an LLM instance based on the specified API"""
    logger.info(f"Getting LLM instance for API: {api_name}, Model: {model_name}")
    if api_name == 'mistral':
        try:
            from llm.mistral import Mistral
            return Mistral(api_key)
        except ImportError:
            logger.error("Mistral LLM class not found. Ensure llm/mistral.py exists.")
            raise
    elif api_name == 'deepseek':
        try:
            from llm.deepseek_api import DeepSeekAPI
            valid_deepseek_models = ['deepseek-chat', 'deepseek-coder']
            if model_name not in valid_deepseek_models and model_name is not None:
                logger.warning(f"Unknown DeepSeek model: {model_name}. Check compatibility.")
            return DeepSeekAPI(api_key=api_key, model=model_name)
        except ImportError:
            logger.error("DeepSeekAPI LLM class not found. Ensure llm/deepseek_api.py exists.")
            raise
    else:
        raise ValueError(f"Unsupported API: {api_name}")


# Placeholder: Function to write failed examples (adapt from generate_thoughts.py)
def write_failed_example(file_path, lock, example_data):
    """Append a failed example to the CSV file, ensuring thread/process safety."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with lock:
        file_exists = os.path.isfile(file_path)
        # Define fieldnames based on expected data for reflection failures
        fieldnames = ['id', 'premise', 'hypothesis', 'true_label', 'thought_process', 'predicted_label', 'error_info']
        # Filter example_data to only include keys present in fieldnames
        filtered_data = {k: example_data.get(k, 'N/A') for k in fieldnames}

        try:
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(file_path) == 0:
                    writer.writeheader()
                writer.writerow(filtered_data)
        except IOError as e:
            logger.error(f"Failed to write to failed examples file {file_path}: {e}")


# Placeholder: Function to validate reflection response
def validate_reflection_response(output: str, schema: dict):
    """
    Validates the LLM output against the reflection schema.
    :param output: Output string from the LLM.
    :param schema: The expected reflection schema dictionary.
    :return: Tuple (validated_dict, error_message)
    """
    cleaned_json_str = clean_json(output)
    try:
        json_data = json.loads(cleaned_json_str)

        # Basic structural validation based on schema keys
        missing_keys = [key for key in schema if key not in json_data]
        if missing_keys:
            return None, f"JSON Validation Error: Missing keys - {', '.join(missing_keys)}. Expected: {list(schema.keys())}. Output: {cleaned_json_str}"

        # Optional: Add more specific type checks if needed (e.g., predicted_label is int)
        if 'predicted_label' in json_data and not isinstance(json_data['predicted_label'], int):
             # Attempt conversion if string representation of int
            if isinstance(json_data['predicted_label'], str) and json_data['predicted_label'].isdigit():
                try:
                    json_data['predicted_label'] = int(json_data['predicted_label'])
                except ValueError:
                     return None, f"JSON Validation Error: 'predicted_label' must be an integer or string digit. Got: {json_data['predicted_label']}"
            else:
                 return None, f"JSON Validation Error: 'predicted_label' must be an integer. Got type: {type(json_data['predicted_label'])}"
        # Backward compatibility for 'label' field
        elif 'label' in json_data and not isinstance(json_data['label'], int):
             # Attempt conversion if string representation of int
            if isinstance(json_data['label'], str) and json_data['label'].isdigit():
                try:
                    json_data['label'] = int(json_data['label'])
                except ValueError:
                     return None, f"JSON Validation Error: 'label' must be an integer or string digit. Got: {json_data['label']}"
            else:
                 return None, f"JSON Validation Error: 'label' must be an integer. Got type: {type(json_data['label'])}"

        # Check if predicted_label is 0 or 1
        if 'predicted_label' in json_data and json_data['predicted_label'] not in [0, 1]:
             return None, f"JSON Validation Error: 'predicted_label' must be 0 or 1. Got: {json_data['predicted_label']}"
        # Backward compatibility for 'label' field
        elif 'label' in json_data and json_data['label'] not in [0, 1]:
             return None, f"JSON Validation Error: 'label' must be 0 or 1. Got: {json_data['label']}"

        # Rename 'label' to 'predicted_label' for consistency
        if 'label' in json_data and 'predicted_label' not in json_data:
            json_data['predicted_label'] = json_data.pop('label')

        return json_data, None # Return the parsed dict if valid

    except json.JSONDecodeError as e:
        return None, f"JSON Decode Error: {str(e)}. Output: {cleaned_json_str}"
    except Exception as e: # Catch other potential errors during validation
        return None, f"Unexpected Validation Error: {str(e)}. Output: {cleaned_json_str}"


# --- Main Processing Function for Single Example ---
def process_single_reflection(example_data, args, api_key, failed_lock, failed_csv_path):
    """
    Processes a single example identified as incorrect.
    """
    example_id = example_data.get('id', 'UNKNOWN_ID')
    logger.info(f"Processing reflection for ID: {example_id}...")

    try:
        # 1. Extract necessary data
        premise = example_data.get('premise')
        hypothesis = example_data.get('hypothesis')
        thought_process = example_data.get('thought_process')
        predicted_label = example_data.get('predicted_label')
        true_label = example_data.get('true_label')

        # Basic validation of extracted data
        if None in [premise, hypothesis, thought_process, predicted_label, true_label]:
            missing = [k for k,v in {
                'premise': premise, 
                'hypothesis': hypothesis, 
                'thought_process': thought_process, 
                'predicted_label': predicted_label, 
                'true_label': true_label
            }.items() if v is None]
            raise ValueError(f"Missing required fields in input JSON: {missing}")

        # Ensure labels are integers if possible
        try:
            predicted_label = int(predicted_label)
            true_label = int(true_label)
        except (ValueError, TypeError):
            raise ValueError(f"Initial or True label is not a valid integer. Initial: {predicted_label}, True: {true_label}")

        # 2. Get LLM instance
        llm = get_llm_instance(args.api, api_key, args.model_name)
        if not llm:
            raise RuntimeError("Failed to initialize LLM instance.")

        # 3. Call the reflection service directly
        reflection_result = generate_reflection(
            id=example_id,
            premise=premise,
            hypothesis=hypothesis,
            thought_process=thought_process,
            predicted_label=predicted_label,
            true_label=true_label,
            llm=llm,
            model_name=args.model_name,
            json_filepath=args.output_reflection_json,
            max_retries=args.max_retries
        )

        # 4. Handle result
        if 'error' in reflection_result:
            logger.error(f"Failed reflection generation for ID {example_id}: {reflection_result['error']} - {reflection_result.get('details')}")
            fail_data = {**example_data, 'error_info': reflection_result.get('details', reflection_result['error'])}
            write_failed_example(failed_csv_path, failed_lock, fail_data)
            return None  # Indicate failure
        else:
            logger.info(f"Successfully generated reflection for ID {example_id}.")
            return reflection_result  # Return the reflection dict

    except Exception as e:
        logger.error(f"Critical error processing reflection for ID {example_id}: {str(e)}", exc_info=True)
        fail_data = {**example_data, 'error_info': f"Critical error: {str(e)}"}
        write_failed_example(failed_csv_path, failed_lock, fail_data)
        return None


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate reflections for incorrectly predicted NLI examples using the reflection prompt.')
    parser.add_argument('--input-thoughts-json', type=str, default='data/original_thoughts/sample_thoughts.json', help='Path to the input JSON Lines file containing initial thoughts. Defaults to sample_thoughts.json.')
    parser.add_argument('--output-reflection-json', type=str, default='data/reflected_thoughts/sample_reflections.json', help='Path to the output JSON Lines file where reflection results will be saved. Defaults to sample_reflections.json.')
    parser.add_argument('--failed-csv', type=str, help='Path to save details of examples that failed during reflection generation. If not specified, will be auto-generated.')
    parser.add_argument('--api', type=str, default='mistral', choices=['mistral', 'deepseek'], help='Which API to use (mistral or deepseek).')
    parser.add_argument('--model-name', type=str, default='open-mistral-nemo', help='Name of the model to use for reflection (e.g., open-mistral-7b, deepseek-coder).')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for parallel execution. Default is 1.')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retries for LLM generation per example.')
    # Optional start/end index for large files
    parser.add_argument('--start-index', type=int, default=0, help='Start processing from this index in the input JSON.')
    parser.add_argument('--end-index', type=int, default=None, help='Stop processing at this index (exclusive) in the input JSON.')

    args = parser.parse_args()

    # Auto-generate failed CSV path if not provided
    if not args.failed_csv:
        input_base = os.path.basename(args.input_thoughts_json).split('.')[0]
        args.failed_csv = os.path.join('data', 'reflected_thoughts', f"failed_reflection_{args.api}_{args.model_name}_{input_base}.csv")
        logger.info(f"Failed examples CSV not specified, using: {args.failed_csv}")

    # Make output path use data/reflected_thoughts directory if not absolute path
    if not os.path.isabs(args.output_reflection_json) and not args.output_reflection_json.startswith('data/'):
        filename = os.path.basename(args.output_reflection_json)
        args.output_reflection_json = os.path.join('data', 'reflected_thoughts', filename)
        logger.info(f"Using data/reflected_thoughts directory for output: {args.output_reflection_json}")

    return args

# --- Main Function ---
def main():
    args = parse_arguments()

    # Load environment variables (for API keys)
    load_dotenv()
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
    else:
        logger.error(f"Unsupported API specified: {args.api}")
        sys.exit(1)

    # Prepare failed examples file (delete if exists)
    if os.path.exists(args.failed_csv):
        try:
            os.remove(args.failed_csv)
            logger.info(f"Removed existing failed examples file: {args.failed_csv}")
        except OSError as e:
            logger.warning(f"Could not remove existing failed examples file {args.failed_csv}: {e}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_reflection_json)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Could not create output directory {output_dir}: {e}")
            sys.exit(1)

    # --- Load and Filter Input Data ---
    incorrect_examples = []
    total_lines = 0
    try:
        with open(args.input_thoughts_json, 'r', encoding='utf-8') as f_in:
            for line_num, line in enumerate(f_in):
                total_lines += 1
                try:
                    data = json.loads(line.strip())
                    # Check if 'correct' key exists and is explicitly False
                    if 'correct' in data and data['correct'] is False:
                        # Add line number for potential sorting later if needed
                        data['_line_num'] = line_num
                        incorrect_examples.append(data)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {line_num + 1} in {args.input_thoughts_json}")
                except KeyError:
                    logger.warning(f"Skipping line {line_num + 1} due to missing 'correct' key in {args.input_thoughts_json}")
                except Exception as e:
                    logger.warning(f"Skipping line {line_num + 1} due to unexpected error: {e} in {args.input_thoughts_json}")

    except FileNotFoundError:
        logger.error(f"Error: Input thoughts file not found at {args.input_thoughts_json}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading input thoughts file {args.input_thoughts_json}: {e}")
        sys.exit(1)

    logger.info(f"Read {total_lines} lines from {args.input_thoughts_json}. Found {len(incorrect_examples)} examples marked as incorrect.")

    if not incorrect_examples:
        logger.info("No incorrect examples found to process. Exiting.")
        sys.exit(0)

    # Apply start/end indices if provided
    if args.start_index > 0 or args.end_index is not None:
        end_idx = args.end_index if args.end_index is not None else len(incorrect_examples)
        incorrect_examples = incorrect_examples[args.start_index:end_idx]
        logger.info(f"Processing subset from index {args.start_index} to {end_idx}. ({len(incorrect_examples)} examples)")

    # --- Start Processing ---
    start_time = time.time()
    processed_count = 0
    success_count = 0
    fail_count = 0

    # Create a dummy lock for single-threaded mode consistency
    dummy_lock = threading.Lock()
    failed_lock = dummy_lock  # Default to dummy lock

    # Use Manager and Lock for multiprocessing if workers > 1
    manager = None
    results_list = []  # Use a simple list for single-threaded

    if args.workers > 1:
        manager = Manager()
        failed_lock = manager.Lock()
        results_list = manager.list()  # Shared list for results
        logger.info(f"Running in parallel mode with {args.workers} workers.")

        # Prepare args for map
        tasks = [(ex, args, api_key, failed_lock, args.failed_csv) for ex in incorrect_examples]

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_reflection, *task) for task in tasks]
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    processed_count += 1
                    if result:
                        results_list.append(result)  # Append successful results
                        success_count += 1
                    else:
                        fail_count += 1
                    if processed_count % 10 == 0:  # Log progress periodically
                        logger.info(f"Progress: {processed_count}/{len(incorrect_examples)} examples processed...")
                except Exception as exc:
                    logger.error(f'Worker generated an exception for example ~line {incorrect_examples[i].get("_line_num", "N/A")}: {exc}', exc_info=True)
                    fail_count += 1
                    # Log failure to CSV manually here if exception bypasses process_single_reflection's logging
                    fail_data = {**incorrect_examples[i], 'error_info': f"Unhandled exception in worker: {exc}"}
                    write_failed_example(args.failed_csv, failed_lock, fail_data)

    else:  # Single worker mode
        logger.info("Running in single worker mode.")
        for i, example in enumerate(incorrect_examples):
            result = process_single_reflection(example, args, api_key, failed_lock, args.failed_csv)
            processed_count += 1
            if result:
                results_list.append(result)  # Append successful results
                success_count += 1
            else:
                fail_count += 1
            if processed_count % 10 == 0:  # Log progress periodically
                logger.info(f"Progress: {processed_count}/{len(incorrect_examples)} examples processed...")

    # --- Final Summary ---
    end_time = time.time()
    duration = end_time - start_time
    logger.info("=" * 30 + " Summary " + "=" * 30)
    logger.info(f"Input File:          {args.input_thoughts_json}")
    logger.info(f"Output File:         {args.output_reflection_json}")
    logger.info(f"Failed CSV:          {args.failed_csv}")
    logger.info(f"API:                 {args.api}")
    logger.info(f"Model:               {args.model_name}")
    logger.info(f"Workers:             {args.workers}")
    logger.info(f"Total Incorrect:     {len(incorrect_examples)}")
    logger.info(f"Successfully Reflected: {success_count}")
    logger.info(f"Failed Reflection:   {fail_count}")
    logger.info(f"Total Duration:      {duration:.2f} seconds")
    logger.info("=" * 69)

    # Save summary to a file
    summary_file_path = f"{args.output_reflection_json}_summary.txt"
    try:
        with open(summary_file_path, "w", encoding='utf-8') as summary_file:
            summary_file.write("=" * 30 + " Reflection Summary " + "=" * 30 + "\n")
            summary_file.write(f"Input File:          {args.input_thoughts_json}\n")
            summary_file.write(f"Output File:         {args.output_reflection_json}\n")
            summary_file.write(f"Failed CSV:          {args.failed_csv}\n")
            summary_file.write(f"API:                 {args.api}\n")
            summary_file.write(f"Model:               {args.model_name}\n")
            summary_file.write(f"Workers:             {args.workers}\n")
            summary_file.write(f"Total Incorrect Found: {len(incorrect_examples)}\n")
            summary_file.write(f"Successfully Reflected:{success_count}\n")
            summary_file.write(f"Failed Reflection:   {fail_count}\n")
            summary_file.write(f"Total Duration:      {duration:.2f} seconds\n")
            summary_file.write("=" * 78 + "\n")
        logger.info(f"Summary saved to {summary_file_path}")
    except IOError as e:
        logger.error(f"Failed to write summary file {summary_file_path}: {e}")

if __name__ == "__main__":
    main() 