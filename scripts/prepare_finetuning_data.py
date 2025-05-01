#!/usr/bin/env python3
import json
import argparse
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
    ]
)
logger = logging.getLogger(__name__)

def format_example(premise, hypothesis, thought_process, predicted_label):
    """Formats a single example into the SFT training string format."""
    
    # Ensure thought_process and predicted_label are properly formatted for JSON embedding
    # json.dumps automatically handles escaping quotes etc.
    target_json_obj = {
        "thought_process": thought_process,
        "predicted_label": predicted_label
    }
    target_json_str = json.dumps(target_json_obj)

    # Construct the prompt
    prompt = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Use chain of thought reasoning to determine if the hypothesis is entailed by the premise. "
        f"Provide your reasoning and the final label (0 or 1) in JSON format: "
        f'{{"thought_process": "...", "predicted_label": ...}}'
    )

    # Combine into the final format using Mistral-like template
    formatted_string = f"<s>[INST] {prompt.strip()} [/INST] {target_json_str} </s>"
    
    return formatted_string

def main():
    parser = argparse.ArgumentParser(description="Prepare NLI CoT data for Supervised Fine-Tuning (SFT).")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input JSON Lines file (e.g., train_thoughts.json).")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the processed JSON Lines file.")
    parser.add_argument("--filter-correct", action="store_true", help="If set, only include examples where 'correct' is true.")
    parser.add_argument("--label-field", type=str, default="prediction", choices=["prediction", "true_label"], help="Field to use for the target label in the output JSON ('prediction' or 'true_label').")

    args = parser.parse_args()

    logger.info(f"Starting data preparation...")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Filter correct examples: {args.filter_correct}")
    logger.info(f"Using label field: {args.label_field}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(args.output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    example = json.loads(line.strip())

                    # --- Validation and Filtering --- 
                    required_keys = ['premise', 'hypothesis']
                    if args.filter_correct:
                        required_keys.append('correct')

                    # Extract the label value, prioritizing standard naming conventions
                    label_value = None
                    label_field_used = None
                    
                    # First try to use the specified label field from the arguments
                    if args.label_field == 'prediction' and 'prediction' in example:
                        label_value = example['prediction']
                        label_field_used = 'prediction'
                    # Then try our standardized field names in order of preference
                    elif 'predicted_label' in example:
                        label_value = example['predicted_label']
                        label_field_used = 'predicted_label'
                    elif 'label' in example:
                        label_value = example['label']
                        label_field_used = 'label'
                    elif 'true_label' in example:
                        label_value = example['true_label']
                        label_field_used = 'true_label'
                        
                    if label_value is None:
                        logger.warning(f"Line {line_num}: Missing required label field. Skipping.")
                        skipped_count += 1
                        continue

                    # Skip examples where the chosen label is invalid (e.g., -1 for failed generations)
                    if label_value not in [0, 1]:
                        logger.warning(f"Line {line_num}: Invalid label value '{label_value}' in field '{label_field_used}'. Skipping.")
                        skipped_count += 1
                        continue
                        
                    # --- Thought Process Extraction --- 
                    # Use the appropriate thought_process field depending on what's available, prioritizing standard naming
                    thought_process = None
                    thought_field_used = None
                    
                    # Choose the thought process source in order of preference
                    if 'improved_thought_process' in example:
                        thought_process = example['improved_thought_process']
                        thought_field_used = 'improved_thought_process'
                    elif 'thought_process' in example:
                        thought_process = example['thought_process']
                        thought_field_used = 'thought_process' 
                    elif 'initial_thought' in example:
                        thought_process = example['initial_thought']
                        thought_field_used = 'initial_thought'
                        
                    if thought_process is None:
                        logger.warning(f"Line {line_num}: No valid thought process field found. Skipping.")
                        skipped_count += 1
                        continue
                        
                    # If we've reached here, we have valid data to format
                    premise = example['premise']
                    hypothesis = example['hypothesis']
                    predicted_label = int(label_value)
                    
                    logger.debug(f"Line {line_num}: Using '{thought_field_used}' for thought process and '{label_field_used}' for label")
                    
                    formatted_text = format_example(premise, hypothesis, thought_process, predicted_label)
                    
                    # Write as a JSON object with a "text" key
                    output_record = {"text": formatted_text}
                    outfile.write(json.dumps(output_record) + '\n')
                    processed_count += 1

                except json.JSONDecodeError:
                    logger.error(f"Line {line_num}: Failed to decode JSON. Skipping.")
                    error_count += 1
                except Exception as e:
                    logger.error(f"Line {line_num}: Unexpected error processing line: {e}")
                    error_count += 1
                    
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {args.input_file}")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during file processing: {e}")
        return

    logger.info(f"Data preparation finished.")
    logger.info(f"Processed examples: {processed_count}")
    logger.info(f"Skipped examples: {skipped_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main() 