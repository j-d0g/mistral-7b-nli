#!/usr/bin/env python3
import json
import argparse
import logging
import os
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

def format_example(premise, hypothesis, thought_process, label):
    """Formats a single example into the SFT training string format"""
    # Create the target JSON object
    target_json_obj = {
        "thought_process": thought_process,
        "predicted_label": label
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

def load_examples(file_path, is_reflection=False):
    """Load examples from a JSONL file"""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                
                # Extract and standardize fields
                # Each example should have premise, hypothesis, thought_process, and a label
                if not all(k in example for k in ['premise', 'hypothesis']):
                    logger.warning(f"Line {line_num}: Missing required fields. Skipping.")
                    continue
                
                # For reflection data, use improved_thought_process
                if is_reflection:
                    if 'improved_thought_process' not in example:
                        logger.warning(f"Line {line_num}: Missing 'improved_thought_process'. Skipping.")
                        continue
                    thought_process = example['improved_thought_process']
                    # Use true_label for reflected examples
                    if 'true_label' not in example:
                        logger.warning(f"Line {line_num}: Missing 'true_label'. Skipping.")
                        continue
                    label = example['true_label']
                else:
                    # For original data, use thought_process
                    if 'thought_process' not in example:
                        logger.warning(f"Line {line_num}: Missing 'thought_process'. Skipping.")
                        continue
                    thought_process = example['thought_process']
                    # Check for different label field names
                    if 'predicted_label' in example:
                        label = example['predicted_label']
                    elif 'label' in example:
                        label = example['label']
                    else:
                        logger.warning(f"Line {line_num}: No label field found. Skipping.")
                        continue
                
                # Ensure label is an integer
                try:
                    label = int(label)
                    if label not in [0, 1]:
                        logger.warning(f"Line {line_num}: Invalid label value {label}. Skipping.")
                        continue
                except ValueError:
                    logger.warning(f"Line {line_num}: Label {label} is not an integer. Skipping.")
                    continue
                
                # Add the standardized example
                examples.append({
                    'id': example.get('id', line_num),
                    'premise': example['premise'],
                    'hypothesis': example['hypothesis'],
                    'thought_process': thought_process,
                    'label': label,
                    'correct': example.get('correct', None)
                })
                
            except json.JSONDecodeError:
                logger.error(f"Line {line_num}: Failed to decode JSON. Skipping.")
                continue
            except Exception as e:
                logger.error(f"Line {line_num}: Unexpected error: {e}. Skipping.")
                continue
    
    return examples

def main():
    parser = argparse.ArgumentParser(description="Prepare NLI CoT data for Ablation 2 (combining correct predictions and reflected thoughts).")
    parser.add_argument("--original-thoughts", type=str, default="data/original_thoughts/sample_thoughts.json", help="Path to the original thoughts JSON file. Defaults to sample_thoughts.json.")
    parser.add_argument("--reflected-thoughts", type=str, default="data/reflected_thoughts/sample_reflections.json", help="Path to the reflected thoughts JSON file. Defaults to sample_reflections.json.")
    parser.add_argument("--output-file", type=str, default="data/finetune/sample_ft.jsonl", help="Path to save the processed JSON Lines file for fine-tuning. Defaults to sample_ablation2.jsonl.")
    
    args = parser.parse_args()
    
    logger.info("Starting Ablation 2 dataset preparation...")
    logger.info(f"Original thoughts file: {args.original_thoughts}")
    logger.info(f"Reflected thoughts file: {args.reflected_thoughts}")
    logger.info(f"Output file: {args.output_file}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Load examples
    logger.info("Loading original thought examples...")
    original_examples = load_examples(args.original_thoughts)
    logger.info(f"Loaded {len(original_examples)} original examples")
    
    logger.info("Loading reflected thought examples...")
    reflected_examples = load_examples(args.reflected_thoughts, is_reflection=True)
    logger.info(f"Loaded {len(reflected_examples)} reflected examples")
    
    # Separate correct examples from original thoughts
    correct_examples = []
    incorrect_examples = []
    for example in original_examples:
        if example['correct'] is True:
            correct_examples.append(example)
        elif example['correct'] is False:
            incorrect_examples.append(example)
        else:
            # If 'correct' is not explicitly set, compute it based on predicted_label vs true_label
            # This should not happen if all examples have 'correct' properly set
            logger.warning(f"Example {example['id']} missing 'correct' field. Skipping.")
            
    logger.info(f"Found {len(correct_examples)} correct examples from original thoughts")
    logger.info(f"Found {len(incorrect_examples)} incorrect examples from original thoughts")
    
    # Combine correct original examples with reflected examples
    combined_examples = correct_examples + reflected_examples
    logger.info(f"Combined dataset has {len(combined_examples)} examples")
    
    # Count labels
    label_counts = Counter(example['label'] for example in combined_examples)
    logger.info(f"Label distribution: {dict(label_counts)}")
    
    # Format examples for fine-tuning
    logger.info("Formatting examples for fine-tuning...")
    formatted_examples = []
    for example in combined_examples:
        formatted_text = format_example(
            example['premise'],
            example['hypothesis'],
            example['thought_process'],
            example['label']
        )
        formatted_examples.append({
            "text": formatted_text,
            "id": example.get('id')
        })
    
    # Write the output file
    logger.info(f"Writing {len(formatted_examples)} examples to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for example in formatted_examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info("Dataset preparation complete!")
    
if __name__ == "__main__":
    main() 