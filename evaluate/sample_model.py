#!/usr/bin/env python3
"""
NLI Inference Script for 4-bit Quantized Mistral v0.3 with Chain-of-Thought

This script runs inference on a 4-bit quantized Mistral v0.3 model for an NLI task.
It processes a test set of 1977 samples, optimized for speed with batch size 32.
Features include Chain-of-Thought reasoning, checkpointing, and resuming from interruptions.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# Add project root to sys.path to allow importing prompts
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from prompts import FINETUNE_PROMPT
# For loading LoRA adapters
try:
    from peft import PeftModel, PeftConfig
except ImportError:
    print("PEFT library not found. Fine-tuned models with LoRA adapters may not load correctly.")
    print("You can install it with: pip install peft")

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to run this script")

def parse_args():
    parser = argparse.ArgumentParser(description="Run NLI inference with 4-bit quantized Mistral v0.3")
    parser.add_argument(
        "--model_id", 
        type=str,
        default="mistralai/Mistral-7B-v0.3", 
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="data/original_data/test.csv", 
        help="Path to the test CSV file"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="results/predictions.json", 
        help="Path to save predictions"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use_cot", 
        action="store_true", 
        help="Use Chain-of-Thought reasoning"
    )
    parser.add_argument(
        "--save_every", 
        type=int, 
        default=1, 
        help="Save checkpoint after this many batches"
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="GPU ID to use for inference (if multiple GPUs available)"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None, 
        help="Optional path to save predictions as a single-column CSV"
    )
    return parser.parse_args()

def create_nli_prompt(premise, hypothesis, use_cot=False):
    """Create a prompt for NLI task, with or without chain-of-thought reasoning."""
    if use_cot:
        # Use the fine-tuning prompt from prompts.py
        formatted_prompt = FINETUNE_PROMPT.format(
            premise=premise,
            hypothesis=hypothesis
        )
        # Wrap in Mistral's instruction format
        prompt = f"[INST] {formatted_prompt} [/INST]"
    else:
        # Direct classification prompt
        prompt = f"""[INST] You are an AI assistant trained to classify the relationship between a given premise and hypothesis. Provide your prediction as a JSON object with the key 'label' and the value as either 0 (no entailment) or 1 (entailment).

Premise: '{premise}' 
Hypothesis: {hypothesis} [/INST]"""
    return prompt

def extract_prediction(output_text, use_cot=False):
    """Extract the prediction (0 or 1) from the model's output text using improved extraction logic."""
    output_text = output_text.strip()
    
    # Look for JSON objects - prioritize the one with "predicted_label"
    json_texts = []
    depth = 0
    start_idx = -1
    
    # Find all potential JSON objects
    for i, char in enumerate(output_text):
        if char == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_idx != -1:
                json_texts.append(output_text[start_idx:i+1])
    
    # Try each JSON object, prioritizing those with the expected keys
    for json_text in json_texts:
        try:
            response_dict = json.loads(json_text)
            
            # Check for predicted_label first, then label
            if 'predicted_label' in response_dict:
                label_value = response_dict['predicted_label']
            elif 'label' in response_dict:
                label_value = response_dict['label']
            else:
                continue  # Skip this JSON if it doesn't have either key
            
            # Handle string or number
            if isinstance(label_value, str):
                # For strings, do targeted extraction
                if label_value == "0" or label_value == "no entailment" or label_value.lower() == "false":
                    return 0
                elif label_value == "1" or label_value == "entailment" or label_value.lower() == "true":
                    return 1
                try:
                    return int(label_value)
                except:
                    # If we can't parse directly, continue to the next JSON
                    continue
            else:
                # For numbers, convert to int
                return int(label_value)
        except:
            # If this JSON fails, try the next one
            continue
    
    # If no valid JSON with label found, check for explicit label statements in text
    lower_text = output_text.lower()
    if "final label: 0" in lower_text or "final prediction: 0" in lower_text:
        return 0
    if "final label: 1" in lower_text or "final prediction: 1" in lower_text:
        return 1
        
    # Only if nothing better is found, check for conclusion statements
    if "not entailed" in lower_text or "no entailment" in lower_text:
        return 0
    if "is entailed" in lower_text or "entailment" in lower_text:
        return 1
    
    # For CoT reasoning, try to extract from "step 3" conclusion
    if use_cot and "step 3" in lower_text:
        step3_text = lower_text.split("step 3")[1].split("step 4")[0].split("\n\n")[0]
        if "not entailed" in step3_text or "no entailment" in step3_text:
            return 0
        if "is entailed" in step3_text or "entailment" in step3_text:
            return 1
    
    # Default fallback to the majority class (for NLI datasets, often label 1)
    return 1

def prepare_model_and_tokenizer(model_id, gpu_id=0):
    """Prepare the model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {model_id} on GPU {gpu_id}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Set the device map to use the specified GPU
    device_map = {"": gpu_id}
    
    # Check if model_id is a local path (fine-tuned model) or a HF model ID
    is_local_path = os.path.exists(model_id)
    
    if is_local_path:
        print(f"Loading fine-tuned model from local path: {model_id}")
        # For fine-tuned models with LoRA adapters, use PEFT
        try:
            # First load base model with quantization
            # Get the base model name from adapter config
            adapter_config_path = os.path.join(model_id, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "mistralai/Mistral-7B-v0.3")
                print(f"Found adapter config. Using base model: {base_model_name}")
            else:
                base_model_name = "mistralai/Mistral-7B-v0.3"
                print(f"No adapter config found. Using default base model: {base_model_name}")
            
            # Load base model with quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True
            )
            
            # Load tokenizer from the adapter path (it might have different vocab size)
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, # Use the checkpoint path for the tokenizer
                use_fast=True,
                padding_side="left"
            )

            # Resize token embeddings if tokenizer vocab size differs from base model
            if len(tokenizer) != base_model.config.vocab_size:
                print(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
                base_model.resize_token_embeddings(len(tokenizer))

            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, model_id)
            print("Successfully loaded the model with LoRA adapter")
            
        except Exception as e:
            print(f"Error loading with PEFT: {e}")
            print("Falling back to standard loading...")
            # Fall back to standard loading
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                padding_side="left"
            )
    else:
        # Load the model from Hugging Face with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            padding_side="left"
        )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_checkpoint_path(args):
    """Get the path to the checkpoint file based on the output file."""
    output_dir = os.path.dirname(args.output_file)
    output_filename = os.path.basename(args.output_file)
    checkpoint_filename = f"checkpoint_{output_filename}"
    return os.path.join(output_dir, checkpoint_filename)

def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def save_checkpoint(args, results, current_idx, start_time):
    """Save a checkpoint to resume from later."""
    checkpoint_path = get_checkpoint_path(args)
    
    # Calculate accuracy so far
    processed_samples = len(results)
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    accuracy_so_far = correct_predictions / processed_samples if processed_samples > 0 else 0
    
    # Calculate elapsed time and estimated remaining time
    elapsed_time = time.time() - start_time
    estimated_remaining_time = elapsed_time / current_idx * (args.total_samples - current_idx) if current_idx > 0 else 0
    
    # Create checkpoint data
    checkpoint = {
        'current_idx': current_idx,
        'processed_samples': processed_samples,
        'accuracy_so_far': accuracy_so_far,
        'elapsed_time': elapsed_time,
        'estimated_remaining_time': estimated_remaining_time,
        'results': results
    }
    
    # Convert to JSON serializable format
    json_serializable_checkpoint = convert_to_json_serializable(checkpoint)
    
    # Save checkpoint
    with open(checkpoint_path, 'w') as f:
        json.dump(json_serializable_checkpoint, f)
    
    print("\n--- Checkpoint saved ---")
    print(f"Processed {processed_samples}/{args.total_samples} samples "
          f"({processed_samples/args.total_samples*100:.1f}%)")
    # Only print accuracy if we have any processed samples
    if processed_samples > 0:
        print(f"Current accuracy: {accuracy_so_far:.4f} "
              f"({correct_predictions}/{processed_samples})")
    print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
    print(f"Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
    print(f"Checkpoint saved to: {checkpoint_path}")

def load_checkpoint(args):
    """Load a checkpoint if it exists."""
    checkpoint_path = get_checkpoint_path(args)
    
    if not os.path.exists(checkpoint_path):
        return None, 0, []
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        print(f"Resuming from checkpoint:")
        print(f"Processed {checkpoint['processed_samples']}/{args.total_samples} samples "
              f"({checkpoint['processed_samples']/args.total_samples*100:.1f}%)")
        print(f"Current accuracy: {checkpoint['accuracy_so_far']:.4f}")
        print(f"Elapsed time so far: {checkpoint['elapsed_time']/60:.1f} minutes")
        
        return checkpoint['elapsed_time'], checkpoint['current_idx'], checkpoint['results']
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0, 0, []

def run_inference(model, tokenizer, test_data, args):
    """Run inference on the test data and return predictions."""
    # Set total_samples for checkpoint progress reporting
    args.total_samples = len(test_data)
    
    # Load checkpoint if requested
    elapsed_time, start_idx, results = (0, 0, []) if not args.resume else load_checkpoint(args)
    
    # Start timing
    start_time = time.time() - elapsed_time
    
    # Check if we have labels in the test data
    has_labels = 'label' in test_data.columns
    correct_predictions = sum(1 for r in results if r.get('correct', False)) if has_labels else 0
    
    total_samples = len(test_data)
    batch_count = 0
    
    # Process in batches for efficiency
    for i in tqdm(range(start_idx, total_samples, args.batch_size), desc="Processing batches"):
        batch_data = test_data.iloc[i:min(i+args.batch_size, total_samples)]
        batch_prompts = [
            create_nli_prompt(row['premise'], row['hypothesis'], args.use_cot) 
            for _, row in batch_data.iterrows()
        ]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=args.max_length
        ).to(model.device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                temperature=1.0
            )
        
        # Decode and extract predictions
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Process each sample in the batch
        for j, text in enumerate(generated_texts):
            idx = i + j
            if idx >= total_samples:
                break
                
            # Extract the prediction
            try:
                prediction = extract_prediction(text, args.use_cot)
                
                # Create result object
                result = {
                    'premise': batch_data.iloc[j]['premise'],
                    'hypothesis': batch_data.iloc[j]['hypothesis'],
                    'predicted_label': prediction,
                    'output': text
                }
                
                # If we have labels, add accuracy information
                if has_labels:
                    true_label = batch_data.iloc[j]['label']
                    is_correct = prediction == true_label
                    if is_correct:
                        correct_predictions += 1
                    
                    result['true_label'] = int(true_label)
                    result['correct'] = is_correct
                
                # Save the result
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
        
        # Free up memory
        del inputs, outputs, generated_texts
        torch.cuda.empty_cache()
        
        # Save checkpoint periodically
        batch_count += 1
        if batch_count % args.save_every == 0:
            save_checkpoint(args, results, i + args.batch_size, start_time)
    
    # Calculate final metrics if we have labels
    metrics = {}
    if has_labels and total_samples > 0:
        # Basic accuracy
        accuracy = correct_predictions / total_samples
        
        # Extract true and predicted labels for more detailed metrics
        y_true = [r['true_label'] for r in results]
        y_pred = [r['predicted_label'] for r in results]
        
        # Calculate detailed metrics using scikit-learn
        try:
            # Calculate just the 4 core metrics (macro avg)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro'
            )
            
            # Store just the 4 core metrics
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
            print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        except ImportError:
            print("scikit-learn not available. Only computing basic accuracy.")
            metrics = {'accuracy': float(accuracy)}
        except Exception as e:
            print(f"Error computing detailed metrics: {e}")
            metrics = {'accuracy': float(accuracy)}
    else:
        print("No 'label' column found in data. Skipping metrics calculation.")
    
    # Final elapsed time
    elapsed_time = time.time() - start_time
    
    # Convert results to JSON serializable format
    json_serializable_results = convert_to_json_serializable(results)
    
    # Build the output structure - metadata first
    output = {
        'model': args.model_id,
        'inference_time_seconds': elapsed_time,
        'samples_per_second': total_samples / elapsed_time if elapsed_time > 0 else 0,
        'use_cot': args.use_cot
    }
    
    # Add the 4 core metrics if available
    if metrics:
        output.update(metrics)
    
    # Add results AFTER metrics - this is the large array!
    output['results'] = json_serializable_results
    
    # Add inference statistics AFTER results
    if has_labels:
        output['extraction_stats'] = {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': float(accuracy) if 'accuracy' not in metrics else metrics['accuracy']
        }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Remove checkpoint file if process completed successfully
    checkpoint_path = get_checkpoint_path(args)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return results, metrics, elapsed_time

def main():
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir: # Create directory only if output_file path includes a directory
        os.makedirs(output_dir, exist_ok=True)
        
    # Also ensure output CSV directory exists if specified
    if args.output_csv:
        csv_output_dir = os.path.dirname(args.output_csv)
        if csv_output_dir:
            os.makedirs(csv_output_dir, exist_ok=True)

    # Load test data
    print(f"Loading test data from: {args.test_file}")
    test_data = pd.read_csv(args.test_file)
    print(f"Loaded {len(test_data)} test samples")
    
    # Set the environment variable to use the specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_id, 0)  # Use device 0 since we've set CUDA_VISIBLE_DEVICES
    
    # Run inference
    start_time = time.time()
    results, metrics, elapsed_time = run_inference(model, tokenizer, test_data, args)
    
    print(f"Results saved to: {args.output_file}")
    print(f"Inference completed in {elapsed_time/60:.1f} minutes")
    print(f"Throughput: {len(test_data) / elapsed_time:.2f} samples/second")

    # Display metrics summary
    if metrics and 'accuracy' in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        if 'macro_avg' in metrics:
            print(f"Macro Avg - Precision: {metrics['macro_avg']['precision']:.4f}, "
                  f"Recall: {metrics['macro_avg']['recall']:.4f}, "
                  f"F1: {metrics['macro_avg']['f1_score']:.4f}")
    else:
        print("No metrics calculated (no 'label' column in test data)")

    # Save predictions to CSV if requested
    if args.output_csv:
        try:
            predictions = [item['predicted_label'] for item in results]
            predictions_df = pd.DataFrame({'predicted_label': predictions})
            predictions_df.to_csv(args.output_csv, index=False, header=True)
            print(f"Predictions saved to CSV: {args.output_csv}")
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")

if __name__ == "__main__":
    main() 