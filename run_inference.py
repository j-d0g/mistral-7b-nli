#!/usr/bin/env python3
"""
NLI Inference Script for 4-bit Quantized Mistral v0.3 with Chain-of-Thought

This script runs inference on a 4-bit quantized Mistral v0.3 model for an NLI task.
It processes a test set of 1977 samples, optimized for speed with batch size 32.
Features include Chain-of-Thought reasoning, checkpointing, and resuming from interruptions.
"""

import os
import json
import time
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
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
    """Extract the prediction (0 or 1) from the model's output text."""
    try:
        # Try to find and parse a JSON object in the output
        output_text = output_text.strip()
        
        # Find json-like content
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_text = output_text[start_idx:end_idx]
            response_dict = json.loads(json_text)
            
            # Extract the label from the response, checking all possible field names
            if 'predicted_label' in response_dict:
                # This is the primary field name for Chain-of-Thought results
                label_value = response_dict['predicted_label']
            elif 'label' in response_dict:
                # This is a fallback field name
                label_value = response_dict['label']
            else:
                raise ValueError("No label field found in JSON")
                
            # Convert label to integer if it's a string
            if isinstance(label_value, str):
                # Try to extract a number from the string
                if '0' in label_value and '1' not in label_value:
                    return 0
                elif '1' in label_value and '0' not in label_value:
                    return 1
                else:
                    # Try to convert directly
                    return int(label_value)
            else:
                # Already a number
                return int(label_value)
        
        # If we couldn't parse JSON or find the label, check for direct "0" or "1" in the text
        if '0' in output_text and '1' not in output_text:
            return 0
        elif '1' in output_text and '0' not in output_text:
            return 1
    except Exception as e:
        print(f"Error extracting prediction: {e}")
        print(f"Output text: {output_text}")
    
    # Default fallback - predict the majority class (analyze your dataset to determine this)
    return 1  # Assuming 1 is the majority class, adjust if needed

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
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, model_id)
            print("Successfully loaded the model with LoRA adapter")
            
            # Load tokenizer from the same path
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                padding_side="left"
            )
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

def save_checkpoint(args, results, current_idx, start_time):
    """Save the current progress to a checkpoint file."""
    checkpoint_path = get_checkpoint_path(args)
    
    # Calculate current statistics
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # Count correct predictions
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    processed_samples = len(results)
    
    # Create a JSON-compatible version of results
    # Convert all NumPy types to Python native types
    json_compatible_results = []
    for result in results:
        cleaned_result = {}
        for k, v in result.items():
            # Convert numpy types to native Python types
            if isinstance(v, np.integer):
                cleaned_result[k] = int(v)
            elif isinstance(v, np.floating):
                cleaned_result[k] = float(v)
            elif isinstance(v, np.ndarray):
                cleaned_result[k] = v.tolist()
            elif isinstance(v, np.bool_):
                cleaned_result[k] = bool(v)
            else:
                cleaned_result[k] = v
        json_compatible_results.append(cleaned_result)
    
    checkpoint = {
        'model': args.model_id,
        'current_idx': current_idx,
        'elapsed_time': elapsed_time,
        'processed_samples': processed_samples,
        'correct_predictions': correct_predictions,
        'accuracy_so_far': correct_predictions / processed_samples if processed_samples > 0 else 0,
        'use_cot': args.use_cot,
        'results': json_compatible_results
    }
    
    temp_path = f"{checkpoint_path}.temp"
    with open(temp_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Rename to final path to ensure atomic operation
    os.replace(temp_path, checkpoint_path)
    
    # Calculate and print progress statistics
    samples_per_second = processed_samples / elapsed_time if elapsed_time > 0 else 0
    remaining_samples = args.total_samples - current_idx
    estimated_remaining_time = remaining_samples / samples_per_second if samples_per_second > 0 else 0
    
    print(f"\n--- Checkpoint saved ---")
    print(f"Processed {processed_samples}/{args.total_samples} samples "
          f"({processed_samples/args.total_samples*100:.1f}%)")
    print(f"Current accuracy: {correct_predictions/processed_samples:.4f} "
          f"({correct_predictions}/{processed_samples})")
    print(f"Speed: {samples_per_second:.2f} samples/second")
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

def run_inference(model, tokenizer, test_data, args):
    """Run inference on the test data and return predictions."""
    # Set total_samples for checkpoint progress reporting
    args.total_samples = len(test_data)
    
    # Load checkpoint if requested
    elapsed_time, start_idx, results = (0, 0, []) if not args.resume else load_checkpoint(args)
    
    # Start timing
    start_time = time.time() - elapsed_time
    correct_predictions = sum(1 for r in results if r.get('correct', False))
    
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
                true_label = batch_data.iloc[j]['label']
                
                # Check if the prediction is correct
                is_correct = prediction == true_label
                if is_correct:
                    correct_predictions += 1
                
                # Save the result
                results.append({
                    'premise': batch_data.iloc[j]['premise'],
                    'hypothesis': batch_data.iloc[j]['hypothesis'],
                    'true_label': int(true_label),
                    'predicted_label': prediction,
                    'correct': is_correct,
                    'output': text
                })
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
        
        # Free up memory
        del inputs, outputs, generated_texts
        torch.cuda.empty_cache()
        
        # Save checkpoint periodically
        batch_count += 1
        if batch_count % args.save_every == 0:
            save_checkpoint(args, results, i + args.batch_size, start_time)
    
    # Calculate final accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    # Final elapsed time
    elapsed_time = time.time() - start_time
    
    # Convert results to JSON serializable format
    json_serializable_results = convert_to_json_serializable(results)
    
    # Save final results
    output = {
        'model': args.model_id,
        'accuracy': accuracy,
        'inference_time_seconds': elapsed_time,
        'samples_per_second': total_samples / elapsed_time if elapsed_time > 0 else 0,
        'use_cot': args.use_cot,
        'results': json_serializable_results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Remove checkpoint file if process completed successfully
    checkpoint_path = get_checkpoint_path(args)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return results, accuracy, elapsed_time

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
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
    results, accuracy, elapsed_time = run_inference(model, tokenizer, test_data, args)
    
    print(f"Results saved to: {args.output_file}")
    print(f"Inference completed in {elapsed_time/60:.1f} minutes")
    print(f"Throughput: {len(test_data) / elapsed_time:.2f} samples/second")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 