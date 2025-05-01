#!/usr/bin/env python3
"""
NLI Inference Script for 4-bit Quantized Mistral v0.3

This script runs inference on a 4-bit quantized Mistral v0.3 model for an NLI task.
It processes a test set of 1977 samples, optimized for speed on an NVIDIA RTX 4090.
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
        default=8, 
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=2048, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--use_cot", 
        action="store_true", 
        help="Use Chain-of-Thought reasoning"
    )
    return parser.parse_args()

def create_nli_prompt(premise, hypothesis, use_cot=False):
    """Create a prompt for NLI task, with or without chain-of-thought reasoning."""
    if use_cot:
        # Chain-of-Thought prompt
        prompt = f"""[INST] You are an expert in natural language reasoning and inference. Your task is to analyze pairs of sentences and determine if the second sentence (hypothesis) can be logically inferred from the first sentence (premise). For each example, I will provide the premise and hypothesis. Your response should be in the following JSON format:
{{
  "thought_process": "Step 1. <Identify key information and relationships in the premise, considering logical connections, commonsense understanding, and factual consistency>. Step 2. <Analyze how the hypothesis relates to or contradicts the premise based on the information identified in Step 1. Evaluate if the hypothesis can be reasonably inferred from the premise>. Step 3. <Explain your final reasoning and conclusion on whether the hypothesis is entailed by the premise or not>",
  "label": "<0 for no entailment, 1 for entailment>"
}}
Please provide a clear multi-step reasoning chain explaining how you arrived at your final answer, breaking it down into logical components. Ground your response in the given information, logical principles and common-sense reasoning.

Premise: '{premise}' Hypothesis: {hypothesis}. [/INST]"""
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
            
            # Extract the label from the response
            if use_cot and 'label' in response_dict:
                return int(response_dict['label'])
            elif not use_cot and 'label' in response_dict:
                return int(response_dict['label'])
        
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

def prepare_model_and_tokenizer(model_id):
    """Prepare the model and tokenizer with 4-bit quantization."""
    print(f"Loading model: {model_id}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
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

def run_inference(model, tokenizer, test_data, args):
    """Run inference on the test data and return predictions."""
    results = []
    correct_predictions = 0
    total_samples = len(test_data)
    
    # Process in batches for efficiency
    for i in tqdm(range(0, total_samples, args.batch_size), desc="Processing batches"):
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
    
    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    
    return results, accuracy

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load test data
    print(f"Loading test data from: {args.test_file}")
    test_data = pd.read_csv(args.test_file)
    print(f"Loaded {len(test_data)} test samples")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_id)
    
    # Run inference
    start_time = time.time()
    results, accuracy = run_inference(model, tokenizer, test_data, args)
    inference_time = time.time() - start_time
    
    # Save results
    output = {
        'model': args.model_id,
        'accuracy': accuracy,
        'inference_time_seconds': inference_time,
        'samples_per_second': len(test_data) / inference_time,
        'use_cot': args.use_cot,
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {args.output_file}")
    print(f"Inference completed in {inference_time:.2f} seconds")
    print(f"Throughput: {len(test_data) / inference_time:.2f} samples/second")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 