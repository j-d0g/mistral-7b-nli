#!/usr/bin/env python3
import pandas as pd
import sys
import argparse
import os
import json
import logging
import time
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------

# Import relevant functions
from prompts import get_prompt, INITIAL_GENERATION_PROMPT, INITIAL_GENERATION_SCHEMA
from service.prediction_service import predict_label, validate_response

# Configure logging
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs', 'eval'), exist_ok=True)
log_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'eval', 'evaluation.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on NLI test set with Chain-of-Thought reasoning')
    
    # Input and output file paths
    parser.add_argument('--input-csv', type=str, default='data/original_data/test.csv', help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    
    # Model selection and configuration
    parser.add_argument('--model-type', type=str, choices=['hf', 'api'], default='hf', help='Model type: HuggingFace (hf) or API (api)')
    parser.add_argument('--model-path', type=str, help='Path to local model or HF model ID. For base Mistral use: "mistralai/Mistral-7B-v0.3"')
    parser.add_argument('--adapter-path', type=str, help='Path to LoRA adapter for HF models (optional)')
    parser.add_argument('--api-type', type=str, choices=['mistral', 'deepseek'], help='API type if using API model')
    parser.add_argument('--api-model', type=str, help='API model name if using API model')
    
    # Inference parameters
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--max-new-tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
    parser.add_argument('--use-4bit', action='store_true', help='Use 4-bit quantization for HF models')
    parser.add_argument('--use-8bit', action='store_true', help='Use 8-bit quantization for HF models')
    parser.add_argument('--limit', type=int, help='Limit the number of examples to process')
    
    return parser.parse_args()

def initialize_hf_model(model_path, adapter_path=None, use_4bit=False, use_8bit=False):
    """Initialize a HuggingFace model with optional LoRA adapter"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        logger.info(f"Loading HF model: {model_path}")
        
        # Set up quantization if requested
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization")
        elif use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            logger.info("Using 8-bit quantization")
        else:
            quantization_config = None
            logger.info("Using full precision")
        
        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not (use_4bit or use_8bit) else None
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load LoRA adapter if specified
        if adapter_path:
            logger.info(f"Loading LoRA adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        
        # Create a simple LLM class compatible with our predict_label function
        class HFModelWrapper:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
                self.messages = []
            
            def add_messages(self, messages):
                self.messages.extend(messages)
            
            def reset_messages(self):
                self.messages = []
            
            def get_messages(self):
                return self.messages
            
            def prompt_template(self, role, content):
                # For HF models, we just use a simple format
                if role == "system":
                    return {"role": "system", "content": content}
                elif role == "user":
                    return {"role": "user", "content": content}
                elif role == "assistant":
                    return {"role": "assistant", "content": content}
                else:
                    return f"{content}"
            
            def generate_text(self, model_name=None, max_new_tokens=1024, temperature=0.7):
                # Convert messages to a prompt string
                prompt = ""
                for msg in self.messages:
                    if msg["role"] == "system":
                        prompt += f"{msg['content']}\n\n"
                    elif msg["role"] == "user":
                        prompt += f"{msg['content']}\n\n"
                    elif msg["role"] == "assistant":
                        prompt += f"{msg['content']}\n\n"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                    )
                
                response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                return response
        
        return HFModelWrapper(model, tokenizer)
    
    except Exception as e:
        logger.error(f"Error initializing HF model: {str(e)}")
        raise

def initialize_api_model(api_type, api_model):
    """Initialize an API model"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Import the appropriate API client
        if api_type == 'mistral':
            from llm.mistral import Mistral
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables")
            return Mistral(api_key=api_key)
        
        elif api_type == 'deepseek':
            from llm.deepseek_api import DeepSeekAPI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            return DeepSeekAPI(api_key=api_key, model=api_model)
        
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
    
    except Exception as e:
        logger.error(f"Error initializing API model: {str(e)}")
        raise

def evaluate_model(model, model_name, test_df, output_path, is_api=False, limit=None, max_new_tokens=1024):
    """
    Evaluate a model on the test set and save results
    """
    results = []
    predictions = []
    true_labels = []
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Limit the number of examples if specified
    if limit:
        test_df = test_df.head(limit)
    
    # Get the prompt and schema from our prompts module
    prompt, schema = get_prompt('initial_generation')
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {model_name}"):
        try:
            prediction = predict_label(
                id=row.get('id', idx),
                premise=row['premise'],
                hypothesis=row['hypothesis'],
                true_label=row['true_label'],
                llm=model,
                model_name=model_name,
                sys=prompt,
                json_format=schema,
                json_filepath=output_path,
                api=is_api,
                max_retries=3
            )
            
            if prediction and 'predicted_label' in prediction and prediction['predicted_label'] != -1:
                results.append({
                    'id': row.get('id', idx),
                    'premise': row['premise'],
                    'hypothesis': row['hypothesis'],
                    'true_label': row['true_label'],
                    'predicted_label': prediction['predicted_label'],
                    'thought_process': prediction['thought_process'],
                    'correct': prediction['predicted_label'] == row['true_label']
                })
                
                predictions.append(prediction['predicted_label'])
                true_labels.append(row['true_label'])
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1} examples")
            else:
                logger.warning(f"Failed to get prediction for example {idx}")
        
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
    
    # Calculate metrics
    if predictions:
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions),
            'recall': recall_score(true_labels, predictions),
            'f1': f1_score(true_labels, predictions),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
        }
        
        # Save metrics
        metrics_path = output_path.replace('.jsonl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation results for {model_name}:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    return results

def main():
    args = parse_args()
    
    # Load test data
    try:
        test_df = pd.read_csv(args.input_csv)
        logger.info(f"Loaded test data: {len(test_df)} examples")
    except Exception as e:
        logger.error(f"Failed to load test data: {str(e)}")
        return
    
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model name for output files
    if args.model_type == 'hf':
        model_base_name = os.path.basename(args.model_path) if args.model_path else "unknown"
        adapter_suffix = "-" + os.path.basename(args.adapter_path) if args.adapter_path else ""
        model_name = f"{model_base_name}{adapter_suffix}"
        output_path = os.path.join(args.output_dir, f"{model_name}_results.jsonl")
    else:  # API model
        model_name = f"{args.api_type}_{args.api_model}"
        output_path = os.path.join(args.output_dir, f"{model_name}_results.jsonl")
    
    # Initialize the appropriate model
    try:
        if args.model_type == 'hf':
            if not args.model_path:
                logger.error("Model path must be specified for HF models")
                return
            
            model = initialize_hf_model(
                args.model_path,
                args.adapter_path,
                use_4bit=args.use_4bit,
                use_8bit=args.use_8bit
            )
            is_api = False
        
        else:  # API model
            if not args.api_type:
                logger.error("API type must be specified for API models")
                return
            
            model = initialize_api_model(args.api_type, args.api_model)
            is_api = True
        
        logger.info(f"Initialized model: {model_name}")
    
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return
    
    # Evaluate the model
    try:
        results = evaluate_model(
            model=model,
            model_name=model_name,
            test_df=test_df,
            output_path=output_path,
            is_api=is_api,
            limit=args.limit,
            max_new_tokens=args.max_new_tokens
        )
        
        logger.info(f"Evaluation complete, results saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    main() 