"""
NLI Transformer Demo - Approach C

This script contains all the necessary code for running inference with the Mistral-7B model for NLI.
You can copy sections of this code into notebook cells.

Usage:
1. Start Jupyter in Docker: ./run_notebook.sh
2. Create a new notebook
3. Copy each section below into separate cells
"""

# ============================
# Cell 1: Imports
# ============================
"""
# Standard library imports
import os
import sys
import json
import time

# Third-party imports
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.notebook import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dotenv import load_dotenv

# Add project root to import path
sys.path.append(os.path.dirname(os.getcwd()))

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("⚠️ Warning: HF_TOKEN not found in environment variables.")
else:
    print("✅ HF_TOKEN found in environment variables.")
"""

# ============================
# Cell 2: Check CUDA Availability
# ============================
"""
if torch.cuda.is_available():
    print(f"✅ CUDA is available with {torch.cuda.device_count()} devices")
    for i in range(torch.cuda.device_count()):
        print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"  - Current device: {torch.cuda.current_device()}")
else:
    print("❌ CUDA is not available. This notebook requires GPU acceleration.")
"""

# ============================
# Cell 3: Import Prompt Template
# ============================
"""
# Try to import from utils module
try:
    from utils.prompts import FINETUNE_PROMPT
    print("✅ Successfully imported prompt template from utils.prompts")
except ImportError:
    # Fallback if import fails
    FINETUNE_PROMPT = \"\"\"\\
You are tasked with Natural Language Inference (NLI). Given a premise and a hypothesis, determine if the hypothesis is entailed by the premise.

Premise: {premise}
Hypothesis: {hypothesis}

Please work through this step-by-step:

Step 1: Analyze the premise and identify its key facts and implications.
Step 2: Analyze the hypothesis and identify what it claims.
Step 3: Determine if the hypothesis necessarily follows from the premise.

If the hypothesis is entailed by the premise, the label is 1.
If the hypothesis is not entailed by the premise, the label is 0.

Your final answer should be in this JSON format:
{{
    "step_1": "your analysis of the premise",
    "step_2": "your analysis of the hypothesis",
    "step_3": "your reasoning about entailment",
    "predicted_label": 0 or 1
}}
\"\"\"
    print("⚠️ Could not import prompt from utils.prompts, using fallback template.")

print("\nPrompt template:\n")
print(FINETUNE_PROMPT)
"""

# ============================
# Cell 4: Prompt Creation and Prediction Extraction
# ============================
"""
def create_nli_prompt(premise, hypothesis):
    \"\"\"Create the standard NLI prompt using the fine-tuning format.\"\"\"
    # Use the fine-tuning prompt template
    formatted_prompt = FINETUNE_PROMPT.format(
        premise=premise,
        hypothesis=hypothesis
    )
    # Wrap in Mistral's instruction format
    prompt = f"[INST] {formatted_prompt} [/INST]"
    return prompt

def extract_prediction(output_text):
    \"\"\"Extract the prediction (0 or 1) from the model's output text.\"\"\"
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
    
    # Default fallback
    return 1  # Most NLI datasets have label 1 as majority class
"""

# ============================
# Cell 5: Model Loading Function
# ============================
"""
def load_model(model_id, gpu_id=0):
    \"\"\"Load the model and tokenizer with 4-bit quantization.\"\"\"
    print(f"Loading model: {model_id} on GPU {gpu_id}")
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Set the device map to use the specified GPU
    device_map = {"":gpu_id}
    
    # Check if model_id is a local path or a HF model ID
    is_local_path = os.path.exists(model_id)
    
    if is_local_path:
        print(f"Loading from local path: {model_id}")
        # Check if this is a LoRA adapter
        adapter_config_path = os.path.join(model_id, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                # Import PEFT for LoRA
                from peft import PeftModel, PeftConfig
                
                # First, get the base model from the adapter config
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "mistralai/Mistral-7B-v0.3")
                print(f"Found adapter config. Using base model: {base_model_name}")
                
                # Load base model with quantization
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
                
                # Load tokenizer from the adapter path
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_fast=True,
                    padding_side="left",
                    token=HF_TOKEN
                )
                
                # Resize token embeddings if needed
                if len(tokenizer) != base_model.config.vocab_size:
                    print(f"Resizing token embeddings from {base_model.config.vocab_size} to {len(tokenizer)}")
                    base_model.resize_token_embeddings(len(tokenizer))
                
                # Load LoRA adapter
                model = PeftModel.from_pretrained(base_model, model_id)
                print("Successfully loaded with LoRA adapter")
            except Exception as e:
                print(f"Error loading with PEFT: {e}")
                raise e
        else:
            # Standard local model loading
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                token=HF_TOKEN
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                padding_side="left",
                token=HF_TOKEN
            )
    else:
        # Loading from Hugging Face Hub
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            padding_side="left",
            token=HF_TOKEN
        )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer
"""

# ============================
# Cell 6: Inference Function
# ============================
"""
def run_inference(model, tokenizer, test_data, batch_size=8, max_length=512):
    \"\"\"Run inference on test data and return predictions.\"\"\"
    # Check if we have labels in the test data
    has_labels = 'label' in test_data.columns
    
    total_samples = len(test_data)
    results = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch_data = test_data.iloc[i:min(i+batch_size, total_samples)]
        batch_prompts = [
            create_nli_prompt(row['premise'], row['hypothesis'])
            for _, row in batch_data.iterrows()
        ]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(model.device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
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
                prediction = extract_prediction(text)
                
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
                    result['true_label'] = int(true_label)
                    result['correct'] = is_correct
                
                # Save the result
                results.append(result)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
        
        # Free up memory
        del inputs, outputs, generated_texts
        torch.cuda.empty_cache()
    
    # Calculate metrics if labels are available
    if has_labels:
        y_true = [r['true_label'] for r in results]
        y_pred = [r['predicted_label'] for r in results]
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        metrics = None
        print("No 'label' column found. Skipping metrics calculation.")
    
    return results, metrics
"""

# ============================
# Cell 7: Load Sample Dataset
# ============================
"""
# You can change this to any test file
test_file = "data/original_data/sample.csv"

# Load the test data
try:
    test_data = pd.read_csv(test_file)
    print(f"✅ Successfully loaded {len(test_data)} samples from {test_file}")
    display(test_data.head())
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    # Create a minimal sample if file not found
    test_data = pd.DataFrame({
        'premise': [
            "The cat is sleeping on the mat.",
            "All birds can fly."
        ],
        'hypothesis': [
            "There is a cat on the mat.",
            "Penguins can fly."
        ],
        'label': [1, 0]  # entailment, no entailment
    })
    print("⚠️ Using fallback sample data instead.")
    display(test_data)
"""

# ============================
# Cell 8: Load Model
# ============================
"""
# You can change this to any model path or HF model ID
model_path = "models/mistral-7b-nli-cot"

try:
    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)
    print(f"✅ Successfully loaded model from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nCheck if the model path exists and try again.")
    # Raise the exception to stop execution
    raise e
"""

# ============================
# Cell 9: Run Inference
# ============================
"""
# Run inference on the test data
batch_size = 4  # Smaller batch size for demonstration
max_length = 512

# Use a small subset for demo purposes (optional)
demo_size = min(10, len(test_data))
demo_data = test_data.iloc[:demo_size].copy()

print(f"Running inference on {len(demo_data)} samples with batch size {batch_size}...")
results, metrics = run_inference(model, tokenizer, demo_data, batch_size, max_length)
"""

# ============================
# Cell 10: Examine Results
# ============================
"""
# Create a DataFrame with results
results_df = pd.DataFrame([
    {
        'premise': r['premise'],
        'hypothesis': r['hypothesis'],
        'predicted_label': r['predicted_label'],
        'true_label': r.get('true_label', 'Unknown'),
        'correct': r.get('correct', 'Unknown')
    }
    for r in results
])

display(results_df)
"""

# ============================
# Cell 11: Detailed Example Analysis
# ============================
"""
# Select an example to analyze in detail
example_idx = 0  # Change this to analyze different examples

if results:
    example = results[example_idx]
    
    print("=== Example Analysis ===\n")
    print(f"Premise: {example['premise']}")
    print(f"Hypothesis: {example['hypothesis']}")
    print(f"Predicted Label: {example['predicted_label']} ({'entailment' if example['predicted_label'] == 1 else 'no entailment'})")
    
    if 'true_label' in example:
        print(f"True Label: {example['true_label']} ({'entailment' if example['true_label'] == 1 else 'no entailment'})")
        print(f"Correct: {'✓' if example['correct'] else '✗'}")
    
    print("\n=== Model Output (Chain-of-Thought) ===\n")
    print(example['output'])
"""

# ============================
# Cell 12: Save Predictions to CSV
# ============================
"""
# Save predictions to CSV
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Generate a filename based on model and data
model_name = os.path.basename(model_path.rstrip('/'))
data_name = os.path.splitext(os.path.basename(test_file))[0]
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_csv = f"{output_dir}/{model_name}-{data_name}-{timestamp}.csv"

# Extract just the predictions
predictions = [r['predicted_label'] for r in results]
predictions_df = pd.DataFrame({'prediction': predictions})

# Save to CSV
predictions_df.to_csv(output_csv, index=False)
print(f"✅ Predictions saved to: {output_csv}")
"""

# ============================
# Cell 13: Full Test Set Function
# ============================
"""
def run_on_full_test(model_path, test_file_path, batch_size=8, max_length=512):
    \"\"\"
    Run inference on the full test set and save results.
    \"\"\"
    print(f"Loading test data from {test_file_path}...")
    full_test_data = pd.read_csv(test_file_path)
    print(f"Loaded {len(full_test_data)} samples.")
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path)
    
    print("Running inference...")
    results, metrics = run_inference(model, tokenizer, full_test_data, batch_size, max_length)
    
    # Generate output filenames
    model_name = os.path.basename(model_path.rstrip('/'))
    data_name = os.path.splitext(os.path.basename(test_file_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save full results (with Chain-of-Thought)
    output_json = f"results/{model_name}-{data_name}-{timestamp}.json"
    with open(output_json, 'w') as f:
        json.dump({
            'model': model_path,
            'metrics': metrics,
            'results': results
        }, f, indent=2)
    print(f"Full results saved to: {output_json}")
    
    # Save predictions only (for submission)
    output_csv = f"results/{model_name}-{data_name}-{timestamp}.csv"
    predictions = [r['predicted_label'] for r in results]
    predictions_df = pd.DataFrame({'prediction': predictions})
    predictions_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")
    
    return output_csv, metrics

# Uncomment the following line to run on the full test set
# output_csv, metrics = run_on_full_test("models/mistral-7b-nli-cot", "data/original_data/test.csv", batch_size=8)
""" 