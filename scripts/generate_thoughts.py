import pandas as pd
import sys
import argparse
import os

# --- Add project root to sys.path --- Added ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------

from service.prediction_service import predict_label
from llm.mistral import Mistral
from dotenv import load_dotenv
import json

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Generate Chain-of-Thought augmentations for NLI examples.')
parser.add_argument('--input-csv', type=str, required=True, help='Path to the input CSV file containing premise, hypothesis, label, and id.')
parser.add_argument('--output-json', type=str, required=True, help='Path to the output JSON file where results will be appended.')
parser.add_argument('--model-name', type=str, default='open-mistral-7b', help='Name of the Mistral model to use.')
parser.add_argument('--start-index', type=int, default=0, help='Start processing from this index in the input CSV.')
parser.add_argument('--end-index', type=int, default=None, help='Stop processing at this index (exclusive) in the input CSV.')
args = parser.parse_args()
# --- End Argument Parsing ---


# -- Load data from specified CSV ---
try:
    input_df = pd.read_csv(args.input_csv)
    # Ensure required columns exist
    required_cols = ['id', 'premise', 'hypothesis', 'label']
    if not all(col in input_df.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain columns: {', '.join(required_cols)}")
except FileNotFoundError:
    print(f"Error: Input file not found at {args.input_csv}")
    sys.exit(1)
except ValueError as e:
    print(f"Error reading input CSV: {e}")
    sys.exit(1)

# --- Slice the dataframe based on start/end indices ---
if args.end_index is None:
    args.end_index = len(input_df)
processing_df = input_df.iloc[args.start_index:args.end_index]

# remove 8th row
# train = train.drop(8)
# add ID column
# train['id'] = train.index
# randomly sample 50 rows
# train_sample = train.sample(n=25, random_state=1)

# Define Prompt
json_schema = {
    "thought_process": "<deductive/common-sense reasoning steps>",
    "label": "<0 or 1>"
}

system_prompt = """You are an expert in natural language reasoning and inference. Your task is to analyze pairs of sentences and determine if the second sentence (hypothesis) can be logically inferred from the first sentence (premise).

For each example, I will provide the premise and hypothesis. Your response should be in the following JSON format:
{
    "thought_process": "Step 1. <Identify key information and relationships in the premise, considering logical connections, commonsense understanding, and factual consistency>. Step 2. <Analyze how the hypothesis relates to or contradicts the premise based on the information identified in Step 1. Evaluate if the hypothesis can be reasonably inferred from the premise>. Step 3. <Explain your final reasoning and conclusion on whether the hypothesis is entailed by the premise or not>",
    "label": "<0 for no entailment, 1 for entailment>"
}
Please provide a clear multi-step reasoning chain explaining how you arrived at your final answer, breaking it down into logical components. Ground your response in the given information, logical principles and common-sense reasoning.

Example:
Premise: The dog chased the cat up the tree. Hypothesis: The cat climbed the tree.

Your response:
{
  "thought_process": "Step 1: the premise indicates a scenario where a dog chases a cat, resulting in the cat moving up a tree. The movement 'up the tree' suggests a vertical ascent, typical of climbing behavior. It is common sense that a cat would climb a tree to escape a chasing dog, and there are no known facts that contradict the premise or hypothesis. Step 2: 'The cat climbed the tree' can be logically inferred from the premise because the action of climbing is a reasonable and necessary part of the cat moving 'up the tree' as described. Thus, the hypothesis logically follows from the premise. Step 3: Based on the logical reasoning, common sense, and lack of contradictory facts, the hypothesis can be inferred from the premise.",
  "label": 1
}
"""

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    print("Error: MISTRAL_API_KEY not found in environment variables.")
    sys.exit(1)

print(f"Processing {len(processing_df)} examples from index {args.start_index} to {args.end_index}...")
llm = Mistral(api_key)

# --- Process rows and append to output JSON ---
output_count = 0
correct_count = 0
with open(args.output_json, "a") as outfile: # Open in append mode
    for index, row in processing_df.iterrows():
        print(f"Processing ID: {row['id']} (Index: {index})...")

        response_json = predict_label(
            id=row['id'],
            sys=system_prompt, # System prompt defined above
            premise=row['premise'],
            hypothesis=row['hypothesis'],
            true_label=row['label'],
            llm=llm,
            model_name=args.model_name,
            json_format=json_schema,
            json_filepath=args.output_json # Pass output path for internal logging if needed
        )
        # Append the result to the output file
        if response_json and 'label' in response_json and response_json['label'] != -1: # Check for valid response
             json.dump(response_json, outfile)
             outfile.write('\n')
             output_count += 1
             # Track if prediction is correct
             if response_json['label'] == row['label']:
                 correct_count += 1
        else:
            print(f"Warning: Failed to process ID: {row['id']}. Response: {response_json}")

print(f"Finished processing. Appended {output_count} valid results to {args.output_json}")

# Calculate and print accuracy metrics
if output_count > 0:
    accuracy = (correct_count / output_count) * 100
    print(f"\nResults Summary (Shared Context):")
    print(f"Total examples processed: {output_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save summary to a file
    summary_file_path = f"{args.output_json}_summary.txt"
    with open(summary_file_path, "w") as summary_file:
        summary_file.write(f"Results Summary (Shared Context):\n")
        summary_file.write(f"Total examples processed: {output_count}\n")
        summary_file.write(f"Correct predictions: {correct_count}\n")
        summary_file.write(f"Accuracy: {accuracy:.2f}%\n")
    
    print(f"Summary saved to {summary_file_path}")
else:
    print("No valid predictions were made.")

# train['response_json'].iloc[start_index:end_index] = train.iloc[start_index:end_index].apply(lambda x: predict_label(...), axis=1)

# train.to_csv(f'data/thoughts_{start_index}_{end_index}.csv', index=False)