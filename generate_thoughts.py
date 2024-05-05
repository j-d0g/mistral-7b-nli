import pandas as pd
import sys
from service.prediction_service import predict_label
from llm.mistral import Mistral
import os
from dotenv import load_dotenv

# Get the start and end indices from command-line arguments
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])
dataset = sys.argv[3]

train = pd.read_csv(f'data/training_data/NLI/{dataset}.csv')
# remove 8th row
train = train.drop(8)
# add ID column
train['id'] = train.index
# randomly sample 50 rows
train_sample = train.sample(n=25, random_state=1)

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
  "thought_process": "Step 1: the premise indicates a scenario where a dog chases a cat, resulting in the cat moving up a tree. The movement 'up the tree' suggests a vertical ascent, typical of climbing behavior. It is common sense that a cat would climb a tree to escape a chasing dog, and there are no known facts that contradict the premise or hypothesis. Step 2: 'The cat climbed the tree' can be logically inferred from the premise because the action of climbing is a reasonable and necessary part of the cat moving 'up the tree' as described. Thus, the hypothesis logically follows from the premise. Step 3: Based on the logical reasoning, common sense, and lack of contradictory facts, the hypothesis can be inferred from the premise."
  "label": 1
}
"""

load_dotenv()

# ... (rest of your code remains the same)

train['response_json'].iloc[start_index:end_index] = train.iloc[start_index:end_index].apply(lambda x: predict_label(
    id=x['id'],
    sys=system_prompt,
    premise=x['premise'],
    hypothesis=x['hypothesis'],
    true_label=x['label'],
    llm=Mistral(os.getenv('MISTRAL_API_KEY')),
    model_name='open-mistral-7b',
    json_format=json_schema,
    json_filepath=f'data/thoughts_{start_index}_{end_index}.json'
), axis=1)

train.to_csv(f'data/thoughts_{start_index}_{end_index}.csv', index=False)