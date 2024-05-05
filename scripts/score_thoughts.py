import sys

root = ''

import pandas as pd
import glob

train_paths = sorted(glob.glob(root + 'data/json_data/*.json'))
dev_paths = sorted(glob.glob(root + 'data/json_data_dev/*.json'))


def read_data(paths: list[str]):
    dfs = []
    for file in paths:
        df = pd.read_json(file, lines=True)
        df['thoughts_len'] = df['thoughts'].apply(len)
        dfs.append(df)
    return pd.concat(dfs)


train = read_data(train_paths)
train.drop_duplicates(subset=['id'], inplace=True)
train.sort_values(by='id', inplace=True)
train.reset_index(drop=True, inplace=True)

dev = read_data(dev_paths)
dev.drop_duplicates(subset=['id'], inplace=True)
dev.sort_values(by='id', inplace=True)
dev.reset_index(drop=True, inplace=True)

import os
from dotenv import load_dotenv

from llm.mistral import Mistral
from service.scoring_service import generate_score

load_dotenv()
API_KEY = os.getenv('MISTRAL_API_KEY')

import json

json_schema = {
    "analysis": " <Analysis of thought-chains usefulness, relevance, and logical coherence>",
    "score": "<A score integer between 1 and 5 inclusive>",
    "improved_thoughts": "Step 1. <Identify key information and relationships in the premise, considering logical connections, commonsense understanding, and factual consistency>. Step 2. <Analyze how the hypothesis relates to or contradicts the premise based on the information identified in Step 1. Evaluate if the hypothesis can be reasonably inferred from the premise>. Step 3. <Explain your final reasoning and conclusion on whether the hypothesis is entailed by the premise or not>",
    "label": "<0 for no-entailment, 1 for entailment>"
}

system_prompt = f"""You are an expert in natural language reasoning and inference. I will give you the predictions and chain-of-thought reasoning steps for an NLI binary-classification task involving premise, hypothesis, and whether the hypothesis entails the premise. Your goal is to score the intermediate thought-process used to generate these predictions. Your score should be between 1-5 inclusive(with 5 being the best, 3 indicating room for improvement and 1 being the worst) based on the thought-chains' logical coherence, relevance to the premise and hypothesis, use of common-sense reasoning, and effectiveness in determining the entailment relationship. Provide a brief analysis/justification before generating each score, followed by a revised and improved version of the thought chains. The shorter the thought-chains, the better. Thoughts with clear room for improvement should be scored 3 or lower.

 Your response should be in the following JSON format: {json.dumps(json_schema)}

Keep your thoughts and analysis concise.
"""

generate_prompt = lambda p, h, l, t: f"""
Premise: {p}
Hypothesis: {h}
Thought-Process: {t}
Predicted Label: {l}
"""

dataset = sys.argv[3]

if dataset == 'train':
    start_index = train[train['id'] == int(sys.argv[1])].index[0]
    end_index = train[train['id'] == int(sys.argv[2])].index[0]
    train_sample = train.iloc[start_index:end_index]

    train_sample['score_json'] = train_sample.apply(lambda x: generate_score(
        id=x['id'],
        sys=system_prompt,
        premise=x['premise'],
        hypothesis=x['hypothesis'],
        thoughts=x['thoughts'],
        predicted_label=x['prediction'],
        true_label=x['true_label'],
        llm=Mistral(os.getenv('MISTRAL_API_KEY')),
        model_name='open-mixtral-8x7b',
        json_format=json_schema,
        json_filepath=f'{root}data/json_data/score'
    ), axis=1)

elif dataset == 'dev':
    start_index = dev[dev['id'] == int(sys.argv[1])].index[0]
    end_index = dev[train['id'] == int(sys.argv[2])].index[0]
    dev_sample = dev.iloc[start_index:end_index]

    dev_sample['score_json'] = dev_sample.apply(lambda x: generate_score(
        id=x['id'],
        sys=system_prompt,
        premise=x['premise'],
        hypothesis=x['hypothesis'],
        thoughts=x['thoughts'],
        predicted_label=x['prediction'],
        true_label=x['true_label'],
        llm=Mistral(os.getenv('MISTRAL_API_KEY')),
        model_name='open-mixtral-8x7b',
        json_format=json_schema,
        json_filepath=f'{root}data/json_data_dev/score'
    ), axis=1)
