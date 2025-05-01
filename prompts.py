import json

# --- Initial Thought Generation ---

INITIAL_GENERATION_SCHEMA = {
    "thought_process": "Step 1. <Analyse the premise carefully for logical, factual and common sense information>. Step 2. <Analyze the hypothesis, identifying any links, (in)consistencies or potential contradictions between the hypothesis and the premise relating to key information from Step 1. Question whether the hypothesis could be reasonably inferred from the premise>. Step 3. <Based on your reasoning, conclude if the hypothesis is entailed by the premise>",
    "predicted_label": "<0 for no entailment, 1 for entailment>"
}

INITIAL_GENERATION_PROMPT = f"""You are an expert at using chain of thought reasoning to solve pairwise binary classification NLI tasks. Given a premise and hypothesis, your task is to use chain of thought reasoning to determine whether the hypothesis can be logically inferred from the premise. You will give your reasoning and final answer in valid JSON format.

# Instructions:
## Response Format / Schema:
For each example, I will provide the premise and hypothesis. Your response **MUST** be in the following JSON format:

{json.dumps(INITIAL_GENERATION_SCHEMA, indent=4)}

## Rules & Guidelines:
- ALWAYS show your reasoning BEFORE giving your final answer. 
- Ensure your reasoning and final answer are grounded in the given information. 
- Use logical principles and common-sense reasoning to support your answer.
- Keep your reasoning short and concise. Longer reasoning will be penalized.
- Your final answer should be a single number, either 0 for no entailment or 1 for entailment.

# Examples

<input>
Premise: The dog chased the cat up the tree. Hypothesis: The cat climbed the tree.
</input>
<response>
{{
  "thought_process": "Step 1: The dog is chasing the cat. \"Up the tree\" suggests the location/direction of the chase: toward the top of the tree. For a chase to happen up the tree, the cat must be moving up the tree, i.e, the cat must climb. Step 2: Is it possible for the dog to chase the cat up the tree if the cat doesn't climb? No, the cat must climb if the chase moves up the tree. Step 3: Based on the logical reasoning, common sense, and lack of contradictory facts, the hypothesis can be inferred from the premise.",
  "predicted_label": 1
}}
</response>
"""

# --- Scoring and Improving Thoughts ---

SCORING_SCHEMA = {
    "analysis": "<Analysis of thought-chain's usefulness, relevance, and logical coherence>",
    "score": "<A score integer between 1 and 5 inclusive>",
    "improved_thought_process": "Step 1. <Identify key information and relationships in the premise...>. Step 2. <Analyze hypothesis relation to premise...>. Step 3. <Explain final reasoning...>",
    "predicted_label": "<0 for no-entailment, 1 for entailment>"
}

SCORING_PROMPT = f"""You are an expert in natural language reasoning and inference tasked with evaluating NLI reasoning chains.

# Task
Evaluate the chain-of-thought reasoning for an NLI binary classification task (premise-hypothesis pairs).

# Input Format
- Premise: The initial statement to be analyzed
- Hypothesis: The statement to be evaluated against the premise
- Original thought process: The reasoning chain used to determine entailment

# Evaluation Criteria
Score each thought process on a scale of 1-5:
- 5: Excellent - Logically sound, relevant, concise reasoning leading to correct conclusion
- 4: Good - Mostly sound reasoning with minor flaws
- 3: Adequate - Contains useful reasoning but significant room for improvement
- 2: Poor - Major logical flaws or irrelevant reasoning
- 1: Unacceptable - Completely flawed or incoherent reasoning

# Response Requirements
1. Provide a brief analysis of the thought chain's strengths and weaknesses
2. Assign a numerical score (1-5)
3. Create an improved version of the thought chain
4. Include the correct entailment label (0=no entailment, 1=entailment)

Important:
- Shorter, more concise thoughts are preferred
- Any thought with clear room for improvement should score 3 or lower

# Output Format
Your response **MUST** be in the following JSON format:
{json.dumps(SCORING_SCHEMA, indent=4)}
"""


# --- Reflection on Initial Reasoning vs True Label ---

REFLECTION_SCHEMA = {
    "error_analysis": "<Brief analysis of logical errors or missing insights in the initial reasoning>",
    "improved_thought_process": "Step 1. <...> Step 2. <...> Step 3. <Logical steps leading to the true label>",
    "predicted_label": "<The provided true_label (0 or 1)>"
}

# This prompt template requires .format(premise=..., hypothesis=..., thought_process=..., predicted_label=..., true_label=..., schema_string=...)
REFLECTION_PROMPT_TEMPLATE = """You are an expert in logical reasoning for NLI tasks. Your goal is to reflect on an initial chain-of-thought reasoning that led to a prediction, identify where that reasoning went wrong (if applicable), and generate improved reasoning that aligns with the true label.

**Input:**
Premise: {premise}
Hypothesis: {hypothesis}
Initial Thought Process: {thought_process}
Initial Prediction: {predicted_label}
True Label: {true_label}

**Instructions:**
1. Compare the initial prediction with the true label
2. If they differ, identify specific logical errors, misinterpretations, or missing insights in the initial reasoning
3. If they match but the reasoning could be improved, identify how
4. Create a new chain-of-thought reasoning that addresses these issues and naturally leads to the true label
5. Keep the improved reasoning concise and focused

**Output Format:**
Your response **MUST** be in the following JSON format:
{schema_string}

**Guidelines:**
- Focus on logical errors rather than just restating the correct answer
- Maintain the same logical structure (step 1, step 2, etc.) as the initial thought process when possible
- The error analysis should be specific enough to help model learning
- The new reasoning should demonstrate better logical clarity than the original
"""

# Function to get specific prompt (optional helper)
def get_prompt(prompt_type):
    if prompt_type == 'initial_generation':
        return INITIAL_GENERATION_PROMPT, INITIAL_GENERATION_SCHEMA
    elif prompt_type == 'scoring':
        return SCORING_PROMPT, SCORING_SCHEMA
    elif prompt_type == 'reflection':
        # Note: The template needs formatting with premise, hypothesis, thought_process, predicted_label, true_label, schema_string
        return REFLECTION_PROMPT_TEMPLATE, REFLECTION_SCHEMA # Return the schema separately
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}") 