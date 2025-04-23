import json
import re
from pprint import pprint

from llm.base_llm import BaseLLM
from models.response_models import NLIResponse
from utils.json_helpers import clean_json, handle_json_error, handle_label_error


def persist_benchmarks(id,
                       premise,
                       hypothesis,
                       response,
                       true_label,
                       messages,
                       reprompts,
                       file_path
                       ):
    benchmarks = {
        "id": id,
        "premise": premise,
        "hypothesis": hypothesis,
        "thoughts": response["thought_process"],
        "prediction": response["label"],
        "true_label": true_label,
        "correct": response["label"] == true_label,
        "chat_history": messages,
        "reprompt_counts": reprompts
    }

    with open(file_path, "a") as file:
        json.dump(benchmarks, file)
        file.write("\n")


def predict_label(id,
                  premise: str,
                  hypothesis: str,
                  true_label: int,
                  llm: BaseLLM,
                  model_name: str,
                  json_format: dict,
                  json_filepath: str,
                  api: bool = True,
                  sys: str = "",
                  user: str = "",
                  max_retries: int = 5,
                  ) -> dict:
    """
    Predicts entailment label for a given premise-hypothesis pair using BaseLLMs.

    :param api: boolean flag to specify if calling API model or HF
    :param json_filepath:
    :param true_label:
    :param json_format:
    :param hypothesis:
    :param premise:
    :param sys: System prompt
    :param user: User prompt
    :param llm: Language model instance
    :param model_name: Name of the language model to use (optional)
    :param max_retries: Maximum number of retries before giving up (default: 15)
    :return: JSON object containing the move and thoughts, or an error message
    """
    try:
        num_of_reprompts = 0

        sys = sys if sys else system_prompt()
        user = user if user else f"Premise: {premise} \nHypothesis: {hypothesis} \nLabel: "

        if api:
            # Generate system and user prompt from templates, add to LLM messages
            sys = llm.prompt_template("system", sys)
            user = llm.prompt_template("user", user)
            llm.add_messages([sys, user])
        else:
            # Non-API model doesn't use system prompt format
            llm.prompt_template(sys, user)

        # print('****** INPUT ******\n ')
        # pprint(llm.get_messages())

        # HANDLE GENERATION & RETRIES
        for retry in range(max_retries):

            # If reached 3 consecutive retries, reset conversation context and start again
            if (retry + 1) % 3 == 0:
                llm.reset_messages()
                llm.add_messages([sys, user])

            output: str = llm.generate_text(model_name=model_name)
            response_json, error_message = validate_response(output, json_format)

            print('****** OUTPUT ******\n ')
            pprint(output)

            # If response is valid, extract and generate benchmark metrics and return response
            if response_json:
                persist_benchmarks(
                    id=id,
                    premise=premise,
                    hypothesis=hypothesis,
                    response=response_json,
                    true_label=true_label,
                    reprompts=num_of_reprompts,
                    messages=llm.get_messages(),
                    file_path=json_filepath
                )
                return response_json

            # Else if error, re-prompt the user with the error-reprompt message
            print(f'****** ERROR {id} ******\n ')
            pprint(error_message)
            print(f'****** RETRY {retry + 1} ******\n')

            error_prompt = f"Error message: {error_message}. "
            regenerate_prompt = f"Previous prompt: '''{user['content']}'''"
            if api:
                llm.add_messages(
                    [llm.prompt_template("assistant", output), llm.prompt_template("user", error_prompt + regenerate_prompt)])
            else:
                llm.prompt_template(error_prompt, regenerate_prompt)
            num_of_reprompts += 1
    # Return error message if maximum retries exceeded
    except Exception as e:
        pass
    return {'thoughts': 'Exceeded maximum retries', 'label': -1}


def validate_response(output: str, json_format: dict):
    """
    Validates the response using Pydantic models.
    :param output: Output string from the LLM
    :param json_format: JSON format for error messages
    :return: Tuple containing the validated response dict and an error message (if any)
    """
    cleaned_json_str = clean_json(output)

    # Attempt to parse JSON first
    try:
        json_data = json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        # Pass only 2 arguments now
        return handle_json_error(e, json_format)

    # Check if label is a string and attempt to convert
    if 'label' in json_data and isinstance(json_data['label'], str):
        # We want to enforce strict typing - reject string labels
        return handle_label_error(ValueError("Label must be an integer"), json_data['label'])

    # Now try to validate with Pydantic
    try:
        validated_response = NLIResponse.model_validate(json_data)
        return validated_response.model_dump(), None
    except ValueError as e:
        error_msg = str(e)
        if "label" in error_msg.lower():
            label_value = json_data.get('label', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_label_error(ValueError(error_msg), label_value)
        # Pass only 2 arguments now
        return handle_json_error(e, json_format)


def system_prompt():
    return """ You are an expert in natural language reasoning and inference. Your task is to analyze pairs of sentences and determine if the second sentence (hypothesis) can be logically inferred from the first sentence (premise). For each example, I will provide the premise and hypothesis. Your response should be in the following JSON format:
    {
      "thought_process":
        "Step 1. <Identify key information and relationships in the premise, considering logical connections, commonsense understanding, and factual consistency>.
        Step 2. <Analyze how the hypothesis relates to or contradicts the premise based on the information identified in Step 1. Evaluate if the hypothesis can be reasonably inferred from the premise>.
        Step 3. <Explain your final reasoning and conclusion on whether the hypothesis is entailed by the premise or not>",
      "label": "<0 for no entailment, 1 for entailment>"
    }
    Please provide a clear multi-step reasoning chain explaining how you arrived at your final answer, breaking it down into logical components. Ground your response in the given information, logical principles and common-sense reasoning.
    
    Example:
    
    Premise: The dog chased the cat up the tree. Hypothesis: The cat climbed the tree. Label:
    
    {
        "thought_process": "
            Step 1: the premise indicates a scenario where a dog chases a cat, resulting in the cat moving up a tree. The movement 'up the tree' suggests a vertical ascent, typical of climbing behavior. It is common sense that a cat would climb a tree to escape a chasing dog, and there are no known facts that contradict the premise or hypothesis.
            Step 2: 'The cat climbed the tree' can be logically inferred from the premise because the action of climbing is a reasonable and necessary part of the cat moving 'up the tree' as described. Thus, the hypothesis logically follows from the premise.
            Step 3: Based on the logical reasoning, common sense, and lack of contradictory facts, the hypothesis can be inferred from the premise.
            ",
        "label": 1
    }
    """
