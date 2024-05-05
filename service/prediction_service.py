import json
import re
from pprint import pprint

from llm.base_llm import BaseLLM


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


def upgrade_model(model_name: str) -> str:
    """
    A solution to consistently bad re-prompts: upgrading to larger models after a certain number of retries.
    :param model_name:
    :return:
    """
    if "mistral" in model_name or "mixtral" in model_name:
        models = {"1": "open-mistral-7b", "2": "open-mixtral-8x7b", "3": "open-mixtral-8x22b",
                  "4": "mistral-medium-latest", "5": "mistral-large-latest"}
    elif "gpt" in model_name:
        models = {"1": "gpt3.5-turbo", "2": "gpt4-turbo"}
    else:
        raise ValueError("Model not supported")

    return upgrade(models, model_name)


def upgrade(models, model_name):
    inverted_models = {v: k for k, v in models.items()}
    current_model = inverted_models[model_name]
    next_model = int(current_model) + 1
    if next_model > len(models):
        return model_name
    return models[str(next_model)]


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

            # If reached 6 consecutive retries, switch zmodel
            if (retry + 1) % 6 == 0:
                model_name = upgrade_model(model_name)
                model_name = "open-mixtral-8x22b"

            # If reached 9 consecutive retries, switch zmodel
            if (retry + 1) % 9 == 0:
                model_name = upgrade_model(model_name)
                model_name = "mistral-large-latest"

            # Generate move, thoughts JSON using LLM
            output: str = llm.generate_text(model_name=model_name)
            response_json, error_message = validate_response(output, json_format)

            # print('****** OUTPUT ******\n ')
            # pprint(output)

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
    Validates the move generated by the LLM.
    :param json_format:
    :param output: Output string from the LLM
    :return: Tuple containing the response JSON (move and thoughts) and an error message (if any)
    """
    cleaned_json: str = clean_json(output)

    # print("****** CLEANED JSON ******\n")
    # pprint(cleaned_json)

    # Handle JSON Validation
    try:
        response_json: dict = json.loads(cleaned_json)
        label = response_json['label']
        if 'thought_process' not in response_json:
            raise KeyError
    except Exception as e:
        return handle_json_error(e, cleaned_json, json_format)

    # print("****** VALID RESPONSE JSON ******\n")
    # pprint(response_json)

    # Handle Label Validation
    if isinstance(label, str):
        label: int = clean_label(label)

    if label in [0, 1]:
        response_json['label'] = label
        return response_json, None

    return handle_label_error(ValueError, label)


def clean_label(label: str) -> int:
    """Cleans label to only contain a number that's 0 or 1"""
    label = label.strip()
    label = re.sub(r'\D', '', label)
    return int(label)


def clean_json(text: str) -> str:
    """
    Cleans the text into a more readable JSON string, removing excess characters that interfere with parsing json/move.
    :param text: Input text string
    :return: Extracted JSON string, or the original text if no JSON is found
    """
    text = text.lower()
    text = re.sub(r'\n', r'', text)
    start = text.find('{')
    end = text.rfind('}')
    return text[start:end + 1] if start != -1 and end != -1 else text


def handle_json_error(error: Exception, json_str: str, json_schema: dict):
    """
    Returns error-specific prompts to regenerate response for JSON errors.
    :param json_schema:
    :param error: The exception object representing the JSON error
    :param json_str: Response JSON string
    :return: Tuple containing None and the corresponding error message
    """
    if isinstance(error, json.JSONDecodeError):
        return None, f'Invalid JSON object. Regenerate your response, providing your thoughts and move in the correct JSON format: {json_schema}.'
    elif isinstance(error, (KeyError, TypeError)):
        return None, f'Invalid JSON format: JSON response missing the "label" and/or "thought_process" key. Regenerate your response, providing the label key in the correct JSON format: {json_schema}.'
    else:
        return None, f'Invalid JSON response. Regenerate your response, providing your thoughts and move in the correct JSON format: {json_schema}.'


def handle_label_error(error, label):
    """
    Returns error-specific prompts to regenerate response for label errors.
    :param error: The exception object representing the label error
    :param label: The label that caused the error
    :return: Tuple containing None and the corresponding error message
    """
    if isinstance(error, ValueError):
        return None, f"Invalid label: '{label}'. Regenerate your response to my last prompt, but this time ensure that the value at the 'label' key is '0' or '1', and contains no other characters."
    else:
        print(error)
        return None, f"Invalid label. Regenerate your response to my last prompt, ensuring you've formatted it as requested."


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
