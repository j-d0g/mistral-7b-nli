import json
import re
from pprint import pprint

from llm.base_llm import BaseLLM


def persist_benchmarks(id,
                       premise,
                       hypothesis,
                       thoughts,
                       predicted_label,
                       true_label,
                       score_response,
                       reprompts,
                       file_path
                       ):
    benchmarks = {
        "id": id,
        "premise": premise,
        "hypothesis": hypothesis,
        "thoughts": thoughts,
        "prediction": predicted_label,
        "true_label": true_label,
        "score_json": score_response,
        "reprompt_counts": reprompts
    }

    with open(file_path, "a") as file:
        json.dump(benchmarks, file)
        file.write("\n")


def generate_score(id,
                   sys: str,
                   premise: str,
                   hypothesis: str,
                   thoughts: str,
                   predicted_label: int,
                   true_label: int,
                   llm: BaseLLM,
                   model_name: str,
                   json_format: dict,
                   json_filepath: str,
                   max_retries: int = 10,
                   depth: int = 0
                   ) -> dict:
    """
    Generates a chess move using the Mistral API.
    :param depth:
    :param json_filepath: path to persist
    :param json_format: schema for response
    :param hypothesis:
    :param premise:
    :param thoughts:
    :param predicted_label:
    :param true_label:
    :param sys: system prompt
    :param llm: basellm instance
    :param model_name: name of model to use (optional)
    :param max_retries: max number of retries before giving up (default: 15)
    :return: JSON object containing analysis, score and thoughts
    """
    try:
        print(id)
        num_of_reprompts = 0

        # Generate system and user prompt from templates, add to LLM messages
        sys = llm.prompt_template("system", sys)
        prompt = f"""Premise: {premise} \nHypothesis: {hypothesis} \nThought-Process: {thoughts} \nPredicted Label: {predicted_label}"""
        user = llm.prompt_template("user", prompt)
        llm.add_messages([sys, user])

        # print('****** INPUT ******\n ')
        # pprint(prompt)

        # HANDLE GENERATION & RETRIES
        for retry in range(max_retries):

            # If reached 3 consecutive retries, reset conversation context and start again
            if (retry + 1) % 3 == 0:
                llm.reset_messages()
                llm.add_messages([sys, user])

            # If reached 6 consecutive retries, switch model
            if (retry + 1) % 6 == 0:
                model_name = "open-mixtral-8x22b"

            if (retry + 1) % 9 == 0:
                model_name = "mistral-large-latest"

            # Generate move, thoughts JSON using LLM
            output: str = llm.generate_text(model_name=model_name)
            response_json, error_message = validate_response(output, json_format)

            # If response is valid, extract and generate benchmark metrics and return response
            if response_json:
                if response_json["score"] >= 4:
                    persist_benchmarks(
                        id=id,
                        premise=premise,
                        hypothesis=hypothesis,
                        thoughts=thoughts,
                        score_response=response_json,
                        predicted_label=predicted_label,
                        true_label=true_label,
                        reprompts=num_of_reprompts,
                        file_path=f'{json_filepath}/gold_standard.json'
                    )

                else:
                    # Recursively re-score and re-prompt until score is adequate or max-depth is reached
                    if depth > 3:
                        print("****** MAX DEPTH REACHED ******\n")
                        persist_benchmarks(
                            id=id,
                            premise=premise,
                            hypothesis=hypothesis,
                            thoughts=thoughts,
                            score_response=response_json,
                            predicted_label=predicted_label,
                            true_label=true_label,
                            reprompts=num_of_reprompts,
                            file_path=f'{json_filepath}/low_standard.json'
                        )

                    print(f"****** RE-SCORE: ID {id} ******\n")
                    llm.reset_messages()
                    response_json = generate_score(
                        id=id,
                        sys=sys,
                        premise=premise,
                        hypothesis=hypothesis,
                        thoughts=response_json["improved_thoughts"],
                        predicted_label=predicted_label,
                        true_label=true_label,
                        llm=llm,
                        model_name='open-mixtral-8x22b',
                        json_format=json_format,
                        json_filepath=json_filepath,
                        depth=depth + 1
                    )

                return response_json

        # Else if error, re-prompt the user with the error-reprompt message
        print(f'****** ERROR: ID {id} ******\n ')
        pprint(error_message)
        print(f'****** RETRY {int(retry) + 1} ******\n')

        regenerate_message = f"{error_message}. Previous prompt: '''{user['content']}'''"
        llm.add_messages(
            [llm.prompt_template("assistant", output), llm.prompt_template("user", regenerate_message)])
        num_of_reprompts += 1
    except Exception as e:
        print(f'****** UKNOWN ERROR: ID-{id} ******\n ')
        return {'thoughts': e, 'label': -1}

    error = {'thoughts': 'Exceeded maximum retries', 'label': -1}
    pprint(error)


def validate_response(output: str, json_format: dict):
    """
    Validates the move generated by the Mistral API.
    :param json_format:
    :param output: Output string from the Mistral API
    :param board: Current chess board state
    :return: Tuple containing the response JSON (move and thoughts) and an error message (if any)
    """
    cleaned_json: str = clean_json(output)

    # print("****** CLEANED JSON ******\n")
    # pprint(cleaned_json)

    # Handle JSON Validation
    try:
        response_json: dict = json.loads(cleaned_json)
        score = response_json['score']
        label = response_json['label']
        improved_thoughts = response_json['improved_thoughts']
    except Exception as e:
        return handle_json_error(e, json_format)

    # print("****** VALID RESPONSE JSON ******\n")
    # pprint(response_json)

    # Clean label
    if label != None and isinstance(label, str):
        label = clean_int(label)
        if (not label) or (label not in [0, 1]):
            return handle_label_error(ValueError, label)
    response_json['label'] = label

    # Clean score
    if isinstance(score, str):
        score = clean_int(score)
        if (not score) or (score not in range(6)):
            return handle_score_error(ValueError, score)
    response_json['score'] = score

    # Re-prompt thoughts over optimal range
    if improved_thoughts:
        if len(improved_thoughts) > 1000:
            return handle_thoughts_error(ValueError)

    return response_json, None


def clean_int(label: str) -> int:
    """ Cleans number """
    label = label.strip()
    label = re.sub(r'\D', '', label)
    return int(label) if label else None


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


def handle_json_error(error: Exception, json_schema: dict):
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
        return None, f'Invalid JSON format: JSON response is missing the field(s) "label", "score" and/or "improved_thoughts". Regenerate your response, providing the label key in the correct JSON format: {json_schema}.'
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
        return None, f"Invalid label. Regenerate your response to my last prompt, ensuring you've formatted it as requested."


def handle_score_error(error, score):
    """
    Returns error-specific prompts to regenerate response for label errors.
    :param score: The score value that caused the error
    :param error: The exception object representing the score error
    :return: Tuple containing None and the corresponding error message
    """
    if isinstance(error, ValueError):
        return None, f"Invalid score: '{score}'. Regenerate your response to my last prompt, but this time ensure that the 'score' field contains a value between 0 and 5 only."
    else:
        return None, f"Invalid score. Regenerate your response to my last prompt, ensuring you've formatted it as requested."


def handle_thoughts_error(error):
    """
    Returns error-specific prompts to regenerate response for label errors.
    :param thoughts:
    :param error: The exception object representing the score error
    :return: Tuple containing None and the corresponding error message
    """
    if isinstance(error, ValueError):
        return None, f"Your thoughts are too long. Please provide a more concise improved thought-chain."
    else:
        return None, f"Invalid score. Regenerate your response to my last prompt, ensuring you've formatted it as requested."
