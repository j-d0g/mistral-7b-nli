import json
from pprint import pprint

from llm.base_llm import BaseLLM
from models.response_models import ScoringResponse
from utils.json_helpers import (
    clean_json, handle_json_error, handle_label_error, 
    handle_score_error, handle_thoughts_error
)


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

    # Return a default error response if all retries fail
    error_response = {'score': 0, 'improved_thoughts': 'Exceeded maximum retries', 'label': predicted_label}
    pprint(error_response)
    return error_response


def validate_response(output: str, json_format: dict):
    """
    Validates the response from the scoring model using Pydantic.
    :param output: Output string from the LLM
    :param json_format: JSON format for error messages
    :return: Tuple containing the validated response dict and an error message (if any)
    """
    cleaned_json: str = clean_json(output)

    # Handle JSON Parsing
    try:
        json_data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        return handle_json_error(e, json_format)
        
    # Add explicit type checks before Pydantic validation
    if 'label' in json_data and isinstance(json_data['label'], str):
        return handle_label_error(ValueError("Label must be an integer"), json_data['label'])
    if 'score' in json_data and isinstance(json_data['score'], str):
        return handle_score_error(ValueError("Score must be an integer"), json_data['score'])
        
    # Handle Pydantic Validation
    try:
        validated_response = ScoringResponse.model_validate(json_data)
        return validated_response.model_dump(), None
    except ValueError as e:
        # Handle Pydantic validation errors
        error_msg = str(e)
        
        # Categorize errors based on field name in error message
        if "label" in error_msg.lower():
            label_value = json_data.get('label', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_label_error(ValueError(error_msg), label_value)
        elif "score" in error_msg.lower():
            score_value = json_data.get('score', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_score_error(ValueError(error_msg), score_value)
        elif "improved_thoughts" in error_msg.lower():
            return handle_thoughts_error(ValueError(error_msg))
            
        return handle_json_error(e, json_format)
