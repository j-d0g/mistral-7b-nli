import json
import time  # Add time import for sleep
from pprint import pprint

from llm.base_llm import BaseLLM
from models.response_models import ScoringResponse
from utils.json_helpers import (
    clean_json, handle_json_error, handle_label_error, 
    handle_score_error, handle_thought_process_error
)
from utils.prompts import get_prompt  # Import get_prompt instead of specific prompts


def persist_benchmarks(id,
                       premise,
                       hypothesis,
                       thought_process,
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
        "thought_process": thought_process,
        "predicted_label": predicted_label,
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
                   thought_process: str,
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
    :param thought_process:
    :param predicted_label:
    :param true_label:
    :param sys: system prompt
    :param llm: basellm instance
    :param model_name: name of model to use (optional)
    :param max_retries: max number of retries before giving up (default: 15)
    :return: JSON object containing analysis, score and thought_process
    """
    try:
        print(id)
        num_of_reprompts = 0

        # Use get_prompt to get the prompt and schema if not provided
        if not sys or not json_format:
            default_prompt, default_schema = get_prompt('scoring')
            sys = sys or default_prompt
            json_format = json_format or default_schema

        # Generate system and user prompt from templates, add to LLM messages
        sys = llm.prompt_template("system", sys)
        prompt = f"""Premise: {premise} \nHypothesis: {hypothesis} \nThought-Process: {thought_process} \nPredicted Label: {predicted_label}"""
        user = llm.prompt_template("user", prompt)
        llm.add_messages([sys, user])

        # print('****** INPUT ******\n ')
        # pprint(prompt)

        # HANDLE GENERATION & RETRIES
        retries_due_to_rate_limit = 0
        max_total_retries = max_retries * 3  # Allow more total retries to handle rate limiting
        
        for retry in range(max_total_retries):

            # If reached 3 consecutive retries, reset conversation context and start again
            if (retry + 1) % 3 == 0 and retries_due_to_rate_limit == 0:
                llm.reset_messages()
                llm.add_messages([sys, user])

            # If reached 6 consecutive retries, switch model
            if (retry + 1) % 6 == 0 and retries_due_to_rate_limit == 0:
                model_name = "open-mixtral-8x22b"

            if (retry + 1) % 9 == 0 and retries_due_to_rate_limit == 0:
                model_name = "mistral-large-latest"

            try:
                # Generate response JSON using LLM
                output: str = llm.generate_text(model_name=model_name)
                
                # Reset rate limit counter on successful call
                retries_due_to_rate_limit = 0
                
                response_json, error_message = validate_response(output, json_format)

                # If response is valid, extract and generate benchmark metrics and return response
                if response_json:
                    if response_json["score"] >= 4:
                        persist_benchmarks(
                            id=id,
                            premise=premise,
                            hypothesis=hypothesis,
                            thought_process=thought_process,
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
                                thought_process=thought_process,
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
                            thought_process=response_json["improved_thought_process"],
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
                error_str = str(e)
                
                # Check for rate limit errors (adapt these patterns based on actual API error messages)
                if any(term in error_str.lower() for term in ["rate limit", "too many requests", "429"]):
                    retries_due_to_rate_limit += 1
                    backoff_time = min(2 ** retries_due_to_rate_limit, 60)  # Exponential backoff, max 60 seconds
                    print(f"Rate limit hit, backing off for {backoff_time} seconds (retry {retries_due_to_rate_limit})")
                    time.sleep(backoff_time)
                    # Don't increment the regular retry counter, just wait and try again
                    continue
                else:
                    # For other errors, treat as a regular retry failure
                    print(f"Non-rate-limit error for ID {id}: {error_str}")
                    num_of_reprompts += 1
                
    except Exception as e:
        print(f'****** UKNOWN ERROR: ID-{id} ******\n ')
        print(f"Error details: {str(e)}")
        return {'thought_process': e, 'predicted_label': -1}

    # Return a default error response if all retries fail
    error_response = {'score': 0, 'improved_thought_process': 'Exceeded maximum retries', 'predicted_label': predicted_label}
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
    if 'predicted_label' in json_data and isinstance(json_data['predicted_label'], str):
        return handle_label_error(ValueError("Label must be an integer"), json_data['predicted_label'])
    # For backward compatibility
    elif 'label' in json_data and isinstance(json_data['label'], str):
        return handle_label_error(ValueError("Label must be an integer"), json_data['label'])
    if 'score' in json_data and isinstance(json_data['score'], str):
        return handle_score_error(ValueError("Score must be an integer"), json_data['score'])
    
    # If using old 'label' field, rename it to 'predicted_label'
    if 'label' in json_data and 'predicted_label' not in json_data:
        json_data['predicted_label'] = json_data.pop('label')
        
    # Handle Pydantic Validation
    try:
        validated_response = ScoringResponse.model_validate(json_data)
        return validated_response.model_dump(), None
    except ValueError as e:
        # Handle Pydantic validation errors
        error_msg = str(e)
        
        # Categorize errors based on field name in error message
        if "predicted_label" in error_msg.lower():
            label_value = json_data.get('predicted_label', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_label_error(ValueError(error_msg), label_value)
        elif "label" in error_msg.lower():
            label_value = json_data.get('label', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_label_error(ValueError(error_msg), label_value)
        elif "score" in error_msg.lower():
            score_value = json_data.get('score', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_score_error(ValueError(error_msg), score_value)
        elif "improved_thought_process" in error_msg.lower():
            return handle_thought_process_error(ValueError(error_msg))
            
        return handle_json_error(e, json_format)
