import json
import re
import time  # Add time import for sleep
from pprint import pprint

from llm.base_llm import BaseLLM
from models.response_models import NLIResponse
from utils.json_helpers import clean_json, handle_json_error, handle_label_error
from prompts import get_prompt  # Import get_prompt instead of specific prompts


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
        "thought_process": response["thought_process"],
        "predicted_label": response["predicted_label"],
        "true_label": true_label,
        "correct": response["predicted_label"] == true_label,
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
    :return: JSON object containing the thought_process and label, or an error message
    """
    try:
        num_of_reprompts = 0

        # Use get_prompt to get the prompt and schema if not provided
        if not sys:
            default_prompt, default_schema = get_prompt('initial_generation')
            sys = default_prompt
            # If json_format wasn't explicitly provided, use the default schema
            if not json_format:
                json_format = default_schema
                
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
        retries_due_to_rate_limit = 0
        max_total_retries = max_retries * 3  # Allow more total retries to handle rate limiting
        
        for retry in range(max_total_retries):
            # If reached 3 consecutive retries, reset conversation context and start again
            if (retry + 1) % 3 == 0 and retries_due_to_rate_limit == 0:
                llm.reset_messages()
                llm.add_messages([sys, user])

            try:
                output: str = llm.generate_text(model_name=model_name)
                
                # Reset rate limit counter on successful call
                retries_due_to_rate_limit = 0
                
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
                
            except Exception as e:
                error_str = str(e)
                
                # Check for rate limit errors (adapt these patterns based on actual API error messages)
                if any(term in error_str.lower() for term in ["rate limit", "too many requests", "429"]):
                    retries_due_to_rate_limit += 1
                    backoff_time = min(2 ** retries_due_to_rate_limit, 60)  # Exponential backoff, max 60 seconds
                    print(f"Rate limit hit, backing off for {backoff_time} seconds (retry {retries_due_to_rate_limit})")
                    time.sleep(backoff_time)
                    # Don't increment the regular retry counter
                    continue
                else:
                    # For other errors, treat as a regular retry failure
                    print(f"Non-rate-limit error: {error_str}")
                    num_of_reprompts += 1
            
    # Return error message if maximum retries exceeded
    except Exception as e:
        print(f"Fatal error in predict_label: {str(e)}")
    return {'thought_process': 'Exceeded maximum retries', 'predicted_label': -1}


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

    # Standardize field names
    
    # If using old 'label' field, rename it to 'predicted_label' for consistency
    if 'label' in json_data and 'predicted_label' not in json_data:
        json_data['predicted_label'] = json_data.pop('label')
        print("Standardized field name: 'label' → 'predicted_label'")

    # Check if label is a string and attempt to convert to an integer
    if 'predicted_label' in json_data and isinstance(json_data['predicted_label'], str):
        try:
            # Try to convert string to integer (only for "0" or "1")
            if json_data['predicted_label'] in ["0", "1"]:
                json_data['predicted_label'] = int(json_data['predicted_label'])
                print(f"Converted string predicted_label '{json_data['predicted_label']}' to integer")
            else:
                return handle_label_error(ValueError("Label must be 0 or 1"), json_data['predicted_label'])
        except ValueError:
            return handle_label_error(ValueError("Label must be an integer"), json_data['predicted_label'])

    # Now try to validate with Pydantic
    try:
        validated_response = NLIResponse.model_validate(json_data)
        return validated_response.model_dump(), None
    except ValueError as e:
        error_msg = str(e)
        if "predicted_label" in error_msg.lower():
            label_value = json_data.get('predicted_label', 'unknown') if 'json_data' in locals() else 'unknown'
            return handle_label_error(ValueError(error_msg), label_value)
        # Pass only 2 arguments now
        return handle_json_error(e, json_format)
