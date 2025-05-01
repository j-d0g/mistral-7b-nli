import json
import time
from pprint import pprint

from llm.base_llm import BaseLLM
from models.response_models import ReflectionResponse
from utils.json_helpers import clean_json, handle_json_error, handle_label_error
from prompts import get_prompt

def persist_reflection(
    id,
    premise,
    hypothesis,
    true_label,
    thought_process,
    predicted_label,
    reflection_result,
    file_path
):
    """
    Persist reflection results to a JSON Lines file.
    """
    # Create a structured output with ordered fields for better readability
    output = {
        "id": id,
        "premise": premise,
        "hypothesis": hypothesis,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "thought_process": thought_process,
        "error_analysis": reflection_result.get("error_analysis", ""),
        "improved_thought_process": reflection_result.get("improved_thought_process", "")
    }
    
    # Add predicted_label from reflection if available
    if "predicted_label" in reflection_result:
        output["predicted_label"] = reflection_result["predicted_label"]
    
    with open(file_path, "a") as file:
        json.dump(output, file)
        file.write("\n")

def validate_reflection_response(output: str, schema: dict):
    """
    Validates the LLM output against the reflection schema.
    """
    cleaned_json_str = clean_json(output)
    
    # Attempt to parse JSON first
    try:
        json_data = json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        return handle_json_error(e, schema)
    
    # Check for required fields
    missing_keys = [key for key in schema if key not in json_data]
    if missing_keys:
        return None, f"JSON Validation Error: Missing keys - {', '.join(missing_keys)}. Expected: {list(schema.keys())}."
    
    # Ensure predicted_label is an integer
    if "predicted_label" in json_data:
        if not isinstance(json_data["predicted_label"], int):
            # Try to convert if it's a string number
            if isinstance(json_data["predicted_label"], str) and json_data["predicted_label"] in ["0", "1"]:
                json_data["predicted_label"] = int(json_data["predicted_label"])
                print(f"Converted string predicted_label '{json_data['predicted_label']}' to integer")
            else:
                return None, f"JSON Validation Error: 'predicted_label' must be 0 or 1. Got: {json_data['predicted_label']}"
        
        # Check if the value is valid
        if json_data["predicted_label"] not in [0, 1]:
            return None, f"JSON Validation Error: 'predicted_label' must be 0 or 1. Got: {json_data['predicted_label']}"
    
    # Validate using ReflectionResponse model
    try:
        validated_response = ReflectionResponse.model_validate(json_data)
        return validated_response.model_dump(), None
    except ValueError as e:
        error_msg = str(e)
        return None, f"JSON Validation Error: {error_msg}"

def generate_reflection(
    id,
    premise: str,
    hypothesis: str,
    thought_process: str,
    predicted_label: int,
    true_label: int,
    llm: BaseLLM,
    model_name: str,
    json_filepath: str,
    max_retries: int = 5
) -> dict:
    """
    Generates a reflection on the initial thought process, analyzing errors and improving the reasoning.
    
    Parameters:
    - id: Unique identifier for the example
    - premise: The premise text
    - hypothesis: The hypothesis text
    - thought_process: The original thought process to reflect on
    - predicted_label: The original predicted label
    - true_label: The ground truth label
    - llm: Language model instance
    - model_name: Name of the model to use
    - json_filepath: Path to save the results
    - max_retries: Maximum retry attempts
    
    Returns:
    - Dictionary containing the reflection results
    """
    try:
        # Clear any previous conversation
        llm.reset_messages()
        
        # Get the reflection prompt and schema
        prompt_template, schema = get_prompt('reflection')
        
        # Format the prompt with the example data
        schema_string = json.dumps(schema, indent=4)
        formatted_prompt = prompt_template.format(
            premise=premise,
            hypothesis=hypothesis,
            thought_process=thought_process,
            predicted_label=predicted_label,
            true_label=true_label,
            schema_string=schema_string
        )
        
        # Add the formatted prompt as a user message
        user_message = llm.prompt_template("user", formatted_prompt)
        llm.add_messages([user_message])
        
        # Track retries
        retries_due_to_rate_limit = 0
        max_total_retries = max_retries * 3  # Extra buffer for rate limiting
        
        for attempt in range(max_total_retries):
            try:
                # Generate the reflection
                output = llm.generate_text(model_name=model_name)
                
                # Reset rate limit counter on successful call
                retries_due_to_rate_limit = 0
                
                # Validate the response
                reflection_result, error_message = validate_reflection_response(output, schema)
                
                # If valid, persist and return the result
                if reflection_result:
                    persist_reflection(
                        id=id,
                        premise=premise,
                        hypothesis=hypothesis,
                        true_label=true_label,
                        thought_process=thought_process,
                        predicted_label=predicted_label,
                        reflection_result=reflection_result,
                        file_path=json_filepath
                    )
                    return reflection_result
                
                # Handle invalid response
                print(f'****** ERROR for ID {id} (Attempt {attempt+1}) ******')
                print(error_message)
                
                # Add error feedback and retry
                error_prompt = f"Error message: {error_message}. Please try again, focusing on the required JSON format."
                llm.add_messages([
                    llm.prompt_template("assistant", output),
                    llm.prompt_template("user", error_prompt)
                ])
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limiting
                if any(term in error_str.lower() for term in ["rate limit", "too many requests", "429"]):
                    retries_due_to_rate_limit += 1
                    backoff_time = min(2 ** retries_due_to_rate_limit, 60)  # Exponential backoff, max 60s
                    print(f"Rate limit hit, backing off for {backoff_time} seconds (retry {retries_due_to_rate_limit})")
                    time.sleep(backoff_time)
                    continue
                else:
                    # Other API errors
                    print(f"Error during reflection generation: {error_str}")
                    if attempt >= max_retries - 1:
                        return {
                            'error': 'Maximum retries exceeded',
                            'details': error_str,
                            'predicted_label': true_label  # Default to true label on failure
                        }
        
        # If we exhaust all retries
        return {
            'error': 'Maximum retries exceeded',
            'details': 'Failed to generate valid reflection after multiple attempts',
            'predicted_label': true_label  # Default to true label on failure
        }
        
    except Exception as e:
        print(f"Fatal error in generate_reflection: {str(e)}")
        return {
            'error': 'Fatal error',
            'details': str(e),
            'predicted_label': true_label  # Default to true label on failure
        } 