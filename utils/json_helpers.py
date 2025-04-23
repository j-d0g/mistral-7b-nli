"""
JSON utilities for working with LLM responses.

This module provides functions for extracting, cleaning and validating 
JSON from LLM responses, particularly for NLI tasks.
"""
import re
import json
from typing import Tuple, Dict, Any, Type, Optional, Union

from pydantic import BaseModel, ValidationError


def clean_json(text: str) -> str:
    """
    Extracts and cleans a JSON object from text that might contain other content.
    
    Args:
        text: Input text string, potentially containing a JSON object
        
    Returns:
        The extracted JSON string, or the original text if no JSON is found
    """
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove newlines to simplify parsing
    text = re.sub(r'\n', r'', text)
    
    # Find the first { and last } to extract the JSON object
    start = text.find('{')
    end = text.rfind('}')
    
    if start != -1 and end != -1:
        return text[start:end + 1]
    else:
        return text


def validate_llm_response(
    response: str, 
    model_class: Type[BaseModel],
    auto_clean: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate LLM response against a Pydantic model.
    
    Args:
        response: Raw LLM response text
        model_class: Pydantic model class to validate against
        auto_clean: Whether to try cleaning the JSON from surrounding text
        
    Returns:
        Tuple of (validated_data_dict, error_message)
        If validation succeeds, error_message will be None
        If validation fails, validated_data_dict will be None
    """
    # Clean the JSON if requested
    json_str = clean_json(response) if auto_clean else response
    
    # First try to parse as JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, f"Invalid JSON. Response must be a valid JSON object conforming to the {model_class.__name__} schema."
    
    # Check for string labels and reject them
    if 'label' in data and isinstance(data['label'], str):
        return None, f"Invalid label: '{data['label']}'. Label must be an integer (0 or 1)."
        
    # Then validate with Pydantic
    try:
        validated_response = model_class.model_validate(data)
        return validated_response.model_dump(), None
    except ValidationError as e:
        error_details = str(e)
        field_name = None
        
        # Extract field name from error for better error messages
        for error in e.errors():
            if 'loc' in error and error['loc']:
                field_name = error['loc'][0]
                break
        
        if field_name:
            return None, f"Validation error in field '{field_name}': {error_details}"
        return None, f"Validation error: {error_details}"


# Error handler functions
def handle_json_error(error: Exception, json_format: Dict[str, Any]) -> Tuple[None, str]:
    """
    Returns error-specific prompts to regenerate response for JSON errors.
    
    Args:
        error: The exception object representing the JSON error
        json_format: JSON format for error messages
        
    Returns:
        Tuple containing None and the corresponding error message
    """
    return None, f'Invalid JSON object. Regenerate your response, providing your analysis in the correct JSON format: {json_format}.'


def handle_label_error(error: Exception, label: Any) -> Tuple[None, str]:
    """
    Returns error-specific prompts to regenerate response for label errors.
    
    Args:
        error: The exception object representing the label error
        label: The label that caused the error
        
    Returns:
        Tuple containing None and the corresponding error message
    """
    return None, f"Invalid label: '{label}'. Regenerate your response to my last prompt, but this time ensure that the value at the 'label' key is '0' or '1', and contains no other characters."


def handle_score_error(error: Exception, score: Any) -> Tuple[None, str]:
    """
    Returns error-specific prompts to regenerate response for score errors.
    
    Args:
        error: The exception object representing the score error
        score: The score that caused the error
        
    Returns:
        Tuple containing None and the corresponding error message
    """
    return None, f"Invalid score: '{score}'. Regenerate your response to my last prompt, but this time ensure that the 'score' field contains a value between 0 and 5 only."


def handle_thoughts_error(error: Exception) -> Tuple[None, str]:
    """
    Returns error-specific prompts to regenerate response for thoughts errors.
    
    Args:
        error: The exception object representing the thoughts error
        
    Returns:
        Tuple containing None and the corresponding error message
    """
    return None, "Invalid improved_thoughts. Regenerate your response, ensuring that the 'improved_thoughts' field is present and contains valid text." 