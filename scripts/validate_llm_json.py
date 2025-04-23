#!/usr/bin/env python3
"""
Utility script for validating LLM JSON outputs with Pydantic.
This can be used as a standalone tool or imported as a module.
"""
import json
import re
import sys
from typing import Tuple, Dict, Any, Type, Optional, List, Union

from pydantic import BaseModel, ValidationError

def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text that might contain other content"""
    # Remove newlines to simplify regex
    text = re.sub(r'\n', ' ', text)
    # Find anything that looks like a JSON object
    match = re.search(r'({.*})', text)
    if match:
        return match.group(1)
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
    json_str = extract_json_from_text(response) if auto_clean else response
    
    # First try to parse as JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None, f"Invalid JSON. Response must be a valid JSON object conforming to the {model_class.__name__} schema."
    
    # Then validate with Pydantic
    try:
        validated_data = model_class.model_validate(data)
        return validated_data.model_dump(), None
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

def main():
    """Demo usage when run as a script"""
    # Import here to avoid circular imports if this module is imported elsewhere
    from models.response_models import NLIResponse
    
    # Example usage
    example_responses = [
        '{"thought_process": "This is valid reasoning", "label": 1}',
        '{"thought_process": "Invalid label", "label": 3}',
        '{"label": 1}',  # Missing thought_process
        'Not a JSON at all',
        'The model said: {"thought_process": "Extracted from text", "label": 0}'
    ]
    
    print("Demonstrating LLM JSON validation with Pydantic\n")
    
    for i, response in enumerate(example_responses):
        print(f"Example {i+1}:")
        print(f"Input: {response}")
        
        result, error = validate_llm_response(response, NLIResponse)
        
        if result:
            print("✅ Valid")
            print(f"Validated data: {result}")
        else:
            print("❌ Invalid")
            print(f"Error: {error}")
        print()

if __name__ == "__main__":
    main() 