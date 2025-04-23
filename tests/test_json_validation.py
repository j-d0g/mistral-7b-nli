"""
Tests for JSON validation functionality.
"""
import json
import pytest

from models.response_models import NLIResponse, ScoringResponse
from utils.json_helpers import clean_json, validate_llm_response


def test_clean_json():
    """Test JSON cleaning function"""
    # Test with valid JSON
    valid_json = '{"label": 1, "thought_process": "analysis"}'
    assert clean_json(valid_json) == valid_json.lower()
    
    # Test with JSON embedded in text
    embedded_json = 'The model says: {"label": 1, "thought_process": "analysis"} and more text'
    assert clean_json(embedded_json) == '{"label": 1, "thought_process": "analysis"}'
    
    # Test with newlines
    json_with_newlines = '{\n"label": 1,\n"thought_process": "analysis"\n}'
    assert clean_json(json_with_newlines) == '{"label": 1,"thought_process": "analysis"}'
    
    # Test with no JSON
    no_json = 'This contains no JSON object'
    assert clean_json(no_json) == no_json.lower()


def test_validate_llm_response_valid():
    """Test validation with valid responses"""
    # Valid NLI response
    valid_nli = '{"thought_process": "This is a valid analysis", "label": 1}'
    result, error = validate_llm_response(valid_nli, NLIResponse)
    assert result is not None
    assert error is None
    assert result["thought_process"] == "this is a valid analysis"
    assert result["label"] == 1
    
    # Valid scoring response
    valid_scoring = '{"score": 4, "label": 1, "improved_thoughts": "Better analysis"}'
    result, error = validate_llm_response(valid_scoring, ScoringResponse)
    assert result is not None
    assert error is None
    assert result["score"] == 4
    assert result["label"] == 1
    assert result["improved_thoughts"] == "better analysis"


def test_validate_llm_response_invalid():
    """Test validation with invalid responses"""
    # Invalid JSON
    invalid_json = '{this is not valid JSON}'
    result, error = validate_llm_response(invalid_json, NLIResponse)
    assert result is None
    assert error is not None
    assert "Invalid JSON" in error
    
    # Missing required field
    missing_field = '{"label": 1}'  # missing thought_process
    result, error = validate_llm_response(missing_field, NLIResponse)
    assert result is None
    assert error is not None
    assert "thought_process" in error.lower()
    
    # Invalid label value
    invalid_label = '{"thought_process": "Analysis", "label": 3}'
    result, error = validate_llm_response(invalid_label, NLIResponse)
    assert result is None
    assert error is not None
    assert "label" in error.lower()
    
    # String label (should be rejected)
    string_label = '{"thought_process": "Analysis", "label": "1"}'
    result, error = validate_llm_response(string_label, NLIResponse)
    assert result is None
    assert error is not None
    assert "label" in error.lower()
    
    # Invalid score
    invalid_score = '{"score": 7, "label": 1, "improved_thoughts": "Analysis"}'
    result, error = validate_llm_response(invalid_score, ScoringResponse)
    assert result is None
    assert error is not None
    assert "score" in error.lower()


def test_json_extraction():
    """Test extraction of JSON from surrounding text"""
    # JSON embedded in a sentence with double quotes
    text = 'I think the answer is: {"thought_process": "Step 1: Analysis", "label": 1} What do you think?'
    result, error = validate_llm_response(text, NLIResponse)
    assert result is not None
    assert error is None
    assert result["thought_process"] == "step 1: analysis"
    assert result["label"] == 1 