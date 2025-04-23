"""
Tests for the prediction service.
"""
import pytest
import os
import json
from unittest.mock import MagicMock, patch

from service.prediction_service import predict_label, validate_response, system_prompt
from llm.mistral import Mistral


@pytest.mark.parametrize("api_response,expected_valid", [
    ('{"thought_process": "Valid analysis", "label": 1}', True),
    ('{"thought_process": "Valid analysis", "label": "1"}', False),  # String label
    ('{"label": 1}', False),  # Missing thought_process
    ('{"thought_process": "Valid analysis", "label": 3}', False),  # Invalid label value
    ('{invalid json}', False),  # Invalid JSON
])
def test_validate_response(api_response, expected_valid):
    """Test validation of API responses"""
    json_format = {
        "thought_process": "...",
        "label": "0 or 1"
    }
    
    result, error = validate_response(api_response, json_format)
    
    if expected_valid:
        assert result is not None
        assert error is None
        assert "thought_process" in result
        assert "label" in result
    else:
        assert result is None
        assert error is not None


def test_system_prompt():
    """Test that system prompt is properly formatted"""
    prompt = system_prompt()
    
    # Should be a string
    assert isinstance(prompt, str)
    
    # Should contain format specification
    assert "JSON format" in prompt
    assert "thought_process" in prompt
    assert "label" in prompt
    
    # Should contain example
    assert "Example" in prompt


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_predict_label_with_mock_llm():
    """Test the prediction function with a mocked LLM client"""
    # Setup a mock LLM that returns a valid response
    mock_llm = MagicMock()
    mock_llm.get_messages.return_value = []
    mock_llm.prompt_template.return_value = {"role": "system", "content": "test"}
    mock_llm.generate_text.return_value = '{"thought_process": "Test analysis", "label": 1}'
    
    result = predict_label(
        id="test-1",
        premise="The dog chased the cat.",
        hypothesis="The cat ran.",
        true_label=1,
        llm=mock_llm,
        model_name="mock-model",
        json_format={"thought_process": "...", "label": "0 or 1"},
        json_filepath="/tmp/test.jsonl",
        max_retries=1
    )
    
    # Verify result
    assert result is not None
    assert result["thought_process"] == "test analysis"
    assert result["label"] == 1
    
    # Verify LLM was called correctly
    mock_llm.generate_text.assert_called_once()
    mock_llm.add_messages.assert_called()


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_integration_predict_label(mistral_client, sample_premise, sample_hypothesis, 
                                  sample_true_label, temp_jsonl_file):
    """Integration test for predict_label with actual API call"""
    # This test will make an actual API call if MISTRAL_API_KEY is set
    # Skip if running offline or if key is not available
    
    json_format = {
        "thought_process": "Step by step reasoning",
        "label": "0 or 1"
    }
    
    result = predict_label(
        id="test-integration",
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        true_label=sample_true_label,
        llm=mistral_client,
        model_name="open-mistral-7b",
        json_format=json_format,
        json_filepath=temp_jsonl_file,
        max_retries=1  # Just do one attempt to speed up test
    )
    
    # Check result structure
    assert result is not None
    assert "thought_process" in result
    assert "label" in result
    assert isinstance(result["label"], int)
    assert result["label"] in [0, 1] 