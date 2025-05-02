"""
Tests for the reflection service.
"""
import pytest
import os
import json
import time
from unittest.mock import MagicMock, patch

from service.reflection_service import generate_reflection, validate_reflection_response, persist_reflection
from models.response_models import ReflectionResponse


@pytest.mark.parametrize("api_response,expected_valid", [
    ('{"error_analysis": "Valid analysis", "improved_thought_process": "Better reasoning", "predicted_label": 1}', True),
    ('{"error_analysis": "Valid analysis", "improved_thought_process": "Better reasoning", "predicted_label": "1"}', True),  # String label gets converted
    ('{"error_analysis": "Valid analysis", "improved_thought_process": "Better reasoning"}', True),  # Missing predicted_label is okay
    ('{"improved_thought_process": "Better reasoning", "predicted_label": 1}', False),  # Missing error_analysis
    ('{"error_analysis": "Valid analysis", "predicted_label": 1}', False),  # Missing improved_thought_process
    ('{"error_analysis": "Valid analysis", "improved_thought_process": "Better reasoning", "predicted_label": 3}', False),  # Invalid label value
    ('{invalid json}', False),  # Invalid JSON
])
def test_validate_reflection_response(api_response, expected_valid):
    """Test validation of reflection API responses"""
    json_format = {
        "error_analysis": "Analysis of the error",
        "improved_thought_process": "Improved reasoning",
        "predicted_label": "0 or 1 (optional)"
    }
    
    result, error = validate_reflection_response(api_response, json_format)
    
    if expected_valid:
        assert result is not None
        assert error is None
        assert "error_analysis" in result
        assert "improved_thought_process" in result
    else:
        assert result is None
        assert error is not None


def test_persist_reflection(temp_jsonl_file):
    """Test that reflection results are correctly persisted to a file"""
    # Test data
    id = "test-1"
    premise = "The dog chased the cat."
    hypothesis = "The cat ran."
    true_label = 1
    thought_process = "Step 1: Analysis..."
    predicted_label = 0  # Incorrect prediction
    reflection_result = {
        "error_analysis": "The original reasoning didn't consider X",
        "improved_thought_process": "Better analysis that considers X",
        "predicted_label": 1  # Corrected prediction
    }
    
    # Persist the reflection
    persist_reflection(
        id=id,
        premise=premise,
        hypothesis=hypothesis,
        true_label=true_label,
        thought_process=thought_process,
        predicted_label=predicted_label,
        reflection_result=reflection_result,
        file_path=temp_jsonl_file
    )
    
    # Read the file and verify the content
    with open(temp_jsonl_file, 'r') as f:
        content = json.loads(f.read().strip())
    
    # Verify all fields are present and correct
    assert content["id"] == id
    assert content["premise"] == premise
    assert content["hypothesis"] == hypothesis
    assert content["true_label"] == true_label
    assert content["thought_process"] == thought_process
    assert content["error_analysis"] == reflection_result["error_analysis"]
    assert content["improved_thought_process"] == reflection_result["improved_thought_process"]
    assert content["predicted_label"] == reflection_result["predicted_label"]


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_generate_reflection_with_mock_llm(sample_premise, sample_hypothesis, sample_thought_process):
    """Test the reflection function with a mocked LLM client"""
    # Setup a mock LLM that returns a valid response
    mock_llm = MagicMock()
    mock_llm.get_messages.return_value = []
    mock_llm.prompt_template.return_value = {"role": "system", "content": "test"}
    mock_llm.generate_text.return_value = '{"error_analysis": "The original reasoning missed key aspects", "improved_thought_process": "Better analysis with all aspects considered", "predicted_label": 1}'
    
    # Create a temporary file path
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
    temp_file.close()
    
    result = generate_reflection(
        id="test-1",
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        thought_process=sample_thought_process,
        predicted_label=0,  # Incorrect prediction
        true_label=1,
        llm=mock_llm,
        model_name="mock-model",
        json_filepath=temp_file.name,
        max_retries=1
    )
    
    # Verify result
    assert result is not None
    assert result["error_analysis"] == "The original reasoning missed key aspects"
    assert result["improved_thought_process"] == "Better analysis with all aspects considered"
    assert result["predicted_label"] == 1
    
    # Verify LLM was called correctly
    mock_llm.generate_text.assert_called_once()
    mock_llm.reset_messages.assert_called_once()
    mock_llm.add_messages.assert_called()
    
    # Clean up temp file
    os.unlink(temp_file.name)


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_generate_reflection_retry_on_validation_error():
    """Test that reflection service retries on validation errors"""
    # Setup a mock LLM that returns invalid JSON first, then valid JSON
    mock_llm = MagicMock()
    mock_llm.get_messages.return_value = []
    mock_llm.prompt_template.return_value = {"role": "system", "content": "test"}
    mock_llm.generate_text.side_effect = [
        '{"error_analysis": "Analysis"}',  # Missing improved_thought_process - invalid
        '{"error_analysis": "Better analysis", "improved_thought_process": "Good thoughts", "predicted_label": 1}'  # Valid
    ]
    
    # Create a temporary file path
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
    temp_file.close()
    
    result = generate_reflection(
        id="test-retry",
        premise="A premise",
        hypothesis="A hypothesis",
        thought_process="Original thoughts",
        predicted_label=0,
        true_label=1,
        llm=mock_llm,
        model_name="mock-model",
        json_filepath=temp_file.name,
        max_retries=2
    )
    
    # Verify result is the second (valid) response
    assert result is not None
    assert result["error_analysis"] == "Better analysis"
    assert result["improved_thought_process"] == "Good thoughts"
    assert result["predicted_label"] == 1
    
    # Verify LLM was called twice
    assert mock_llm.generate_text.call_count == 2
    
    # Clean up temp file
    os.unlink(temp_file.name)


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_generate_reflection_handles_rate_limits():
    """Test that reflection service handles rate limiting with exponential backoff"""
    # Setup a mock LLM that raises rate limit exceptions, then succeeds
    mock_llm = MagicMock()
    mock_llm.get_messages.return_value = []
    mock_llm.prompt_template.return_value = {"role": "system", "content": "test"}
    
    # First two calls raise rate limit errors, third succeeds
    mock_llm.generate_text.side_effect = [
        Exception("Rate limit exceeded: 429 Too Many Requests"),
        Exception("Rate limit exceeded"),
        '{"error_analysis": "Analysis after rate limit", "improved_thought_process": "Better thoughts", "predicted_label": 1}'
    ]
    
    # Create a temporary file path
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
    temp_file.close()
    
    # Mock time.sleep to avoid waiting during tests
    with patch('time.sleep') as mock_sleep:
        start_time = time.time()
        
        result = generate_reflection(
            id="test-rate-limit",
            premise="A premise",
            hypothesis="A hypothesis",
            thought_process="Original thoughts",
            predicted_label=0,
            true_label=1,
            llm=mock_llm,
            model_name="mock-model",
            json_filepath=temp_file.name,
            max_retries=3
        )
        
        end_time = time.time()
        
        # Verify sleep was called for backoff
        assert mock_sleep.call_count == 2
        # Check that exponential backoff was applied - first 2 seconds, then 4 seconds
        assert mock_sleep.call_args_list[0][0][0] == 2  # First backoff
        assert mock_sleep.call_args_list[1][0][0] == 4  # Second backoff
    
    # Verify final result is correct
    assert result is not None
    assert result["error_analysis"] == "Analysis after rate limit"
    assert result["improved_thought_process"] == "Better thoughts"
    assert result["predicted_label"] == 1
    
    # Verify LLM was called three times
    assert mock_llm.generate_text.call_count == 3
    
    # Clean up temp file
    os.unlink(temp_file.name)


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_generate_reflection_handles_exceptions():
    """Test that reflection service gracefully handles exceptions"""
    # Setup a mock LLM that raises an exception
    mock_llm = MagicMock()
    mock_llm.get_messages.return_value = []
    mock_llm.prompt_template.return_value = {"role": "system", "content": "test"}
    mock_llm.generate_text.side_effect = Exception("API error")
    
    # Create a temporary file path
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False)
    temp_file.close()
    
    result = generate_reflection(
        id="test-exception",
        premise="A premise",
        hypothesis="A hypothesis",
        thought_process="Original thoughts",
        predicted_label=0,
        true_label=1,
        llm=mock_llm,
        model_name="mock-model",
        json_filepath=temp_file.name,
        max_retries=1
    )
    
    # Verify error is returned
    assert result is not None
    assert "error" in result
    assert result["predicted_label"] == 1  # Should default to true_label on failure
    
    # Clean up temp file
    os.unlink(temp_file.name)


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_integration_generate_reflection(mistral_client, sample_premise, sample_hypothesis, 
                                        sample_thought_process, temp_jsonl_file):
    """Integration test for generate_reflection with actual API call"""
    # This test will make an actual API call if MISTRAL_API_KEY is set
    # Skip if running offline or if key is not available
    
    result = generate_reflection(
        id="test-integration",
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        thought_process=sample_thought_process,
        predicted_label=0,  # Intentionally wrong to test reflection
        true_label=1,
        llm=mistral_client,
        model_name="open-mistral-7b",
        json_filepath=temp_jsonl_file,
        max_retries=1  # Just do one attempt to speed up test
    )
    
    # Check result structure (if we get a valid response)
    if result and isinstance(result, dict) and "error" not in result:
        assert "error_analysis" in result
        assert "improved_thought_process" in result
        assert isinstance(result["error_analysis"], str)
        assert isinstance(result["improved_thought_process"], str)
        if "predicted_label" in result:
            assert result["predicted_label"] in [0, 1] 