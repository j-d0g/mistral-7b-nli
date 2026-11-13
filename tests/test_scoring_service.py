"""
Tests for the scoring service.
"""
import pytest
import os
from unittest.mock import MagicMock

from service.scoring_service import generate_score, validate_response


@pytest.mark.parametrize("api_response,expected_valid", [
    ('{"score": 4, "predicted_label": 1, "improved_thought_process": "Better analysis"}', True),
    ('{"score": 4, "predicted_label": "1", "improved_thought_process": "Analysis"}', False),  # String label
    ('{"score": "4", "predicted_label": 1, "improved_thought_process": "Analysis"}', False),  # String score
    ('{"predicted_label": 1, "improved_thought_process": "Analysis"}', False),  # Missing score
    ('{"score": 4, "predicted_label": 1}', False),  # Missing improved_thought_process
    ('{"score": 6, "predicted_label": 1, "improved_thought_process": "Analysis"}', False),  # Invalid score value
    ('{invalid json}', False),  # Invalid JSON
])
def test_validate_response(api_response, expected_valid):
    """Test validation of scoring API responses"""
    json_format = {
        "score": "0-5 rating",
        "predicted_label": "0 or 1",
        "improved_thought_process": "Improved reasoning"
    }
    
    result, error = validate_response(api_response, json_format)
    
    if expected_valid:
        assert result is not None
        assert error is None
        assert "score" in result
        assert "predicted_label" in result
        assert "improved_thought_process" in result
    else:
        assert result is None
        assert error is not None


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_generate_score_with_mock_llm(sample_premise, sample_hypothesis, sample_thought_process):
    """Test the scoring function with a mocked LLM client"""
    # Setup a mock LLM that returns a valid response
    mock_llm = MagicMock()
    mock_llm.get_messages.return_value = []
    mock_llm.prompt_template.return_value = {"role": "system", "content": "test"}
    mock_llm.generate_text.return_value = '{"score": 4, "predicted_label": 1, "improved_thought_process": "Better analysis"}'
    
    # Create a temporary directory path
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Define system prompt for scoring
    system_prompt = """You are a world-class expert in natural language inference and logical reasoning."""
    
    result = generate_score(
        id="test-1",
        sys=system_prompt,
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        thoughts=sample_thought_process,
        predicted_label=1,
        true_label=1,
        llm=mock_llm,
        model_name="mock-model",
        json_format={"score": "0-5", "predicted_label": "0 or 1", "improved_thought_process": "analysis"},
        json_filepath=temp_dir,
        max_retries=1,
        depth=0
    )
    
    # Verify result
    assert result is not None
    assert result["score"] == 4
    assert result["predicted_label"] == 1
    assert result["improved_thought_process"] == "better analysis"
    
    # Verify LLM was called correctly
    mock_llm.generate_text.assert_called_once()
    mock_llm.add_messages.assert_called()


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_integration_generate_score(mistral_client, sample_premise, sample_hypothesis, 
                                   sample_thought_process, temp_dir):
    """Integration test for generate_score with actual API call"""
    # This test will make an actual API call if MISTRAL_API_KEY is set
    # Skip if running offline or if key is not available
    
    json_format = {
        "score": "0-5 rating",
        "predicted_label": "0 or 1",
        "improved_thought_process": "Improved reasoning"
    }
    
    # Define system prompt for scoring
    system_prompt = """You are a world-class expert in natural language inference and logical reasoning. 
    Rate the quality of the thought process on a scale of 0-5, where 5 is excellent.
    Provide your response in JSON format with score, label, and improved_thought_process fields."""
    
    result = generate_score(
        id="test-integration",
        sys=system_prompt,
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        thoughts=sample_thought_process,
        predicted_label=1,
        true_label=1,
        llm=mistral_client,
        model_name="open-mistral-7b",
        json_format=json_format,
        json_filepath=temp_dir,
        max_retries=1,  # Just do one attempt to speed up test
        depth=0
    )
    
    # Check result structure (if we get a valid response)
    if result and isinstance(result, dict):
        assert "score" in result
        assert "predicted_label" in result
        assert "improved_thought_process" in result
        assert isinstance(result["score"], int)
        assert 0 <= result["score"] <= 5
        assert isinstance(result["predicted_label"], int)
        assert result["predicted_label"] in [0, 1] 