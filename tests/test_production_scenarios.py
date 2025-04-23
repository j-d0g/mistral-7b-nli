"""
Tests for production-critical scenarios in the NLI services.

These tests focus on retry logic, model switching, and depth termination
which are critical when processing large volumes of examples.
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch, call

from llm.base_llm import BaseLLM
from service.scoring_service import generate_score
from models.response_models import ScoringResponse


class MockLLM(BaseLLM):
    """Mock LLM client for testing with controlled responses."""
    
    def __init__(self, responses=None):
        """
        Initialize with predefined responses.
        
        Args:
            responses: List of strings to return from generate_text in sequence
        """
        super().__init__()
        self.responses = responses or []
        self.response_index = 0
        self.calls = []
        self.models_used = []
    
    def get_models(self):
        return ["mock-model-1", "mock-model-2"]
    
    def generate_text(self, model_name="mock-model", max_tokens=100, temperature=0.7, top_p=1.0):
        """Return next response from the sequence."""
        self.calls.append((model_name, max_tokens, temperature, top_p))
        self.models_used.append(model_name)
        
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        return '{"score": 3, "label": 1, "improved_thoughts": "Default mock response"}'
    
    def prompt_template(self, role, message):
        """Mock prompt template."""
        return {"role": role, "content": message}


# Use separate patching to prevent tests from affecting each other
@patch('service.scoring_service.generate_score', side_effect=lambda *args, **kwargs: {'score': 0, 'label': 1, 'improved_thoughts': 'Patched recursion'})
def test_retry_logic(_):
    """Test that the scoring service properly retries when validation fails."""
    # Setup mock LLM with invalid responses followed by valid response
    responses = [
        # Invalid JSON - should trigger retry
        "This is not valid JSON",
        # Invalid format (missing field) - should trigger retry
        '{"score": 3, "label": 1}',
        # Valid response - should succeed
        '{"score": 3, "label": 1, "improved_thoughts": "Valid response after retries"}'
    ]
    mock_llm = MockLLM(responses)
    
    # Test parameters
    temp_dir = tempfile.mkdtemp()
    json_format = {"score": "rating", "label": "class", "improved_thoughts": "analysis"}
    
    # Call the function under test with recursion patched out
    with patch('service.scoring_service.generate_score', return_value={"score": 0, "label": 1, "improved_thoughts": "Patched recursion"}):
        result = generate_score(
            id="test-retry",
            sys="Test system prompt",
            premise="Test premise",
            hypothesis="Test hypothesis",
            thoughts="Initial thoughts",
            predicted_label=1,
            true_label=1,
            llm=mock_llm,
            model_name="mock-model",
            json_format=json_format,
            json_filepath=temp_dir,
            max_retries=5,
            depth=0
        )
    
    # Verify function made correct number of calls to generate_text
    assert len(mock_llm.calls) == 3, f"Expected 3 calls, got {len(mock_llm.calls)}"


def test_model_switching():
    """Test that the service switches models at the proper retry thresholds."""
    # Create invalid responses to trigger model switching
    responses = ["Invalid JSON"] * 15  # Many invalid responses to trigger switches
    
    mock_llm = MockLLM(responses)
    
    # Test parameters
    temp_dir = tempfile.mkdtemp()
    json_format = {"score": "rating", "label": "class", "improved_thoughts": "analysis"}
    
    # Call the function under test, but mock the recursive calls and persist_benchmarks
    with patch('service.scoring_service.generate_score', return_value={"score": 0, "label": 1, "improved_thoughts": "Mocked recursion"}), \
         patch('service.scoring_service.persist_benchmarks'):
        result = generate_score(
            id="test-switch",
            sys="Test system prompt",
            premise="Test premise",
            hypothesis="Test hypothesis",
            thoughts="Initial thoughts",
            predicted_label=1,
            true_label=1,
            llm=mock_llm,
            model_name="initial-model",
            json_format=json_format,
            json_filepath=temp_dir,
            max_retries=10,  # Sufficient to trigger model switch
            depth=0
        )
    
    # Verify model switching occurred at correct thresholds
    actual_models = mock_llm.models_used[:10]  # Look at first 10 calls
    
    # The actual indexing in the code is on (retry + 1) % 6 == 0, so switching happens at index 5
    assert "open-mixtral-8x22b" in actual_models, "Should have switched to open-mixtral-8x22b"
    assert "mistral-large-latest" in actual_models, "Should have switched to mistral-large-latest"
    
    # Check exact sequence if needed:
    switching_indexes = [i for i, model in enumerate(actual_models) if model != "initial-model" and (i == 0 or model != actual_models[i-1])]
    assert 5 in switching_indexes, "Should switch model at retry 5 (index 5)"
    mistral_large_index = next((i for i, model in enumerate(actual_models) if model == "mistral-large-latest"), None)
    assert mistral_large_index is not None and mistral_large_index >= 8, "Should switch to mistral-large-latest at retry 8 (index 8)"


def test_depth_termination():
    """Test that recursion stops when depth exceeds maximum."""
    # Create a mock LLM that returns a response with score < 4, which would normally trigger recursion
    mock_llm = MockLLM(['{"score": 3, "label": 1, "improved_thoughts": "This would normally trigger recursion"}'])
    
    # Test parameters
    temp_dir = tempfile.mkdtemp()
    json_format = {"score": "rating", "label": "class", "improved_thoughts": "analysis"}
    
    # Critical: patch the recursive call to generate_score to prevent actual recursion
    with patch('service.scoring_service.print') as mock_print, \
         patch('service.scoring_service.generate_score', side_effect=lambda *args, **kwargs: {'score': 0, 'label': 1, 'improved_thoughts': 'Mocked recursive call'}), \
         patch('service.scoring_service.persist_benchmarks'):
        
        result = generate_score(
            id="test-depth",
            sys="Test system prompt",
            premise="Test premise",
            hypothesis="Test hypothesis",
            thoughts="Initial thoughts",
            predicted_label=1,
            true_label=1,
            llm=mock_llm,
            model_name="mock-model",
            json_format=json_format,
            json_filepath=temp_dir,
            max_retries=3,
            depth=4  # Exceeds maximum depth (3)
        )
    
    # Verify output includes MAX DEPTH REACHED
    mock_print.assert_any_call("****** MAX DEPTH REACHED ******\n")
    
    # With proper patching, we should have only one generate_text call
    assert len(mock_llm.calls) == 1, f"Expected exactly 1 API call, got {len(mock_llm.calls)}"


def test_file_persistence():
    """Test that benchmarks are correctly persisted to the specified paths."""
    # Create a mock LLM that returns a high-scoring response
    mock_llm = MockLLM(['{"score": 4, "label": 1, "improved_thoughts": "High quality thoughts"}'])
    
    # Create temporary directory structure for testing persistence
    temp_base = tempfile.mkdtemp()
    
    # Call the function to trigger persistence
    with patch('builtins.open', create=True) as mock_open, \
         patch('json.dump') as mock_json_dump, \
         patch('service.scoring_service.generate_score', return_value={"score": 0, "label": 1, "improved_thoughts": "Mocked recursion"}):
        
        result = generate_score(
            id="test-persistence",
            sys="Test system prompt",
            premise="Test premise",
            hypothesis="Test hypothesis",
            thoughts="Initial thoughts",
            predicted_label=1,
            true_label=1,
            llm=mock_llm,
            model_name="mock-model",
            json_format={"score": "rating"},
            json_filepath=temp_base,
            max_retries=1,
            depth=0
        )
    
    # Verify the file was opened with the correct path (gold_standard.json since score >= 4)
    expected_path = f'{temp_base}/gold_standard.json'
    mock_open.assert_called_with(expected_path, 'a')
    
    # Verify that json.dump was called to persist the data
    assert mock_json_dump.called, "json.dump should have been called to persist data"


def test_max_retries_exceeded():
    """Test behavior when max retries are exceeded."""
    # Create a mock LLM that always returns invalid responses
    mock_llm = MockLLM(['This is not valid JSON'] * 20)
    
    # Test parameters
    temp_dir = tempfile.mkdtemp()
    json_format = {"score": "rating", "label": "class", "improved_thoughts": "analysis"}
    
    # Call the function with limited retries and prevent actual recursion
    with patch('service.scoring_service.generate_score', side_effect=lambda *args, **kwargs: {'score': 0, 'label': 1, 'improved_thoughts': 'Patched recursion'}), \
         patch('service.scoring_service.persist_benchmarks'):
        result = generate_score(
            id="test-max-retries",
            sys="Test system prompt",
            premise="Test premise",
            hypothesis="Test hypothesis",
            thoughts="Initial thoughts",
            predicted_label=1,
            true_label=1,
            llm=mock_llm,
            model_name="mock-model",
            json_format=json_format,
            json_filepath=temp_dir,
            max_retries=3,  # Low retry limit to trigger failure
            depth=0
        )
    
    # Verify function made exactly max_retries calls
    assert len(mock_llm.calls) == 3, f"Expected exactly 3 API calls, got {len(mock_llm.calls)}"
    
    # Verify result indicates exceeded retries
    assert result["score"] == 0
    assert "exceeded maximum retries" in result["improved_thoughts"].lower() 