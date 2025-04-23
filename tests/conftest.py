"""
Fixtures for pytest.

This module contains fixtures that can be used across all tests.
"""
import os
import json
import tempfile
import pytest
from dotenv import load_dotenv

from llm.mistral import Mistral
from models.response_models import NLIResponse, ScoringResponse


@pytest.fixture
def sample_premise():
    """Sample premise for testing."""
    return "The dog chased the cat up the tree."


@pytest.fixture
def sample_hypothesis():
    """Sample hypothesis for testing."""
    return "The cat climbed the tree."


@pytest.fixture
def sample_true_label():
    """Sample true label for testing."""
    return 1  # Entailment


@pytest.fixture
def sample_thought_process():
    """Sample thought process for testing."""
    return "Step 1: The premise describes a scenario where a dog chases a cat up a tree. " \
           "Step 2: For a cat to go up a tree, it would need to climb. " \
           "Step 3: Therefore, the hypothesis is entailed by the premise."


@pytest.fixture
def sample_response_json():
    """Sample response JSON for testing."""
    return {
        "thought_process": "Step 1: Analysis of the premise... Step 2: Comparison... Step 3: Conclusion...",
        "label": 1
    }


@pytest.fixture
def sample_scoring_json():
    """Sample scoring JSON for testing."""
    return {
        "score": 4,
        "label": 1,
        "improved_thoughts": "Step 1: Better analysis... Step 2: Better comparison... Step 3: Better conclusion..."
    }


@pytest.fixture
def mistral_client():
    """Create a Mistral client for testing."""
    load_dotenv()
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not found in .env file")
    return Mistral(api_key)


@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSONL file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
        tmp_name = tmp.name
    
    yield tmp_name
    
    # Cleanup
    if os.path.exists(tmp_name):
        os.unlink(tmp_name)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    # We don't need to clean up as we're not writing anything 