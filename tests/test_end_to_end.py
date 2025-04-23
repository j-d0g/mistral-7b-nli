"""
End-to-end test for Pydantic validation in prediction_service and scoring_service.
This test verifies the complete workflow with real API calls.
"""
import os
import sys
import json
import tempfile
import pytest
from dotenv import load_dotenv

from llm.mistral import Mistral
from service.prediction_service import predict_label
from service.scoring_service import generate_score


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_prediction_service(sample_premise, sample_hypothesis, sample_true_label, temp_jsonl_file):
    """Test the prediction_service end-to-end"""
    print("\n======= TESTING PREDICTION SERVICE =======\n")
    
    # Load API key and create Mistral client
    mistral_client = Mistral(os.getenv('MISTRAL_API_KEY'))
    
    # Define JSON format for error messages
    json_format = {
        "thought_process": "Step by step reasoning",
        "label": "0 or 1"
    }
    
    print(f"Premise: {sample_premise}")
    print(f"Hypothesis: {sample_hypothesis}")
    print(f"True label: {sample_true_label}")
    print("\nCalling predict_label()...")
    
    # Call the prediction service
    result = predict_label(
        id="test-e2e-1",
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        true_label=sample_true_label,
        llm=mistral_client,
        model_name="open-mistral-7b",
        json_format=json_format,
        json_filepath=temp_jsonl_file,
        max_retries=2  # Reduced for testing
    )
    
    # Verify result
    assert result is not None
    assert 'label' in result
    assert 'thought_process' in result
    assert isinstance(result['label'], int)
    assert result['label'] in [0, 1]
    
    print("\n✅ Prediction service test PASSED")
    print(f"Predicted label: {result['label']}")
    print(f"Thought process: {result['thought_process'][:100]}...")


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_scoring_service(sample_premise, sample_hypothesis, sample_true_label, temp_dir):
    """Test the scoring_service end-to-end"""
    print("\n======= TESTING SCORING SERVICE =======\n")
    
    # First run prediction to get thoughts
    prediction_result = test_prediction_service(
        sample_premise, sample_hypothesis, sample_true_label, 
        tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
    )
    
    if not prediction_result:
        pytest.skip("Prediction test failed, skipping scoring test")
    
    # Load API key and create Mistral client
    mistral_client = Mistral(os.getenv('MISTRAL_API_KEY'))
    
    # Define JSON format for scoring
    json_format = {
        "score": "0-5 rating of thought quality",
        "label": "0 or 1",
        "improved_thoughts": "Improved thought chain"
    }
    
    # Define system prompt for scoring
    system_prompt = """You are a world-class expert in natural language inference and logical reasoning. You have been tasked with reviewing and scoring the quality of a thought process generated for an NLI task.

For each example, I will provide:
1. A premise statement
2. A hypothesis statement
3. A thought process from another model
4. The predicted label (0 for no entailment, 1 for entailment)

Your job is to:
1. Score the quality of the thought process on a scale from 0-5, where:
   - 0 = Completely incorrect reasoning with major logical errors
   - 1 = Poor reasoning with significant gaps or irrelevancies
   - 2 = Basic reasoning but missing important aspects
   - 3 = Adequate reasoning with minor issues
   - 4 = Good reasoning with clear logical flow
   - 5 = Excellent, comprehensive reasoning

2. Provide an improved version of the thought process that:
   - Addresses any errors or gaps in the original
   - Provides a clearer logical flow
   - Focuses on the most relevant aspects of the premise-hypothesis relationship

3. Indicate whether you agree with the predicted label (0 or 1)

Your response should be in the following JSON format:
{
  "score": <0-5 integer rating>,
  "improved_thoughts": "<Your improved step-by-step thought process>",
  "label": <0 or 1, based on your reasoning>
}
"""
    
    print(f"Testing scoring with thought process from prediction")
    print("\nCalling generate_score()...")
    
    # Call the scoring service
    result = generate_score(
        id="test-e2e-1",
        sys=system_prompt,
        premise=sample_premise,
        hypothesis=sample_hypothesis,
        thoughts=prediction_result['thought_process'],
        predicted_label=prediction_result['label'],
        true_label=sample_true_label,
        llm=mistral_client,
        model_name="open-mistral-7b",
        json_format=json_format,
        json_filepath=temp_dir,
        max_retries=2,  # Reduced for testing
        depth=0
    )
    
    # Verify result
    assert result is not None
    assert 'score' in result
    assert 'improved_thoughts' in result
    assert 'label' in result
    assert isinstance(result['score'], int)
    assert 0 <= result['score'] <= 5
    assert isinstance(result['label'], int)
    assert result['label'] in [0, 1]
    
    print("\n✅ Scoring service test PASSED")
    print(f"Score: {result['score']}")
    print(f"Label: {result['label']}")
    print(f"Improved thoughts: {result['improved_thoughts'][:100]}...")


@pytest.mark.skipif(not os.getenv('MISTRAL_API_KEY'), reason="MISTRAL_API_KEY not set")
def test_full_workflow(sample_premise, sample_hypothesis, sample_true_label, temp_jsonl_file, temp_dir):
    """Test the full workflow from prediction to scoring"""
    print("Starting end-to-end tests for Pydantic validation in NLI services")
    
    # Test prediction service
    test_prediction_service(sample_premise, sample_hypothesis, sample_true_label, temp_jsonl_file)
    
    # Test scoring service 
    test_scoring_service(sample_premise, sample_hypothesis, sample_true_label, temp_dir)
    
    print("\nEnd-to-end tests completed successfully") 