# 1. Data Augmentation Process

A key innovation in this project is leveraging Chain-of-Thought (CoT) reasoning to improve NLI classification. This section describes the process of creating high-quality CoT training data.

## The Challenge

NLI (Natural Language Inference) requires understanding the relationship between premise and hypothesis sentences. While traditional classification fine-tuning produces decent results, the model's reasoning process remains opaque. I hypothesized that:

1. Exposing the model to explicit reasoning steps would improve classification accuracy
2. Models trained with high-quality CoT would better generalize to challenging examples 
3. The resulting predictions would be more interpretable and trustworthy

## Creating Chain-of-Thought Data

I developed a multi-stage pipeline to generate and refine reasoning data:

### Stage 1: Initial Thought Generation

I used the Mistral API with the open-mistral-7b model to generate reasoning for each premise-hypothesis pair. The prompt instructed the model to:
1. Analyze the meaning of each sentence
2. Identify relationships between key concepts
3. Provide step-by-step reasoning
4. Produce a final label (0 or 1)

**Key Findings:**
- The base model achieved ~74% accuracy on the dev set
- CoT reasoning quality varied significantly between examples
- Longer, more detailed reasoning correlated with better predictions
- Most errors occurred in examples with complex negation or subtle word choice differences

Example of generated reasoning:
```
step 1: the premise states "nearby is the new, architecturally inventive supreme court building."
step 2: the hypothesis claims "the supreme court building was built in the early 1980s."
step 3: the premise describes the building as "new" which suggests recent construction
step 4: the hypothesis specifies a time period (early 1980s) that would make the building about 40 years old
step 5: a 40-year-old building would not typically be described as "new" in the 2020s
step 6: therefore, the premise does not entail the hypothesis
```

### Stage 2: Reflection and Improvement

For examples where the model's prediction was incorrect, I used a slightly stronger model (open-mistral-nemo) to reflect on the errors and generate improved reasoning. The reflection prompt included:
- The original premise and hypothesis
- The incorrect original reasoning
- The correct label
- Instructions to identify flaws and produce better reasoning

**Key Observations:**
- The reflection process improved reasoning quality significantly
- Common error corrections included:
  - Misinterpreted negations or double negatives
  - Overlooked critical details in the premise
  - Introduced external knowledge not present in the premise
  - Made unwarranted assumptions about time periods or causality

### Stage 3: Data Preparation for Different Ablation Studies

I prepared three different datasets to evaluate different training approaches:

1. **Ablation 0**: All original examples (both correct and incorrect)
2. **Ablation 1**: Only examples where the original model's prediction was correct
3. **Ablation 2**: Correct examples from original predictions + reflected examples for incorrect predictions

Ablation 2 represents our primary approach, providing the model with only high-quality reasoning examples.

## Challenges and Solutions

### API Reliability
- The Mistral API occasionally returned errors or timeouts
- **Solution**: Implemented robust retry mechanisms and parallel processing

### Reasoning Format Consistency
- Models sometimes generated inconsistent JSON formatting
- **Solution**: Developed a resilient parsing system to handle variations

### Thought Process Quality
- Some generated reasoning was superficial or logically flawed
- **Solution**: Used the reflection process to correct errors and improve quality

## Key Metrics

| Dataset | Total Examples | Original Correct | Reflected Examples | Final Accuracy |
|---------|----------------|------------------|-------------------|----------------|
| Train   | 5,000          | 3,720 (74.4%)    | 1,280 (25.6%)     | 100%           |
| Dev     | 1,000          | 745 (74.5%)      | 255 (25.5%)       | 100%           |
| Test    | 1,977          | N/A              | N/A               | N/A (hidden)   |

> **Note:** For those who want to skip the data augmentation process and use the pre-generated datasets, see the [Downloading Datasets](#downloading-datasets) section. 