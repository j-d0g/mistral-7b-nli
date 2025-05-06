---
language:
- en
license: apache-2.0
---

# Mistral-7B NLI Chain-of-Thought Dataset

<div align="center">
  <img src="metrics/dataset_banner.png" alt="NLI Dataset Banner" width="600"/>
  <p><i>Placeholder for dataset banner - generated after running metrics script</i></p>
</div>

## Dataset Description

This dataset was created as part of a university assignment at The University of Manchester for fine-tuning language models on Natural Language Inference (NLI) tasks with a focus on Chain-of-Thought (CoT) reasoning. It combines premise-hypothesis pairs with detailed reasoning chains that lead to binary entailment classifications.

### Dataset Summary

- **Task Type**: Natural Language Inference with Chain-of-Thought Reasoning
- **Languages**: English
- **Size**: [TOTAL_EXAMPLES] examples (approximately 30,000)
- **Format**: JSONL with premise, hypothesis, reasoning chain, and label
- **License**: Apache 2.0
- **Assignment Context**: Developed at The University of Manchester

## Dataset Creation

### Source Data

The dataset was created from a collection of premise-hypothesis pairs with binary entailment labels (entailment/no-entailment). Unlike some publicly available datasets that use ternary classification (entailment/neutral/contradiction), this dataset uses a simplified binary approach focused on whether the hypothesis logically follows from the premise.

### Data Collection and Augmentation Pipeline

<div align="center">
  <img src="metrics/data_pipeline.png" alt="Data Pipeline" width="800"/>
  <p><i>Placeholder for data pipeline visualization - generated after running metrics script</i></p>
</div>

The dataset creation involved three key phases:

```
┌─────────────┐     ┌─────────────────┐     ┌───────────────────┐     ┌────────────────┐
│ Original    │     │ Original        │     │ Reflected         │     │ Fine-tuning    │
│ Data (CSV)  │────▶│ Thoughts (JSON) │────▶│ Thoughts (JSON)   │────▶│ Data (JSONL)   │
└─────────────┘     └─────────────────┘     └───────────────────┘     └────────────────┘
```

#### 1. Base Data Preparation

The original dataset was split into training ([TRAIN_PERCENTAGE]%), validation ([DEV_PERCENTAGE]%), and test ([TEST_PERCENTAGE]%) sets with balanced label distribution:

- **Training set**: [TRAIN_EXAMPLES] examples used for fine-tuning
- **Validation set**: [DEV_EXAMPLES] examples used for hyperparameter tuning
- **Test set**: [TEST_EXAMPLES] examples reserved for final evaluation before submission to hidden test set.

#### 2. Thought Generation

For each premise-hypothesis pair, we used the base Mistral-7B model to generate Chain-of-Thought reasoning paths:

```
For premise: "All birds can fly." and hypothesis: "Penguins can fly."
Generate a step-by-step reasoning path to determine if the hypothesis is entailed by the premise.
```

This step resulted in detailed reasoning chains that break down the inference process into logical steps.

#### 3. Reflection Generation

<div align="center">
  <img src="metrics/reflection_process.png" alt="Reflection Process" width="700"/>
  <p><i>Placeholder for reflection process visualization - generated after running metrics script</i></p>
</div>

To enhance reasoning quality, we implemented a novel reflection mechanism:

1. Generated initial thoughts for each example
2. Prompted the model to reflect on those thoughts with:
   ```
   Review the reasoning above. Is there any flaw or oversight in the logic?
   Could the reasoning be improved? Is the conclusion correct?
   ```
3. Refined reasoning chains based on these reflections

### Data Processing

The final dataset underwent several processing steps:

1. **Formatting**: Structured as JSONL with the following fields:
   - `premise`: The premise statement
   - `hypothesis`: The hypothesis to evaluate
   - `thought_process`: Generated reasoning chain
   - `predicted_label`: Binary label (1 for entailment, 0 for non-entailment)
   - `reflection`: Self-critique of the reasoning (included in training)

2. **Quality Filtering**:
   - Removed examples with inconsistencies between reasoning and label
   - Eliminated truncated or incomplete reasoning chains
   - Filtered out examples with circular reasoning

3. **Dataset Splitting**:
   - 90% training, 5% validation, 5% test
   - Stratified by label to maintain balance

## Dataset Structure

### Data Fields

```json
{
  "premise": "All birds can fly.",
  "hypothesis": "Penguins can fly.",
  "thought_process": "Let me analyze this carefully. The premise states 'All birds can fly,' which is a universal statement about birds. However, this premise is actually factually incorrect in reality. Penguins are birds that cannot fly - they have wings that evolved for swimming instead of flying. So if I accept the premise as true within this logical world (that all birds can fly), and I know that penguins are birds, then in this context I would have to conclude that penguins can fly. However, I need to be careful here. I'm not being asked about the real world, but whether the hypothesis logically follows from the premise. Since penguins are birds, and the premise states all birds can fly, then the hypothesis 'Penguins can fly' would be entailed by the premise.",
  "predicted_label": 1,
  "reflection": "My reasoning is correct. I correctly identified that while the premise is factually incorrect in reality (not all birds can fly), I need to evaluate the logical entailment given the premise. Since penguins are birds and the premise states all birds can fly, the hypothesis is entailed by the premise within this logical world."
}
```

### Dataset Statistics

<div align="center">
  <img src="metrics/dataset_statistics.png" alt="Dataset Statistics" width="600"/>
  <p><i>Placeholder for dataset statistics visualization - generated after running metrics script</i></p>
</div>

| Metric | Value |
|--------|-------|
| Total Examples | [TOTAL_EXAMPLES] |
| Training Set | [TRAIN_EXAMPLES] ([TRAIN_PERCENTAGE]%) |
| Validation Set | [DEV_EXAMPLES] ([DEV_PERCENTAGE]%) |
| Test Set | [TEST_EXAMPLES] ([TEST_PERCENTAGE]%) |
| Entailment Examples | [ENTAILMENT_EXAMPLES] ([ENTAILMENT_PERCENTAGE]%) |
| Non-entailment Examples | [NO_ENTAILMENT_EXAMPLES] ([NO_ENTAILMENT_PERCENTAGE]%) |

### Token Length Analysis

<div align="center">
  <img src="metrics/token_lengths.png" alt="Token Lengths" width="700"/>
  <p><i>Placeholder for token length visualization - generated after running metrics script</i></p>
</div>

| Component | Average Tokens | Min | Max | Median |
|-----------|----------------|-----|-----|--------|
| Premise | [PREMISE_AVG_TOKENS] | [PREMISE_MIN_TOKENS] | [PREMISE_MAX_TOKENS] | [PREMISE_MEDIAN_TOKENS] |
| Hypothesis | [HYPOTHESIS_AVG_TOKENS] | [HYPOTHESIS_MIN_TOKENS] | [HYPOTHESIS_MAX_TOKENS] | [HYPOTHESIS_MEDIAN_TOKENS] |
| Reasoning Chain | [THOUGHT_AVG_TOKENS] | [THOUGHT_MIN_TOKENS] | [THOUGHT_MAX_TOKENS] | [THOUGHT_MEDIAN_TOKENS] |
| Reflection | [REFLECTION_AVG_TOKENS] | [REFLECTION_MIN_TOKENS] | [REFLECTION_MAX_TOKENS] | [REFLECTION_MEDIAN_TOKENS] |

### Evaluation Metrics

The dataset and resulting models are evaluated using:

| Metric | Description |
|--------|-------------|
| Accuracy | Percentage of correctly classified examples |
| Precision | True positives / (True positives + False positives) |
| Recall | True positives / (True positives + False negatives) |
| F1 Score | Harmonic mean of precision and recall |
| Thought Quality | Manual evaluation of reasoning coherence (1-5 scale) |
| Average Token Length | Distribution of token lengths in generated reasoning |

## Dataset Creation Rationale

This dataset was specifically designed as part of a university assignment to address several limitations in existing NLI training:

1. **Transparency**: Standard NLI tasks often lack visibility into model reasoning
2. **Robustness**: Models trained on classification alone may rely on spurious correlations
3. **Error Analysis**: Without reasoning chains, it's difficult to understand model failures
4. **Educational Value**: The assignment aimed to demonstrate how Chain-of-Thought reasoning can improve model performance and explainability

By pairing premise-hypothesis examples with explicit reasoning chains, models trained on this dataset learn to:
- Break down complex inference problems into steps
- Apply logical rules systematically
- Consider multiple perspectives before reaching a conclusion
- Explain their decision-making process to users

<div align="center">
  <img src="metrics/reasoning_benefits.png" alt="Reasoning Benefits" width="700"/>
  <p><i>Placeholder for reasoning benefits visualization - generated after running metrics script</i></p>
</div>

## Considerations for Using the Data

### Social Impact of Dataset

This dataset aims to improve the reasoning capabilities of language models in NLI tasks, which has several potential positive impacts:
- Enhanced explainability in AI decision-making
- Better identification of logical fallacies
- More transparent reasoning in sensitive applications
- Educational value for users learning logic and critical thinking

### Discussion of Biases

The dataset may contain biases from:
1. **Data Distribution**: Despite balancing for labels, certain reasoning patterns may be overrepresented
2. **Generation Model**: Reasoning chains reflect biases in the base Mistral-7B model
3. **Academic Context**: Created as a university assignment, so may reflect academic reasoning styles

### Other Known Limitations

1. **English-Only**: The dataset is limited to English language examples
2. **Limited Topics**: May not cover all domains or specialized knowledge areas
3. **Reasoning Style**: Demonstrates a particular approach to reasoning that may not be universally optimal
4. **Perfect Information Assumption**: Examples assume all relevant information is contained in the premise

## Additional Information

### Dataset Curators

This dataset was created as a university assignment at The University of Manchester.

### Citation Information

If you use this dataset, please cite:

```bibtex
@misc{mistral-nli-thoughts-dataset,
  author = {Your Name},
  title = {Mistral-7B NLI Chain-of-Thought Dataset},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {University of Manchester Assignment}
}
``` 

---

*This dataset card was created as part of a university assignment at The University of Manchester. The metrics reported were generated using the `generate_card_metrics.py` script.* 