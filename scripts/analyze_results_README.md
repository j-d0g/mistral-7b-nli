# Analyze Results

This utility script generates visualizations and analyzes results from Natural Language Inference (NLI) chain-of-thought experiments.

## Purpose

The `analyze_results.py` script processes JSON result files produced by the thought generation experiments and creates:

1. **Visualizations** - Various plots to understand model performance and trends
2. **Analysis Tables** - Detailed tables with examples and statistical analysis
3. **Data Exports** - CSV files for further analysis

## Usage

### Command Line

```bash
python scripts/analyze_results.py --results-json data/original_thoughts/sample_thoughts.json --output-dir analysis_output
```

### In a Jupyter Notebook

The script is designed to be easily used in Jupyter notebooks. Here's a sample notebook usage:

```python
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from scripts.analyze_results import (
    load_results, 
    prepare_dataframe,
    plot_accuracy_by_token_count,
    plot_token_count_distribution,
    plot_confusion_matrix,
    plot_metrics_summary,
    plot_thought_length_vs_accuracy,
    create_examples_table,
    create_sample_thoughts_table,
    analyze_common_phrases
)

# Load your results
json_path = "data/original_thoughts/sample_thoughts.json"
metadata, results = load_results(json_path)

# Prepare the data
df = prepare_dataframe(results)

# Generate visualizations
plot_accuracy_by_token_count(df)
plot_token_count_distribution(df)
plot_confusion_matrix(df)
plot_metrics_summary(df)
plot_thought_length_vs_accuracy(df)

# Create analysis tables
examples = create_examples_table(df)
samples = create_sample_thoughts_table(df)
phrases = analyze_common_phrases(df)

# Display some interesting examples
display(examples.head())

# Analyze correlation between token count and accuracy
correlation = df['token_count'].corr(df['is_correct'].astype(int))
print(f"Correlation between token count and accuracy: {correlation:.4f}")
```

## Visualizations

The script generates the following visualizations:

### 1. Accuracy by Token Count Range

Shows how accuracy varies across different token count quartiles.

![Accuracy by Token Count Range](example_imgs/accuracy_by_token_count.png)

### 2. Token Count Distribution

Histogram showing the distribution of token counts, with correct/incorrect predictions differentiated.

![Token Count Distribution](example_imgs/token_count_distribution.png)

### 3. Confusion Matrix

Shows true positives, false positives, true negatives, and false negatives.

![Confusion Matrix](example_imgs/confusion_matrix.png)

### 4. Metrics Summary

Bar chart showing accuracy, precision, recall, and F1 score.

![Metrics Summary](example_imgs/metrics_summary.png)

### 5. Thought Length vs. Accuracy

Scatter plot with trend lines showing the relationship between thought length and prediction accuracy.

![Thought Length vs. Accuracy](example_imgs/thought_length_vs_accuracy.png)

## Analysis Tables

The script also generates the following analysis tables:

### 1. Example Predictions

Sample predictions across different categories (correct entailment, incorrect contradiction, etc.).

### 2. Sample Thoughts

Examples of thought processes from different token count quartiles.

### 3. Phrase Analysis

Analysis of common phrases in correct vs. incorrect thought processes.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Advanced Usage

You can import the individual functions to create custom analyses:

```python
# Custom analysis combining different visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Accuracy by token bucket with custom style
bucket_stats = df.groupby('token_bucket').agg(
    count=('is_correct', 'count'),
    accuracy=('is_correct', 'mean')
).reset_index()
bucket_stats['accuracy'] *= 100  # Convert to percentage

sns.barplot(x='token_bucket', y='accuracy', data=bucket_stats, ax=ax1, 
            palette='viridis', order=sorted(bucket_stats['token_bucket']))
ax1.set_title('Accuracy by Token Range')
ax1.set_ylim(0, 100)

# Plot 2: Accuracy for entailment vs contradiction
label_acc = df.groupby('true_label')['is_correct'].mean() * 100
label_acc.plot(kind='bar', ax=ax2, color=['#e74c3c', '#2ecc71'])
ax2.set_title('Accuracy by Label Type')
ax2.set_xticklabels(['Contradiction (0)', 'Entailment (1)'], rotation=0)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.show()
``` 