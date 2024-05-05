# Mistral-7B-GPTQ

## Description

Fine-tuning Mistral-7B on NLI tasks using an NLI dataset augmented with chain-of-thought, generated and self-improved via a recursive self-scoring pipeline using a combination of Mistral-7b, Mixtral 8x7b and Mixtral 8x22b to reduce noise in the data.

I used a 4-bit quantised model for this experiment and trained it over 10 epochs across different hyper-parameter configurations, namely LoRA rank, alpha-value, batch size and learning rate. For this experiment, I framed this as a text-completion task, where I extract the thoughts and class label from a JSON output with the schema {"thoughts": <step 1> ... <step 4>, "label": <0 for entailment, 1 for no-entailment>.
