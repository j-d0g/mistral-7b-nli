# NLIstral-7B-QLoRA: Interpretable NLI with Augmented Chain-of-Thought Fine-Tuning

## Abstract

This paper describes our approach to fine-tuning the Mistral-7B language model for Natural Language Inference (NLI) tasks with Chain-of-Thought (CoT) reasoning. We introduce a novel Reflection-CoT mechanism that improves upon naive data augmentation by addressing label disagreement in a targeted manner. Our approach not only produces accurate classifications but also generates interpretable reasoning processes that explain the model's decisions. By combining parameter-efficient QLoRA fine-tuning with our multi-stage data augmentation pipeline, we create a system that balances performance, efficiency, and transparency for NLI tasks.

## Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
   - 1.1. [Problem Statement](#problem-statement)
   - 1.2. [Motivation & Background](#motivation-and-background)
   - 1.3. [Overview of Architecture](#overview-of-architecture)
   - 1.4. [Contributions](#contributions)
2. [Related Work](#related-work)
   - 2.1. [Natural Language Inference Techniques](#natural-language-inference-techniques)
   - 2.2. [Chain-of-Thought Reasoning](#chain-of-thought-reasoning)
   - 2.3. [Synthetic Data Generation and Augmentation for NLP](#synthetic-data-generation-and-augmentation-for-nlp)
   - 2.4. [LLM Self-Correction and Reflection Mechanisms](#llm-self-correction-and-reflection-mechanisms)
   - 2.5. [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
3. [Methodology](#methodology)
   - 3.1. [Synthetic Data Augmentation](#synthetic-data-augmentation)
   - 3.2. [Model Architecture](#model-architecture)
   - 3.3. [Training Approach](#training-approach)
4. [Experiment I - Synthetic Data Augmentation](#experiment-i---synthetic-data-augmentation)
   - 4.1. [Initial Experiments: Learning From Failure](#initial-experiments-learning-from-failure)
   - 4.2. [Less-Is-All-You-Need: Improved Thought Generation](#less-is-all-you-need-improved-thought-generation)
   - 4.3. [LLM-As-A-Judge: Iterative Self-Critique & Improvement](#llm-as-a-judge-iterative-self-critique--improvement)
   - 4.4. [Learn-From-Your-Mistakes: Self-Reflection & Correction](#learn-from-your-mistakes-self-reflection--correction)
   - 4.5. [Dataset Composition](#dataset-composition)
5. [Experiment II - Fine-Tuning with QLoRA](#experiment-ii---fine-tuning-with-qlora)
   - 5.1. [Dockerized Training Environment](#dockerized-training-environment)
   - 5.2. [Mistral-7B: 4-bit Quantized Model](#mistral-7b-4-bit-quantized-model)
   - 5.3. [Parameter-Efficient Fine-Tuning with QLoRA](#parameter-efficient-fine-tuning-with-qlora)
   - 5.4. [Ablation Studies & Hyper-Parameter Tuning](#ablation-studies--hyper-parameter-tuning)
6. [Results](#results)
   - 6.1. [Baseline Performance](#baseline-performance)
   - 6.2. [Fine-Tuned Performance](#fine-tuned-performance)
   - 6.3. [Benchmarks](#benchmarks)
   - 6.4. [Thought Quality](#thought-quality)
7. [Discussion](#discussion)
   - 7.1. [Model Bias](#model-bias)
   - 7.2. [Labeller Bias](#labeller-bias)
   - 7.3. [Trade-Offs and Assumptions](#trade-offs-and-assumptions)
8. [Limitations and Future Work](#limitations-and-future-work)
   - 8.1. [Self-Consistency](#self-consistency)
   - 8.2. [Tree-of-Thought](#tree-of-thought)
   - 8.3. [RLHF Optimization](#rlhf-optimization)
   - 8.4. [Refining Scoring](#refining-scoring)
   - 8.5. [Full Fine-Tuning](#full-fine-tuning)
9. [Conclusion](#conclusion)
10. [References](#references)

## Introduction and Motivation

Natural Language Inference (NLI) is a fundamental NLP task that tests a model's ability to reason about the relationship between text passages. While modern language models have achieved high accuracy on NLI benchmarks, they often function as black boxes, making decisions without providing insight into their reasoning process. This limits both trustworthiness and opportunities for improvement.

### Problem Statement

Natural Language Inference (NLI) involves determining whether a "hypothesis" sentence can be logically inferred from a "premise" sentence. This task is challenging due to:
- Linguistic ambiguity requiring nuanced understanding of language
- The need for implicit world knowledge not directly stated in the texts
- Complex logical reasoning between statements that may appear unrelated
- Potential subjectivity in edge cases where human annotators might disagree

In this work, we focus on binary NLI classification—determining whether a hypothesis is entailed (1) by a premise or not (0)—while simultaneously generating explanatory reasoning.

### Motivation & Background

Chain-of-Thought (CoT) reasoning has emerged as a powerful technique for improving both the performance and interpretability of language models on complex tasks. By encouraging models to generate step-by-step reasoning before producing a final answer, CoT prompting provides several advantages particularly relevant to NLI:

- It makes the model's reasoning transparent, allowing users to understand *why* certain inferences were made
- It provides a structured approach to breaking down complex logical relationships
- It serves as a natural error analysis tool, enabling identification of specific reasoning failures
- It can help models avoid logical fallacies and hasty generalizations common in NLI errors

For our base model, we selected Mistral-7B due to its strong performance-to-parameter ratio, particularly for reasoning tasks, and the availability of API versions (like `open-mistral-7b` and `open-mistral-nemo`) that facilitated our multi-stage data generation process.

### Overview of Architecture

Our approach has two main components:

1. **Multi-Stage Data Augmentation Pipeline:**
   - **Initial Thought Generation:** Using `open-mistral-7b` to generate Chain-of-Thought reasoning for all premise-hypothesis pairs, capturing the base model's natural reasoning approach.
   - **Reflection and Correction:** For examples where the initial CoT led to incorrect predictions, employing a stronger model (`open-mistral-nemo`) to analyze the errors and generate improved reasoning paths, which we call our **Reflection-CoT mechanism**.
   - **Final Dataset:** Combining initially correct examples with reflected examples to create a comprehensive, balanced training dataset.

2. **Parameter-Efficient Fine-Tuning:**
   - Applying QLoRA (Quantized Low-Rank Adaptation) to fine-tune Mistral-7B efficiently, allowing training on consumer-grade hardware.
   - Optimizing sequence length and other hyperparameters based on empirical analysis of CoT examples.

### Contributions

Our primary contributions are:

1. The **Reflection-CoT** mechanism: A novel data augmentation approach that addresses label disagreement by providing models with both the correct label and flawed reasoning, enabling targeted correction of logical errors.

2. An empirical analysis of **CoT brevity and effectiveness**: Demonstrating that concise reasoning (150-350 tokens) correlates with higher accuracy for NLI tasks, informing both our prompt engineering and training optimizations.

3. A comprehensive case study of **QLoRA fine-tuning for reasoning tasks**: Providing practical insights into hyperparameter selection and training stability for parameter-efficient fine-tuning of reasoning-focused language models.

## Related Work

*[This section will review relevant literature on NLI techniques, Chain-of-Thought reasoning, synthetic data generation, LLM self-correction mechanisms, and parameter-efficient fine-tuning methods. It will position our work within this context and highlight the novelty of our approach.]*

## Methodology

In this section, we detail our technical approach to generating NLI data with Chain-of-Thought reasoning and fine-tuning the Mistral-7B model.

### Synthetic Data Augmentation

The core of our NLI fine-tuning dataset consists of synthetically generated CoT examples with binary labels. Our approach builds on the insight that while large language models can generate plausible reasoning chains, they may not always align with the labels in annotated datasets, particularly in cases of subjective or ambiguous examples.

#### Initial Thought Generation

For the first stage of our pipeline, we used `open-mistral-7b` to generate CoT reasoning and predicted labels for all premise-hypothesis pairs in our dataset. The model was prompted to:

1. Analyze the premise and hypothesis carefully
2. Generate step-by-step reasoning (encouraged to be brief, typically 3 steps)
3. Provide a final binary label (0 for no-entailment, 1 for entailment)
4. Format the output as JSON: `{"thought_process": "...", "predicted_label": 0|1}`

This process yielded an initial set of CoT examples, approximately 65-70% of which matched the dataset's gold standard labels. We chose Mistral-7B for this stage based on empirical observations that its concise reasoning patterns often aligned better with the dataset's binary labels compared to larger models that sometimes "overthought" examples, particularly for subjective cases.

The choice to capture the base model's "natural" reasoning before any correction was deliberate—it provided a baseline understanding of the model's approach and would potentially minimize friction between the base model and the fine-tuned LoRA adapter by preserving reasoning patterns the model was already comfortable with.

#### Model Architecture

We selected `mistralai/Mistral-7B-v0.3` as our base model for fine-tuning. This choice was driven by:

1. **Performance-to-Parameter Ratio:** Mistral-7B demonstrates strong reasoning capabilities despite its relatively modest size in the current LLM landscape.

2. **API Ecosystem:** The availability of API-accessible versions (like `open-mistral-7b` and `open-mistral-nemo`) facilitated our multi-stage data generation process.

3. **Speed and Efficiency:** The model offers favorable inference speed and resource requirements compared to larger alternatives, making it accessible for deployment on consumer hardware.

4. **Reasoning Capabilities:** Preliminary experiments showed Mistral-7B performed well on NLI tasks even before fine-tuning, providing a strong foundation.

#### Training Approach

To enable fine-tuning Mistral-7B on accessible hardware, we employed Parameter-Efficient Fine-Tuning (PEFT) with QLoRA (Quantized Low-Rank Adaptation). This approach has several components:

1. **Quantization:** The base model weights were loaded in 4-bit precision using NF4 ("NormalFloat 4") to drastically reduce memory requirements.

2. **LoRA Adapters:** We added small, trainable adapter matrices to attention mechanism layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`), keeping the base model frozen.

3. **Hyperparameter Optimization:**
   - **Effective Batch Size:** Varied from 16 to 64 through gradient accumulation
   - **Learning Rate:** Typically between 2e-4 and 5e-5 with a cosine scheduler
   - **Fixed Warmup Steps:** Used 50-150 steps rather than ratio-based warmup for better stability
   - **Sequence Length:** Optimized to 512 tokens based on analysis of typical CoT lengths

4. **Training Environment:** Dockerized setup to ensure reproducibility across different hardware environments, with configuration managed through Python files.

5. **Optimizations:** Several technical optimizations were applied:
   - Gradient checkpointing to reduce memory usage
   - BF16 precision using autocast wrappers to handle dtype mismatches
   - 8-bit Adam optimizer for memory efficiency

## Experiment I - Synthetic Data Augmentation

This section details our iterative journey to develop an effective data augmentation strategy for NLI Chain-of-Thought fine-tuning.

### Initial Experiments: Learning From Failure

Our first approach to fine-tuning was naive in several ways. We used a pre-quantized GPTQ version of Mistral-7B with poorly configured hyperparameters and an overly simplistic dataset. This initial dataset suffered from multiple limitations:

1. **Limited Coverage:** We only used examples where `open-mistral-7b` predictions already matched the dataset labels (approximately 65-70% of examples), discarding challenging cases.

2. **Data Reduction:** After further filtering to balance label distribution, we trained on only about 10,000 examples.

3. **Confirmation Bias:** By only including examples the model already handled correctly, we reinforced existing patterns without addressing blind spots.

The result was predictable: the model overfit extremely quickly, often within a single epoch, and failed to generalize to more challenging examples. This experience highlighted the critical importance of data quality and diversity in CoT fine-tuning. We needed a more sophisticated data generation approach that could handle the full spectrum of examples, including those where the model initially disagreed with dataset labels.

### Less-Is-All-You-Need: Improved Thought Generation

After our initial failure, we conducted a thorough analysis of the generated thought chains to understand patterns in high-performing versus low-performing examples. This analysis revealed a striking pattern: the length of the reasoning chain strongly correlated with accuracy.

Specifically, we found:
- Thought chains between 150-350 tokens showed the highest accuracy
- Examples with 350-500 tokens still performed well
- Performance degraded linearly as token count increased beyond 750-1000 tokens
- Very short examples (0-100 tokens) also performed poorly

This finding suggested an important insight: concise, focused reasoning was more likely to arrive at the correct label according to the dataset. Manual inspection confirmed this pattern, revealing that shorter chains were typically "concise, simplistic, logical," while longer ones often exhibited "overthinking" and struggled with examples that had subjective elements.

Based on this analysis, we redesigned our prompts to explicitly encourage brevity and structure (typically a 3-step reasoning process). This simple change had a remarkable impact, improving the initial generation accuracy from around 65% to 70-75% without changing the underlying model. This insight would later inform our choice of `max_seq_length=512` for fine-tuning, creating a "double win" of better accuracy and improved efficiency.

### LLM-As-A-Judge: Iterative Self-Critique & Improvement

With improved thought generation in place, we still needed to address the approximately 30% of examples where our generated thoughts disagreed with dataset labels. Rather than simply discarding these examples, we explored an automated approach to improve them.

We developed an LLM-as-a-judge system (implemented in `score_thoughts.py`) that would:
1. Score the quality of a generated thought process based on criteria such as coherence, correctness, and alignment with the label
2. For examples scoring below a threshold, attempt to improve the reasoning through iterative refinement
3. Recursively apply this process, potentially escalating to stronger models when necessary

This approach revealed a significant challenge: in our dataset, there was a stark precision-recall imbalance (approximately 90% precision but only 50% recall). This suggested either model bias, labeler bias, or both—many "no-entailment" examples in the dataset appeared to be entailed according to both the model's reasoning and our manual verification.

While promising in concept, this self-improvement approach proved prohibitively expensive (costing >£100 for a subset of examples) and often failed to converge on subjective examples. Despite numerous iterations and model upgrades, some premise-hypothesis pairs simply wouldn't yield a satisfactory agreement with the dataset label.

### Learn-From-Your-Mistakes: Self-Reflection & Correction

After consulting with university advisors, we confirmed that despite apparent subjectivity in some cases, the dataset should be treated as the "gold standard" for this task. This guidance led to our final and most successful data augmentation approach: the Reflection-CoT mechanism.

Rather than trying to coax models into naturally arriving at potentially counterintuitive labels or discarding challenging examples, we embraced a more direct approach:

1. For examples where the initial prediction was incorrect, we provided a stronger model (`open-mistral-nemo`, a 12B parameter model) with:
   - The original premise and hypothesis
   - The true label from the dataset 
   - The original, flawed thought process

2. We explicitly prompted the model to:
   - Analyze why the initial reasoning led to an incorrect conclusion
   - Identify the specific logical flaws or oversights
   - Generate a corrected reasoning path that logically leads to the true label

This approach was superior to simply asking a model to generate reasoning for a given label from scratch because:
- It specifically addressed the flaws in the original reasoning rather than generating any arbitrary path to the label
- It preserved the style and approach of the original CoT while fixing only what was necessary
- It created more natural, grounded corrections than might emerge from forcing a model to justify a label it disagreed with
- It avoided potential hallucination that might occur when a model was asked to support a conclusion that seemed counterintuitive to it

Manual inspection of a random sample of these reflections confirmed they effectively identified and corrected flaws in the original reasoning while maintaining consistency in style and approach.

### Dataset Composition

Our final fine-tuning dataset combined:
1. Examples where the initial `open-mistral-7b` generation produced the correct label and reasoning
2. Examples where the initial generation was incorrect but were corrected through our Reflection-CoT process

This approach effectively doubled our usable training data compared to our initial naive approach, improved class balance, and ensured coverage of the full spectrum of examples, including challenging edge cases.

**[TODO: Add specific dataset statistics here - total examples, percentage of reflected examples, class distribution]**

## Experiment II - Fine-Tuning with QLoRA

After developing our data augmentation pipeline, we turned to the technical challenges of fine-tuning Mistral-7B on this data efficiently.

### Dockerized Training Environment

To ensure consistency and reproducibility across different hardware environments, we developed a Dockerized training environment. This approach was crucial given the complex dependencies between specific versions of CUDA, PyTorch, and various libraries like PEFT, TRL, and bitsandbytes.

**[TODO: Add details about the Docker setup, base image, key dependencies]**

### Mistral-7B: 4-bit Quantized Model

Full fine-tuning of a 7B parameter model requires substantial GPU memory and computational resources. To make this process more accessible, we employed 4-bit quantization via the bitsandbytes library:

1. **NF4 Data Type:** The "NormalFloat 4" format provides a good balance between compression and representational capacity.
2. **Double Quantization:** For further memory reduction, we quantized both the weights and the quantization constants themselves.
3. **BFloat16 Computation:** While weights were stored in 4-bit precision, actual forward/backward computations used BFloat16 for stability.

This quantization strategy reduced the memory footprint by approximately 75% compared to FP16 training, making it possible to fine-tune on consumer GPUs with 24GB of VRAM.

### Parameter-Efficient Fine-Tuning with QLoRA

Building on the quantized model, we implemented Low-Rank Adaptation (LoRA) to further increase parameter efficiency:

1. **Target Modules:** We applied LoRA to the query, key, value, and output projection matrices in the attention mechanism (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
2. **Rank and Alpha:** We experimented with different rank values (typically `r=16` or `r=32`) and scaling factors (`lora_alpha=32` or `lora_alpha=64`), finding that higher ranks were beneficial for capturing the complexity introduced by our reflection data.
3. **Dropout:** A small dropout value (`lora_dropout=0.05`) was applied for regularization.

This QLoRA approach allowed us to fine-tune the model while updating only 0.1-0.2% of the parameters, dramatically reducing computational requirements while maintaining performance.

### Ablation Studies & Hyper-Parameter Tuning

We conducted several ablation studies to identify optimal training parameters:

1. **Batch Size and Learning Rate:**
   - **Ablation 0 (Small Batch):** Effective batch size 16, `lr=2e-4`, 2 epochs
   - **Ablation 1 (Medium Batch):** Effective batch size 32, `lr=2e-4`, 2 epochs, with gradient checkpointing
   - **Ablation 2 (Large Capacity/Batch):** Effective batch size 64, `lr=5e-5`, 5 epochs, with gradient clipping at 1.0

2. **Sequence Length:** Based on our token analysis, we set `max_seq_length=512`, which safely accommodated all generated CoT examples (maximum observed length <400 tokens) while significantly reducing computation time and memory usage.

3. **Warmup Strategy:** We found that fixed `warmup_steps` (ranging from 50-150) provided better stability than ratio-based warmup, particularly when resuming training or adjusting training duration.

**[TODO: Add table with detailed hyperparameter settings for each ablation]**

## Results

**[TODO: This section will present quantitative and qualitative results from our experiments, including:]**
- **Baseline Performance:** How Mistral-7B performed on NLI before fine-tuning
- **Fine-Tuned Performance:** Accuracy, precision, recall, and F1 scores across different models
- **Benchmarks:** Comparison to other approaches or models if available
- **Thought Quality:** Assessment of the quality and interpretability of generated reasoning

## Discussion

**[TODO: This section will interpret our results and discuss broader implications, including:]**

### Model Bias

**[TODO: Discuss potential biases in the model's reasoning patterns and predictions]**

### Labeller Bias

**[TODO: Discuss the observed precision-recall imbalance and its implications for dataset quality and subjective judgment in NLI]**

### Trade-Offs and Assumptions

**[TODO: Analyze the trade-offs between efficiency, accuracy, and interpretability in our approach]**

## Limitations and Future Work

**[TODO: This section will acknowledge limitations and suggest directions for future research:]**

### Self-Consistency

**[TODO: Discuss the potential for generating multiple reasoning paths and selecting the most common answer]**

### Tree-of-Thought

**[TODO: Explore the possibility of considering branching reasoning paths rather than linear chains]**

### RLHF Optimization

**[TODO: Consider how reinforcement learning from human feedback could further improve reasoning quality]**

### Refining Scoring

**[TODO: Discuss improvements to the LLM-as-a-judge system for more reliable assessment of reasoning quality]**

### Full Fine-Tuning

**[TODO: Analyze the potential benefits of full fine-tuning versus our QLoRA approach]**

## Conclusion

**[TODO: Summarize key findings and contributions]**

## References

**[TODO: Add references to relevant papers and resources]**

(Remaining sections to be completed in future updates. For practical implementation details, please refer to the corresponding documentation files: [DATA.md](DATA.md), [TRAINING.md](TRAINING.md), and [EVALUATION.md](EVALUATION.md))