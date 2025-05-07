# Project Blog & Experimental Journey

## Introduction

This document captures the chronological development, key decisions, and rationale behind the Mistral-7B NLI fine-tuning project. It provides a narrative of our experimental journey, documenting both successes and failures to provide context for the final approach.

## Project Objective

**Goal:** Fine-tune Mistral-7B for Natural Language Inference (NLI) with interpretable Chain-of-Thought (CoT) reasoning.

The core task was binary NLI classification, traditionally approached with encoder-based models like BERT. We chose to use a fine-tuned autoregressive LLM (Mistral-7B) combined with CoT for several reasons:

*   **Interpretability:** CoT provides a window into *how* the model reaches its classification, addressing potential label subjectivity and ambiguity.
*   **State-of-the-Art Approach:** Using CoT generation with a large autoregressive model represents a more advanced technique compared to standard classification.
*   **Technical Challenge:** This approach involves generation, structured output parsing, and careful prompt engineering, demonstrating deeper engagement with LLM capabilities.
*   **Synergy with QLoRA Fine-Tuning:** QLoRA doesn't alter the base weights, so fine-tuning data should complement, not contradict, the base model's reasoning patterns.

## Model Selection

We selected `mistralai/Mistral-7B-v0.3` based on several factors:

*   **Performance/Cost Balance:** 7B parameters offered significant capability without excessive training/inference costs or hardware demands.
*   **Inference Speed:** Relatively fast for a 7B model, important for processing thousands of examples.
*   **Reasoning Capability:** Strong reasoning abilities compared to other open-source models in the same weight class.
*   **Empirical Performance:** Early tests showed `open-mistral-7b` achieving the best accuracy and precision on NLI samples compared to other models in the same size category.
*   **API Ecosystem:** The availability of both `open-mistral-7b` and the stronger `open-mistral-nemo` (12B) through their API enabled our two-stage data generation process.

## Chronological Experimental Journey

### 1. Initial Approach: Learning From Failure

Our first approach was naive in several ways:

1. We used a pre-quantized GPTQ version of Mistral-7B with poorly configured hyperparameters.
2. We created an overly simplistic dataset that only included examples where `open-mistral-7b` predictions already matched the dataset labels (approximately 65-70% of examples).
3. After filtering to balance label distribution, we trained on only about 10,000 examples.

**Result:** The model overfit extremely quickly, often within a single epoch, and failed to generalize to more challenging examples. This highlighted the critical importance of data quality and diversity in CoT fine-tuning.

### 2. Data Analysis: Less-Is-All-You-Need

After our initial failure, we conducted a thorough analysis of the generated thought chains. This revealed a striking pattern: the length of the reasoning chain strongly correlated with accuracy.

Specifically, we found:
- Thought chains between 150-350 tokens showed the highest accuracy
- Examples with 350-500 tokens still performed well
- Performance degraded linearly as token count increased beyond 750-1000 tokens
- Very short examples (0-100 tokens) also performed poorly

Manual inspection confirmed that shorter chains were typically "concise, simplistic, logical," while longer ones often exhibited "overthinking" and struggled with subjective examples.

**Action:** We redesigned our prompts to explicitly encourage brevity and structure (typically a 3-step reasoning process). This simple change improved the initial generation accuracy from around 65% to 70-75% without changing the model.

### 3. Attempted Solution: LLM-As-A-Judge

With improved thought generation, we still needed to address the approximately 30% of examples where our generated thoughts disagreed with dataset labels. Rather than discarding these examples, we explored an automated approach to improve them.

We developed an LLM-as-a-judge system that would:
1. Score the quality of generated thought processes based on coherence, correctness, and alignment with the label
2. For examples scoring below a threshold, attempt to improve the reasoning through iterative refinement
3. Recursively apply this process, potentially escalating to stronger models when necessary

**Challenge:** We discovered a stark precision-recall imbalance (approximately 90% precision but only 50% recall), suggesting model bias, labeler bias, or both. Many "no-entailment" examples appeared to be entailed according to both the model's reasoning and our manual verification.

**Result:** This approach proved prohibitively expensive (costing >£100 for a subset of examples) and often failed to converge on subjective examples. Despite numerous iterations and model upgrades, some premise-hypothesis pairs simply wouldn't yield satisfactory agreement with the dataset label.

### 4. Final Solution: Learn-From-Your-Mistakes (Reflection-CoT)

After consulting with university advisors, we confirmed that despite apparent subjectivity, the dataset should be treated as the "gold standard" for this task. This guided us to our final data augmentation approach: the Reflection-CoT mechanism.

Rather than trying to coax models into naturally arriving at potentially counterintuitive labels or discarding challenging examples, we embraced a more direct approach:

1. For examples where the initial prediction was incorrect, we provided a stronger model (`open-mistral-nemo`, 12B) with:
   - The original premise and hypothesis
   - The true label from the dataset 
   - The original, flawed thought process

2. We explicitly prompted `open-mistral-nemo` to:
   - Analyze why the initial reasoning led to an incorrect conclusion
   - Identify the specific logical flaws or oversights
   - Generate a corrected reasoning path that logically leads to the true label

**Advantage:** This approach was superior to simply asking a model to generate reasoning for a given label from scratch because:
- It specifically addressed the flaws in the original reasoning rather than generating any arbitrary path
- It preserved the style and approach of the original CoT while fixing only what was necessary
- It created more natural corrections than forcing a model to justify a label it disagreed with
- It avoided potential hallucination that might occur when supporting a counterintuitive conclusion

Our final fine-tuning dataset combined:
1. Examples where the initial `open-mistral-7b` generation produced the correct label and reasoning
2. Examples where the initial generation was incorrect but were corrected through our Reflection-CoT process

This effectively doubled our usable training data compared to our initial approach, improved class balance, and ensured coverage of the full spectrum of examples.

## Fine-Tuning Evolution & Optimization

### Training Evolution with QLoRA

1. **Initial Attempt (Correct Predictions Only):**
   - The first fine-tuning runs used only the subset of data where the initial `open-mistral-7b` CoT generation produced the correct label.
   - **QLoRA Settings:** Started with common recommendations: `r=16`, `lora_alpha=32`, `dropout=0.05`.
   - **Outcome:** The model overfit very quickly (often before completing a single epoch) and converged to suboptimal performance.

2. **Second Attempt (Combined Correct + Reflected Data):**
   - The fine-tuning dataset combined initially correct examples with newly generated reflected examples.
   - **Impact:** This effectively doubled the usable training data, created a more balanced class distribution, and introduced more diverse reasoning paths.
   - **Outcome:** Using the same QLoRA settings, the model showed much better training dynamics, improving well past the first epoch.

3. **Higher Rank Experiments:**
   - We hypothesized that the added complexity from reflected data might benefit from increased capacity in LoRA adapters.
   - We tested doubled rank and alpha (`r=32`, `lora_alpha=64`), with increased batch size and gradient clipping.
   - **Initial Result:** This configuration trained successfully but didn't significantly outperform the simpler setup.

4. **Revisiting Larger Scale with Stability:**
   - The hypothesis about needing more capacity was revisited with crucial stability measures:
     - **Increased Effective Batch Size:** To average out noise from diverse examples
     - **Gradient Clipping:** To prevent gradient explosions
     - **Lower Learning Rate:** Tuned for the larger batch size and potential noise
   - **Outcome:** This stabilized approach with higher rank (`r=32/a=64`) and larger batch size yielded the best results, confirming that the complex reflected dataset benefits from more capacity when properly stabilized.

### Sequence Length Optimization: A Double Win

Reducing the maximum sequence length was a key optimization driven by both performance observations and efficiency needs:

1. **Accuracy Correlation:** Analysis revealed that thought chains resulting in a total length of 250-400 tokens showed the highest accuracy (85-95%), while longer chains had slightly lower accuracy (70-80%).

2. **Prompt Refinement for Brevity:** We tuned prompts to encourage concise reasoning by:
   - Adding explicit brevity instructions
   - Requesting exactly 3 reasoning steps, based on analysis of successful examples
   - Making brevity a criterion in scoring and self-improvement prompts
   
   This refinement itself improved initial thought generation accuracy from ~65% to 70-75%.

3. **Setting `max_seq_length=512`:** After generating the entire dataset with brevity-focused prompts, we confirmed no example exceeded 400 tokens. This allowed a safe reduction from the default 2048 to 512 tokens.

4. **The "Double Win":**
   - **Improved Accuracy:** Concise reasoning correlated with higher accuracy
   - **Massive Efficiency Gains:** Shorter sequences dramatically reduced computation (transformer attention scales quadratically with sequence length) and memory requirements

This optimization made training faster, freed GPU memory for other hyperparameter exploration, reduced inference time significantly, and increased overall project feasibility.

### Learning Rate, Batch Size, and Warmup Stability

We encountered significant instability when adjusting parameters like batch size or training duration:

- **Warmup Ratio Sensitivity:** Initial runs used `warmup_ratio = 0.03`, which worked for specific training lengths but became problematic when total steps changed.
- **Batch Size Interaction:** Learning rates suitable for smaller batches became unstable with larger batches without proper adjustment.
- **Resumption Challenges:** Resuming training often failed to correctly restore the scheduler state, causing learning rate spikes and loss increases.

**Solution:** We shifted from using `warmup_ratio` to specifying fixed `warmup_steps` (around 50-75 for smaller runs, 150 for larger runs). This provided more direct control and made tuning more predictable, especially when resuming training or adjusting duration.

### Epoch Count Impact on Training Dynamics

A crucial lesson was that changing epoch count transforms the entire learning process:

- **Warmup Interaction:** With `warmup_ratio`, doubling epochs halves the effective warmup period relative to total training time.
- **Learning Rate Decay:** Changing epoch count stretches or compresses the decay curve with schedulers like cosine decay.
- **Practical Impact:** Attempts to extend promising but prematurely ended ablations by adding epochs often failed because this altered the warmup/decay balance.

**Best Practice:** Start with conservative epoch estimates and incrementally add epochs in new training runs rather than dramatically increasing them.

## Technical Deep Dives

### Prompt Engineering for CoT Generation

Developing effective prompts involved significant experimentation:

- **Cross-Model Testing:** Initial tests across various models from OpenAI, Mistral AI, Meta, and Anthropic.
- **Structuring:** Adoption of Markdown and XML tags to clearly delineate prompt parts.
- **Output Format:** Enforcing JSON output with few-shot examples in the prompt.
- **CoT Guidance:** Step-by-step instructions informed by NLI problem-solving strategies.
- **Experimentation Cost:** Approximately £5 for testing on 1,000 example subset.

### Reflection Prompting: Guiding the Stronger Model

The choice of *how* to prompt the reflection model was crucial:

1. **Simple Correction Option:** Provide the premise, hypothesis, and true label, asking for a new thought process.
2. **Guided Reflection Option:** Provide premise, hypothesis, true label, AND the original incorrect thought, explicitly asking for analysis and correction.

We chose Option 2 because:
- It addressed the subjectivity issue by showing where reasoning failed rather than forcing any path to the correct label
- It focused on correcting logic rather than just changing the conclusion
- It leveraged `open-mistral-nemo`'s strength for this meta-reasoning task, which requires understanding diverse viewpoints
- It encouraged mimicking the desired concise style while correcting flaws

### Navigating Label Disagreement

The ~30% of examples where model predictions disagreed with dataset labels presented options:

1. **Keep Original:** Trust the model's reasoning despite disagreeing with the label.
2. **Omit:** Discard these disagreeing examples entirely.
3. **Correct Naturally:** Guide the model towards generating reasoning that naturally leads to the correct label.
4. **Correct Forcefully:** Provide the correct label and instruct the model to generate supporting reasoning.

Initially, Options 1 and 2 were appealing since we often agreed with the model over the dataset. However, the coursework required treating the dataset as the "gold standard," shifting our objective from finding absolute NLI truth to learning the dataset's specific mapping.

After the expensive failure of Option 3 (iterative self-improvement), our **Reflection-CoT** represents a hybrid of Options 3 and 4 - providing the target label while also providing the original reasoning as context for a more natural correction.

## Implementation Details

### Docker for Reproducibility

Docker was adopted to ensure consistent environment across different execution platforms:

- **Essential for GPU Tasks:** Strictly necessary for training and inference scripts due to complex GPU dependencies.
- **Convenience for Other Tasks:** Data generation, download scripts, etc. also provided with Docker wrappers for complete environment consistency.

### Configuration System for Training

We implemented a Python-based configuration system inspired by NanoGPT:

- **Pain Point Addressed:** Moving away from long, complex bash commands for different training configurations.
- **Self-Documenting Experiments:** Separate configuration files provide clear, version-controllable records of each experiment.
- **Flexibility:** Easy modification of parameters while allowing command line overrides for specific tweaks.
- **Simplified Entry Point:** Cleaner main training script focusing on launching the container and passing configuration.

### Evaluation Strategy

- **Robust Prediction Extraction:** Corrected parsing logic to reliably extract JSON outputs, even with generation artifacts.
- **Metrics Focus:** Primary focus on accuracy, precision, recall, and F1 for quantitative evaluation.
- **Training vs. Final Evaluation:** SFTTrainer monitors loss and token accuracy during training, while dedicated inference script provides definitive classification metrics.

## Future Directions

### Potential Improvements

- **Agentic Data Synthesis:** Transform the sequential data generation into an integrated, self-improving loop.
- **Deeper Qualitative Analysis:** Structured analysis of reasoning patterns and reflection impact.

### Alternative Reasoning Paradigms

- **Multi-Path Reasoning:** Generate diverse reasoning chains for subjective examples.
- **Advanced Frameworks:** Explore Tree-of-Thoughts or reinforcement learning approaches for reasoning optimization.

### Research Directions

- Further investigate the effectiveness of reflection mechanisms for improving reasoning.
- Study subjective judgment in NLI datasets and its impact on model training.
- Explore the relationship between reasoning brevity and accuracy in different contexts.

---