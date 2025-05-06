# NLIstral-7B-QLoRA: Interpretable NLI with Augmented Chain-of-Thought Fine-Tuning

## Abstract

This paper details the fine-tuning of the Mistral-7B language model for Natural Language Inference (NLI) tasks, employing Chain-of-Thought (CoT) reasoning. Our approach focuses on generating interpretable reasoning processes to accompany NLI classifications, thereby enhancing both model accuracy and transparency. We introduce a multi-stage data augmentation pipeline, including an innovative reflection mechanism to refine CoT examples, and leverage QLoRA for parameter-efficient fine-tuning. The resulting models demonstrate strong NLI performance while providing insights into their decision-making.

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
      - 3.1.1. [Initial Thought Generation](#initial-thought-generation)
      - 3.1.2. [Reflection on Errors: The Reflection-CoT Mechanism](#reflection-on-errors-the-reflection-cot-mechanism)
      - 3.1.3. [Prompt Engineering](#prompt-engineering)
      - 3.1.4. [Final Dataset Composition and Characteristics](#final-dataset-composition-and-characteristics)
   - 3.2. [Model Architecture & Fine-Tuning](#model-architecture--fine-tuning)
      - 3.2.1. [Base Model: Mistral-7B](#base-model-mistral-7b)
      - 3.2.2. [Parameter-Efficient Fine-Tuning (PEFT) with QLoRA](#parameter-efficient-fine-tuning-peft-with-qlora)
   - 3.3. [Training Approach](#training-approach)
      - 3.3.1. [Environment and Configuration](#environment-and-configuration)
      - 3.3.2. [Key Hyperparameters and Optimizations](#key-hyperparameters-and-optimizations)
4. [Experimental Setup](#experimental-setup)
   - 4.1. [Datasets](#datasets)
   - 4.2. [Baselines](#baselines)
   - 4.3. [Evaluation Metrics](#evaluation-metrics)
   - 4.4. [Ablation Studies Design](#ablation-studies-design)
5. [Results and Analysis](#results-and-analysis)
   - 5.1. [Main NLI Performance](#main-nli-performance)
   - 5.2. [Ablation Study Results](#ablation-study-results)
      - 5.2.1. [Impact of the Reflection-CoT Mechanism](#impact-of-the-reflection-cot-mechanism)
      - 5.2.2. [Impact of QLoRA Configuration Choices](#impact-of-qlora-configuration-choices)
   - 5.3. [Reasoning Quality Assessment](#reasoning-quality-assessment)
      - 5.3.1. [Quantitative Metrics (LLM-as-a-Judge)](#quantitative-metrics-llm-as-a-judge)
      - 5.3.2. [Qualitative Analysis and Error Analysis](#qualitative-analysis-and-error-analysis)
   - 5.4. [Efficiency Metrics](#efficiency-metrics)
6. [Discussion](#discussion)
   - 6.1. [Interpretation of Key Findings](#interpretation-of-key-findings)
   - 6.2. [Addressing NLI Subjectivity and Label Quality](#addressing-nli-subjectivity-and-label-quality)
   - 6.3. [Model Biases and Limitations](#model-biases-and-limitations)
   - 6.4. [Trade-Offs in Efficiency and Performance](#trade-offs-in-efficiency-and-performance)
   - 6.5. [Comparison with Related Work](#comparison-with-related-work)
7. [Limitations and Future Work](#limitations-and-future-work)
8. [Conclusion](#conclusion)
9. [References](#references)

## 1. Introduction and Motivation

*This section will outline the problem of NLI, the advantages of using Chain-of-Thought reasoning, and our motivations for pursuing this approach.*
*TODO: Refine to ensure a formal, academic tone. Clearly delineate the gap in existing research that this work addresses.*

### 1.1. Problem Statement

Natural Language Inference (NLI) is the task of determining whether a "hypothesis" sentence can be logically inferred from a "premise" sentence. Common classifications include entailment, contradiction, or neutrality. This project focuses on binary classification: entailment (1) or no-entailment (0). NLI is challenging due to linguistic ambiguity, the need for world knowledge, and complex reasoning.

### 1.2. Motivation & Background

The motivation for this work stems from the desire to enhance both the accuracy and interpretability of NLI models. Traditional NLI models often act as black boxes. Chain-of-Thought (CoT) prompting, which encourages models to generate step-by-step reasoning, offers a path to transparency.
*TODO: Expand on *why* CoT is particularly suited for NLI beyond just "transparency." Does it help with specific types of NLI challenges? Cite seminal CoT papers.*
Mistral-7B was chosen for its strong performance in the 7B parameter class, favorable cost/inference metrics, and API accessibility crucial for our multi-stage data generation.
*TODO: Briefly justify model choice in context of other available models at the time, if possible and relevant.*
The project aligns with coursework objectives requiring creativity, technical depth, and high performance on a hidden NLI test set. A key aspect was to develop a model that not only classifies correctly but also explains *why*.
*TODO: Rephrase coursework motivation into a more general research motivation for broader appeal.*

### 1.3. Overview of Architecture

Our architecture involves two main components:
1.  **Multi-Stage Data Augmentation Pipeline:**
    *   **Initial Thought Generation:** Generating initial CoT reasoning for premise-hypothesis pairs using an LLM (`open-mistral-7b`).
    *   **Reflection and Correction:** Identifying incorrect initial generations and using a stronger LLM (`open-mistral-nemo`) to reflect on the errors and generate corrected reasoning paths leading to the true label. This constitutes our proposed Reflection-CoT mechanism.
    *   The final fine-tuning dataset combines initially correct examples and corrected reflected examples.
2.  **Parameter-Efficient Fine-Tuning (PEFT):**
    *   Employing QLoRA to fine-tune `mistralai/Mistral-7B-v0.3` on the augmented CoT dataset. This involves 4-bit quantization of the base model and training low-rank adapters.
*TODO: Ensure this overview clearly signposts the novel components being introduced.*

### 1.4. Contributions
*TODO: Add this subsection. Clearly list 2-3 specific, novel contributions of this paper based on the critique (e.g., "The Reflection-CoT data augmentation strategy...", "A comprehensive empirical study of Reflection-CoT for NLI on Mistral-7B...", "The release of a new NLI-CoT dataset augmented via Reflection-CoT..."). This is a CRITICAL addition.*

## 2. Related Work
*TODO: CRITICAL SECTION - This entire section needs to be drafted from scratch. It should comprehensively review existing literature and clearly position this work within that context, highlighting differentiation.*

### 2.1. Natural Language Inference Techniques
*TODO: Review traditional ML, non-Transformer DL, and Transformer-based approaches to NLI. Mention key benchmarks (SNLI, MNLI, etc.).*

### 2.2. Chain-of-Thought Reasoning
*TODO: Discuss the origins of CoT (e.g., Wei et al., 2022), its applications, and different methods for generating or improving CoT (e.g., Self-Consistency, Tree of Thoughts if relevant as contrast).*

### 2.3. Synthetic Data Generation and Augmentation for NLP
*TODO: Review literature on using LLMs to generate synthetic data for training other models, especially for reasoning tasks or NLI.*

### 2.4. LLM Self-Correction and Reflection Mechanisms
*TODO: This is key for differentiating the proposed "Reflection-CoT". Review existing work on LLMs performing self-critique, iterative refinement, or learning from error signals. Clearly articulate how the proposed reflection mechanism is novel or distinct.*

### 2.5. Parameter-Efficient Fine-Tuning (PEFT)
*TODO: Briefly introduce PEFT methods, focusing on LoRA and QLoRA as context for the methods used in this paper. Cite relevant papers.*

## 3. Methodology

*This section will detail our technical approach to data augmentation and model training.*
*TODO: Consider renumbering this section and subsequent ones if a different overall paper structure is desired.*

### 2.1. Synthetic Data Augmentation

The core of our NLI fine-tuning data was synthetically generated CoT examples. This section details our proposed **Reflection-CoT** data augmentation pipeline.
*TODO: Rename section 3 to a more appropriate number (e.g., 3) once "Related Work" is section 2. Ensure consistent numbering throughout.*

#### 2.1.1. Initial Thought Generation
The first pass of CoT reasoning and predicted labels was generated using `open-mistral-7b`.
*   **Rationale:** This model was selected for initial CoT generation due to observations from preliminary analyses (as detailed in `BLOG.md`) suggesting its propensity for concise reasoning paths. These simpler paths often demonstrated a higher correlation with the ground truth labels of the NLI dataset, some ofwhich exhibited subjective characteristics. The objective was to capture the base model's "natural" or inherent reasoning patterns on the NLI task prior to any corrective fine-tuning, providing a baseline understanding of its approach and minimizing initial friction with the LoRA adapter's learning process.
*TODO: Detail the specific prompt structure or template used for this stage, if possible with an example. Mention any few-shot examples used in the prompt, and if available, quantify the impact of prompt refinement (e.g., from BLOG.md, initial CoT accuracy improved from ~65% to 70-75% after prompt tuning for brevity/structure). Mention that few-shot examples were crucial for consistent JSON output format adherence.*

#### 2.1.2. Reflection on Errors: The Reflection-CoT Mechanism
A core component of our methodology is the **Reflection-CoT mechanism**, designed to address instances where the initial CoT generation by `open-mistral-7b` yielded predictions inconsistent with the provided dataset labels (approximately 30% of cases, as noted in `BLOG.md`).
*   **Addressing Label Disagreement & Limitations of Automated Correction:** Initial explorations (detailed in `BLOG.md`) into resolving these disagreements by attempting to automatically guide the model towards the correct label without explicit instruction (i.e., "correct naturally" through iterative self-scoring with stronger models) proved both prohibitively expensive (estimated >£100 in API costs for a subset) and largely ineffective. Models often struggled to find coherent, natural reasoning paths to the target label, reinforcing concerns about dataset label subjectivity and the limitations of purely automated refinement in such scenarios.
*   **Mechanism Design:** Consequently, the Reflection-CoT mechanism employs a more direct approach. For premise-hypothesis pairs where `open-mistral-7b`'s prediction was incorrect, a stronger model, `open-mistral-nemo` (a 12B parameter model), was utilized. The `open-mistral-nemo` model was provided with:
    1.  The original premise and hypothesis.
    2.  The **true label** from the dataset (acknowledging it as the "gold standard" for the task).
    3.  The **original, flawed thought process** generated by `open-mistral-7b`.
    The model was then explicitly prompted to analyze the initial mistake and generate a corrected reasoning path that logically leads to the true label.
*   **Rationale for Stronger Model & Contextual Input:** `open-mistral-nemo` was chosen for its enhanced reasoning capabilities, deemed necessary for the complex meta-reasoning task of identifying logical flaws in another model's output and constructing a sound, corrective argument. Providing the original flawed reasoning as context was a deliberate design choice aimed at:
    *   Guiding `open-mistral-nemo` to correct the *specific logical failure* rather than merely generating any valid path to the true label.
    *   Addressing the potential for the reflection model to produce forced or unnatural reasoning if only given the target label, especially given the observed subjectivity of some dataset labels.
    *   Encouraging the reflection model to potentially emulate the concise style of the initial CoT, having seen an example (albeit flawed).
    This process was crucial for systematically handling challenging examples and creating a more robust training set by learning from initial model failures.
*TODO: FORMALIZE this further. Consider a small diagram or pseudo-algorithm illustrating the Reflection-CoT flow. Detail the specific prompt structure/template used for the reflection stage. Clearly articulate how this mechanism differs from or improves upon methods identified in Section 2.4 (LLM Self-Correction and Reflection Mechanisms). Ensure this links to the "Contributions" section.*

#### 2.1.3. Prompt Engineering
Significant effort was invested in prompt design for both generation and reflection:
*   **Structure:** Utilized Markdown and XML tags for clarity.
*   **Output Format:** Enforced JSON output (`{"thought_process": "...", "predicted_label": ...}`) using few-shot examples in the prompt.
*   **CoT Guidance:** Instructions for step-by-step reasoning, refined to encourage brevity and a 3-step structure, which correlated with higher accuracy and efficiency. (Per `BLOG.md`, this refinement itself improved initial CoT accuracy from ~65% to ~70-75% - *TODO: Confirm/update placeholder numbers if possible*).
*   **Reflection Prompt:** Specifically asked the model to analyze the prior (incorrect) thought process and generate a corrected one.
*TODO: Provide concrete examples of the initial and refined prompts if possible. Quantify the impact of prompt refinement if data is available (e.g., "Initial prompts yielded X% JSON format adherence, refined prompts yielded Y%"). Mention that few-shot examples for JSON output were critical here too (from `BLOG.md`).*

#### 2.1.4. Final Dataset Composition and Characteristics
The fine-tuning dataset was constructed by:
*   Including examples where the initial `open-mistral-7b` generation was correct.
*   Including examples where the initial generation was incorrect but were subsequently corrected through the reflection process with `open-mistral-nemo`.
This strategy aimed to leverage the strengths of both models and ensure the fine-tuning dataset covered the full range of original NLI examples with the best available reasoning for each.
*TODO: Provide statistics for the final fine-tuning dataset: total number of examples, number/percentage of reflected examples, distribution of labels. Refer to `HUB_DATASET.md` for placeholders like [TOTAL_EXAMPLES] but integrate them smoothly here. Mention the source of the original NLI data (name of dataset, citation if applicable, or describe if custom).*

### 2.2. Model Architecture & Fine-Tuning

#### 2.2.1. Base Model: Mistral-7B
The base model for fine-tuning was `mistralai/Mistral-7B-v0.3`.
*   **Rationale:** Chosen for its strong performance-to-cost ratio, good reasoning capabilities within its size class, fast inference speed, and the availability of an API ecosystem that facilitated the multi-stage data generation process.

#### 2.2.2. Parameter-Efficient Fine-Tuning (PEFT) with QLoRA
QLoRA was employed for fine-tuning to manage memory constraints.
*   **Quantization:** The base Mistral-7B model weights were loaded in 4-bit precision using NF4 ("NormalFloat 4") via `bitsandbytes`. Double quantization was used for further memory savings.
*   **LoRA (Low-Rank Adaptation):** Small, trainable adapter matrices were injected into the attention mechanism's linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) of the frozen, quantized base model. Only these adapter weights were updated during training.
*   **Key QLoRA Parameters:**
    *   `lora_r` (rank): Typically 16 or 32 in experiments.
    *   `lora_alpha` (scaling): Typically `2 * r`.
    *   `lora_dropout`: e.g., 0.05.
*TODO: Justify choices for QLoRA parameters if possible (e.g., "Target modules were chosen based on common practices for LoRA..."). If specific values were tuned, mention that here or in the experimental section. Refer to `BLOG.md` discussion on `r=16` vs `r=32` and link to experimental findings in Section 5.*

### 2.3. Training Approach

#### 2.3.1. Environment and Configuration
*   **Dockerized Environment:** Training was conducted within a Docker container to ensure reproducibility and manage complex GPU dependencies (CUDA, PyTorch, PEFT libraries). The `Dockerfile` specifies the environment.
*   **Python-based Configuration System:** Inspired by NanoGPT, a system using Python files (`train/configs/*.py`) and a loader (`train/config_loader.py`) was used to manage hyperparameters and experiment settings, offering readability and version control over bash scripts.

#### 2.3.2. Key Hyperparameters and Optimizations
*   **Epochs:** Typically 2-5, with early stopping based on validation loss.
*   **Batch Size:** Effective batch sizes varied (e.g., 16, 32, 64) using `per_device_train_batch_size` and `gradient_accumulation_steps`.
*   **Learning Rate:** e.g., `2e-4` or `5e-5`, with a cosine learning rate scheduler and warmup.
    *   **Warmup Strategy:** Fixed `warmup_steps` (e.g., 50-150, as noted in `BLOG.md`) were preferred over `warmup_ratio`. This decision was based on observed instabilities when `warmup_ratio` interacted with varying total training steps (due to changes in epochs or batch sizes) and challenges in correctly restoring learning rate scheduler states upon resumption of training, which could lead to inappropriate learning rate spikes (`BLOG.md`). Fixed steps provided more predictable and reproducible control over the initial learning phase.
*TODO: Be specific about how early stopping was implemented if used. For LR, provide justification if choices were based on preliminary experiments or standard practices.*
*   **Optimizer:** `paged_adamw_8bit` was used for memory efficiency.
*   **Sequence Length Optimization:** `max_seq_length` was set to 512 tokens. This decision was a crucial post-analysis optimization (`BLOG.md`). After generating the full CoT dataset using brevity-focused prompts, analysis confirmed that no generated example exceeded 400 tokens. Setting `max_seq_length=512` thus safely accommodated all generated CoTs while dramatically reducing computational load (due to quadratic scaling of attention) and GPU VRAM requirements. This optimization was instrumental in enabling the exploration of other hyperparameters (e.g., larger batch sizes, higher LoRA ranks) and achieved a "double win": improved potential accuracy through concise reasoning and significant efficiency gains.
*TODO: If the analysis notebook correlating length/accuracy is found or can be reproduced, cite it or summarize key findings more concretely here. Mention the ~3x inference speedup if attributable and quantifiable.*
*   **BF16 Handling:** An `autocast` wrapper around the model's forward pass was used to mitigate dtype mismatches when using `bfloat16` precision with gradient checkpointing.
*   **Gradient Checkpointing:** Enabled in most configurations to save memory by recomputing activations during the backward pass.
*TODO: Explain *why* these optimizations (BF16 handling, gradient checkpointing) were necessary or beneficial in the context of the hardware used or model size.*

## 3. Experiment I - Synthetic Data Augmentation Evolution
*TODO: Rename this section to something like "Development of the Reflection-CoT Data Augmentation Strategy" and integrate its content more formally into the Methodology (Section 3.1) or a dedicated part of the Experimental Setup (Section 4). The current narrative style is more suited for a blog. Focus on the *findings* from these evolutionary steps.*

### 3.1. Initial Attempts & Learning from Failures
The first fine-tuning attempts used only the subset of data where the initial `open-mistral-7b` CoT generation produced the *correct* label (~65-70% of the data).
*   **Outcome:** Suffered from data bias and overfitting, often before completing one epoch. The model struggled with harder examples or those where initial reasoning was flawed.
*TODO: If possible, quantify "overfitting" (e.g., divergence between training and validation loss).*

### 3.2. Optimizing Thought Generation: Brevity and Structure
Analysis revealed that concise reasoning (250-400 tokens, often 3 core steps) correlated with higher accuracy.
*   **Action:** Prompts were refined to explicitly request brief, 3-step reasoning.
*   **Impact:** Improved initial thought generation accuracy (from ~65% to 70-75%) and enabled the safe use of `max_seq_length=512`, yielding significant efficiency gains.

### 3.3. LLM-As-A-Judge: Iterative Self-Critique
An attempt was made to use LLMs to automatically score and refine thought processes (`scripts/score_thoughts.py`).
*   **Outcome:** While some potential was shown, this approach proved expensive and often struggled with the inherent subjectivity of NLI labels, sometimes failing to converge on improved reasoning. Its results were interpreted cautiously.
*TODO: Briefly describe the scoring criteria used for LLM-as-a-judge. Mention inter-LLM agreement or LLM vs. human agreement if any checks were done, even anecdotally.*

### 3.4. Learn-From-Your-Mistakes: The Reflection Mechanism
The core innovation in data strategy was the reflection process.
*   **Process:** When the initial `open-mistral-7b` prediction was incorrect, the stronger `open-mistral-nemo` model was prompted with the original premise, hypothesis, the *true* label, and the *original incorrect thought process*. It was asked to analyze the mistake and generate a corrected reasoning path.
*   **Rationale:** This directly addressed examples the initial model failed on, using a more capable model to provide high-quality corrective reasoning. It also aimed to make the correction more natural by providing context of the initial error.
*   **Impact:** Effectively doubled the usable training data, improved class balance, and introduced more diverse and complex reasoning paths, leading to better training dynamics and model performance.

## 4. Experimental Setup
*TODO: NEW SECTION - Create this section to detail the experimental design formally.*

### 4.1. Datasets
*TODO: Describe the NLI dataset used for fine-tuning and evaluation (source, splits, any preprocessing not covered in Methodology). Also specify the test set(s) used for reporting final results.*

### 4.2. Baselines
*TODO: CRITICAL - Define the baseline models for comparison. Examples:
    *   Mistral-7B fine-tuned on NLI *without* CoT.
    *   Mistral-7B fine-tuned *only* on initial CoT (before reflection).
    *   Performance of `open-mistral-7b` / `open-mistral-nemo` with zero-shot/few-shot CoT prompting on the NLI task.
    *   Any established public benchmark results for this specific NLI dataset, if applicable.*

### 4.3. Evaluation Metrics
*TODO: List all metrics used:
    *   **NLI Performance:** Accuracy, Precision, Recall, F1-score.
    *   **Reasoning Quality:** Describe the human evaluation protocol if conducted (criteria, number of annotators, agreement metric like Kappa). For LLM-as-a-judge, detail its setup and the metrics it outputs.
    *   **Efficiency Metrics:** Training time, GPU memory usage, inference latency/throughput (specify hardware used for these measurements).*

### 4.4. Ablation Studies Design
*TODO: Formally describe the design of each ablation study. What specific hypothesis was each ablation testing? (e.g., "To evaluate the impact of the Reflection-CoT mechanism, we compare Model A (trained with Reflection-CoT) against Model B (trained only on initial CoT data), keeping all other hyperparameters constant."). Refer to the key ablation families previously mentioned.*

## 5. Results and Analysis
*TODO: NEW SECTION - This section will present and analyze the results from the experiments defined above.*

### 5.1. Main NLI Performance
*TODO: Present the main results comparing your best model(s) (e.g., Ablation2_Best) against the defined baselines using the NLI performance metrics. Use a clear table. Discuss statistical significance if calculated.*
The table from the old section 5.2 ("Fine-Tuned Model Performance") should be presented here, but *with baseline results included for comparison.*
```
| Model                                  | Accuracy     | Precision    | Recall       | F1 Score     |
|----------------------------------------|--------------|--------------|--------------|--------------|
| Baseline 1 (e.g., M7B FT w/o CoT)      | [ACC]        | [PREC]       | [REC]        | [F1]         |
| Baseline 2 (e.g., M7B FT initial CoT)  | [ACC]        | [PREC]       | [REC]        | [F1]         |
| Ablation0_Best (Reflection-CoT)        | [ACCURACY_0] | [PRECISION_0]| [RECALL_0]   | [F1_0]       |
| Ablation1_Best (Reflection-CoT)        | [ACCURACY_1] | [PRECISION_1]| [RECALL_1]   | [F1_1]       |
| Ablation2_Best (Reflection-CoT)        | [ACCURACY_2] | [PRECISION_2]| [RECALL_2]   | [F1_2]       |
```
*TODO: Include training/validation loss curves here (`metrics/training_dynamics.png` placeholder).*
*TODO: Placeholder for `metrics/model_performance.png` (comparison figure).*

### 5.2. Ablation Study Results
*TODO: Present the results of each ablation study clearly, likely in tables, and discuss what they reveal.*

#### 5.2.1. Impact of the Reflection-CoT Mechanism
*TODO: Show direct comparison of models with and without the reflection mechanism. Quantify the improvement.*

#### 5.2.2. Impact of QLoRA Configuration Choices
*TODO: Discuss findings from varying LoRA rank, alpha, batch sizes, etc. Which configurations worked best and why (hypothesized)?*

### 5.3. Reasoning Quality Assessment
*TODO: This subsection replaces the old 5.4.*

#### 5.3.1. Quantitative Metrics (LLM-as-a-Judge)
*TODO: Report results from LLM-as-a-judge if used, acknowledging limitations. If human evaluation was done, present those quantitative scores here (e.g., average ratings for coherence, correctness).*

#### 5.3.2. Qualitative Analysis and Error Analysis
*TODO: Provide 2-3 carefully chosen qualitative examples of generated `thought_process` (good and bad examples). Perform an error analysis: What types of NLI problems or reasoning errors are still prevalent? Does the CoT help identify these?*
*TODO: Placeholder for `metrics/token_vs_accuracy.png` (from `HUB_MODEL.md`), if relevant to reasoning quality discussion.*

### 5.4. Efficiency Metrics
*TODO: Report training times, memory usage, inference speeds. Compare QLoRA efficiency to potential full fine-tuning if estimates are available. Discuss the impact of sequence length optimization here.*

## 6. Discussion
*This section interprets the results and discusses broader implications.*
*TODO: Renumber and ensure content flows from the new "Results and Analysis" section.*

### 6.1. Interpretation of Key Findings
*TODO: NEW SUBSECTION. Synthesize the main takeaways from the results. Why did the proposed methods perform as they did? What are the most significant findings?*

### 6.2. Addressing NLI Subjectivity and Label Quality
A key challenge in NLI is the potential subjectivity or ambiguity of dataset labels. Our initial `open-mistral-7b` generations sometimes disagreed with provided labels, with its reasoning appearing plausible. The reflection mechanism, by providing the "gold standard" true label and the flawed reasoning, forced the stronger model to find a path to the given answer. This pragmatic approach aimed to maximize performance on the dataset as defined, rather than adjudicate absolute logical truth.

### 6.3. Model Biases and Limitations
*   **Inherited Bias:** The fine-tuned models likely inherit biases present in the base Mistral-7B model and the NLI training data.
*   **Reasoning Patterns:** The CoT generation process may lead to specific, learned reasoning patterns that might not cover all valid logical approaches.
*   **Domain Specificity:** Performance may vary on NLI tasks from different domains than the training data.

### 6.4. Trade-Offs in Efficiency and Performance
*   **QLoRA:** Offers significant memory savings, enabling fine-tuning on accessible hardware, but may not achieve the absolute peak performance of full fine-tuning.
*   **CoT Generation:** The data augmentation process, especially API calls for generation and reflection, incurred computational costs and time.
*   **Sequence Length:** The strategic reduction to `max_seq_length=512` was a major efficiency gain, justified by prompt engineering that encouraged concise thoughts without truncating valid reasoning. This was critical for iterating on other hyperparameters.

### 6.5. Comparison with Related Work
*TODO: NEW SUBSECTION. After presenting your results, explicitly compare your findings/methodology to key papers discussed in the "Related Work" section. How does your work advance or differ from theirs based on your empirical results?*

## 7. Limitations and Future Work

*This section outlines the current limitations of the work and potential avenues for future research.*
*TODO: Consider renumbering.*

### 7.1. Current Limitations
*   **Limited Scope of Reasoning Evaluation:** Primarily relies on classification metrics as a proxy for reasoning quality. Deeper qualitative analysis is needed.
*   **Dependency on Base Model Quality:** The quality of generated CoT is dependent on the capabilities of the LLMs used for generation (`open-mistral-7b`, `open-mistral-nemo`).
*   **English-Only:** The current work is limited to English NLI.
*   **Fixed Reasoning Format:** The model is trained to produce a specific JSON output and CoT style.

### 7.2. Future Research Directions
*   **Advanced Reasoning Frameworks:**
    *   **Self-Consistency:** Generating multiple reasoning paths and selecting the most common answer.
    *   **Tree-of-Thoughts (ToT):** Exploring multiple reasoning paths during generation.
*   **Refining Data Augmentation:**
    *   **Agentic Data Synthesis:** Developing a more integrated, self-improving loop for CoT generation, incorporating automated scoring and reflection iteratively.
    *   **RLHF/RLAIF for Reasoning:** Using reinforcement learning to directly optimize the quality or correctness of the generated thought processes.
*   **Broader Evaluation:** Testing on more diverse NLI benchmarks and performing more rigorous human evaluation of reasoning quality.
*   **Full Fine-Tuning:** If resources permit, compare QLoRA results with full fine-tuning.
*   **Multi-lingual NLI with CoT.**

## 8. Conclusion

*(Previously section 9)*
This project successfully demonstrated the fine-tuning of Mistral-7B for NLI tasks using Chain-of-Thought reasoning augmented by a novel reflection mechanism. By combining initial CoT generation with targeted correction of flawed reasoning using a stronger model, we created a high-quality dataset for QLoRA-based fine-tuning. The resulting models show promising NLI classification performance while providing interpretable step-by-step reasoning. Key contributions include the multi-stage data augmentation pipeline, insights into prompt engineering for concise CoT, and practical application of PEFT techniques for a complex reasoning task. This approach enhances both model accuracy and transparency in NLI.

*TODO: Add specific key quantitative finding if a single "best" model's accuracy is to be highlighted.*

## 9. References

*(Previously section 10)*
*TODO: Add actual citations to relevant papers (Mistral, QLoRA, CoT, NLI datasets, etc.) and tools (Hugging Face Transformers, PEFT, bitsandbytes).*
*   [Mistral-7B Paper]
*   [QLoRA Paper]
*   [Chain-of-Thought Paper - Wei et al., 2022 or similar]
*   [Relevant NLI dataset papers - e.g., SNLI, MNLI, ANLI if concepts were drawn]
*   [Hugging Face Transformers, PEFT, TRL, bitsandbytes library citations]

(Remaining sections to be completed in future updates. For practical implementation details, please refer to the corresponding documentation files: [DATA.md](DATA.md), [TRAINING.md](TRAINING.md), and [EVALUATION.md](EVALUATION.md))
*TODO: Add a note about reproducibility: "Code and the Reflection-CoT augmented dataset are available at [GitHub/Hugging Face Link]." *