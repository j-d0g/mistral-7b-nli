# Project Blog & Design Rationale

## CURRENT TASK OBJECTIVE (As of [Current Date])

**Goal:** Restructure project documentation for clarity and maintainability.

**Plan:**
1.  **Centralize Detail:** Consolidate all detailed explanations, rationale, experiments, findings, and methodology into the root `README.md`, using this `project_blog.md` as the primary drafting space.
2.  **Simplify Subdirectories:** Reduce `README.md` files within subdirectories (`data/`, `scripts/`, `train/`, `evaluate/`, etc.) to minimal, functional guides containing only:
    *   Brief purpose statement.
    *   Essential "quick start" / usage commands.
    *   A pointer to the root `README.md` for full details.
3.  **Current Focus:** Consolidate documentation for the **Data Augmentation** phase, involving the `data/` and `scripts/` directories and their respective READMEs.

---

This document captures the informal thoughts, decisions, challenges, and rationale behind the Mistral-7B NLI fine-tuning project. It serves as a log for context that might be useful for the final report, README, and explaining the work.

## Initial Decisions & Context (As of [Current Date])

### Why Chain-of-Thought (CoT) with an Autoregressive LLM for NLI?

The core task is a binary NLI classification. While traditional BERT-style models are common for this, I chose to use a fine-tuned autoregressive LLM (Mistral-7B) combined with CoT for several reasons:

*   **Interpretability:** NLI often involves subjective judgments and subtle reasoning. CoT provides a window into *how* the model reaches its classification (entailment/no-entailment), addressing potential label subjectivity and ambiguity inherent in the dataset. This moves beyond a simple classification score.
*   **SOTA & Novelty:** Using CoT generation and fine-tuning a large autoregressive model for a classification task is a more complex and arguably more state-of-the-art approach compared to standard encoder-based classification, fulfilling the coursework's aim for creativity and exploring recent techniques.
*   **Challenge:** It's a significantly harder path involving generation, structured output parsing, and careful prompt engineering, demonstrating a deeper engagement with LLM capabilities.
*   **Synergy with QLoRA Fine-Tuning:** A crucial motivation, tied to the choice of QLoRA for fine-tuning, was to maximize the synergy between the trained adapter and the frozen base model. Since QLoRA doesn't alter the base weights (which already possess strong NLI capabilities), the fine-tuning data should ideally complement, not contradict, the base model's inherent reasoning patterns. Prompting authentically *without* providing the true label initially allowed capturing Mistral-7B's natural approach. Training the adapter on this minimized the friction between the adapter and the base model, aiming for more coherent collaboration during inference. Even the later reflection step provided context (the original flawed reasoning) to guide corrections gently, rather than forcing sharp U-turns that might clash with the base model's underlying knowledge representation.

### Model Selection for CoT Generation & Reflection

A two-stage approach was adopted for generating the CoT data used in fine-tuning:

1.  **Initial Thought Generation:** `open-mistral-7b` was used to generate the first pass of `thought_process` and `predicted_label` for the training/dev data.
    *   **Rationale:** Analysis suggested this 7B model was less prone to "overthinking" on this specific dataset. Simpler reasoning paths often correlated with the *correct* label according to the (sometimes subjective) ground truth. There was a strong correlation observed: fewer generated tokens often meant higher accuracy.
    *   **Goal:** Maximize alignment with the provided "gold standard" labels, even if some labels seemed questionable upon human review or when compared to stronger models.
    *   <!-- TODO: Insert a concrete example here illustrating how simpler 7B reasoning aligns with a subjective label where a more complex model might diverge. Plan: Sample ~100 examples, generate with open-mistral-7b and a stronger model (e.g., deepseek-coder-v2-instruct or open-mistral-nemo), find cases where 7B is correct (per label) and the stronger model is incorrect. -->

2.  **Reflection on Errors:** For examples where `open-mistral-7b` initially predicted the *wrong* label, `open-mistral-nemo` (a stronger 12B model) was used for reflection.
    *   **Rationale:** When the 7B model failed, a more powerful reasoning model was needed to generate a *corrected* thought process leading to the *true* label (which was provided during the reflection prompt). The 12B model was deemed better suited to exploring diverse reasoning paths required to "fix" the initial incorrect logic.

### Fine-tuning Data Strategy: Combining Strengths

The final fine-tuning dataset isn't based on separate ablations anymore. Instead, it strategically combines:

*   Examples where the *initial* `open-mistral-7b` generation produced the **correct** label and thought process.
*   Examples where the *initial* generation was **incorrect**, but were subsequently **corrected** through reflection using `open-mistral-nemo`.

**Rationale:** This approach aims to create the highest quality training data:
*   It leverages the 7B model's outputs when they aligned with the ground truth.
*   It incorporates the stronger 12B model's reasoning specifically to teach the target model how to handle cases it initially got wrong.
*   This uses the *entire* original NLI dataset, ensuring the model is trained on the full distribution of provided examples, rather than filtering based on the initial model's success rate. The goal is to learn from *all* examples, especially the difficult ones, using the best available reasoning (original or reflected) for each.

### Overarching Goal: Coursework Success

The primary driver is a university coursework assignment requiring high accuracy on a hidden NLI test set, creativity in approach, and comprehensive documentation (poster, model card, README). The chosen CoT/autoregressive LLM path aims to satisfy these requirements, particularly the creativity and depth aspects, while also pushing for strong performance.

### Base Model Choice: Why Mistral-7B?

The selection of `mistralai/Mistral-7B-v0.3` wasn't arbitrary. Several factors made it the most suitable choice for this project's goals and constraints:
*   **Performance/Cost Ratio:** It offered a compelling balance. While larger models exist, 7B parameters provided significant capability without exorbitant training/inference costs or hardware demands, especially when using QLoRA. API access was also reasonably priced for the extensive CoT generation phase.
*   **Inference Speed:** For a 7B model, its inference speed is relatively fast, which was important considering the need to process thousands of examples for evaluation.
*   **Reasoning Prowess:** Benchmarks and community consensus indicated strong reasoning abilities compared to other open-source models in the same weight class (e.g., Llama 2 7B variants available at the time).
*   **Empirical Validation:** Crucially, early tests on a sample of the NLI dataset showed `open-mistral-7b` (the API version) achieving the best accuracy and precision compared to other models tested in the same size category.
*   **API Ecosystem:** The availability of different Mistral models through their API (like the base `open-mistral-7b` and the stronger `open-mistral-nemo`) was a key enabler for the two-stage data generation process (initial generation + reflection on errors), allowing the use of the most appropriate tool for each sub-task.

### Training Script Cleanup

*   `run_training.sh` is the current, correct script for launching training.
*   References to `train.sh` in documentation have been updated - this script was removed to avoid confusion.

### Data Source

*   The raw NLI data and generated thoughts (original/reflected) are downloaded from the Hugging Face dataset repository `jd0g/Mistral-Thinking-NLI`. (Note: Model checkpoints are in `jd0g/Mistral-v0.3-Thinking_NLI`).

## Prompt Engineering for CoT Generation

Developing the prompts for both the initial thought generation (`scripts/generate_thoughts.py`) and the reflection (`scripts/generate_thoughts_reflected.py`) involved significant experimentation:

*   **Cross-Model Testing:** Initial tests were conducted across various API providers and models, including those from OpenAI, Mistral AI, Meta (Llama), and Anthropic.
*   **Model Selection:** `mistral-7b` (specifically `open-mistral-7b`) emerged as a strong performer, particularly considering its cost per token. It often provided concise and effective reasoning, outperforming even some larger models for this specific NLI task where simpler logic seemed beneficial.
*   **Prompt Structuring:** Best practices from Anthropic's documentation were adopted, using Markdown and XML tags to clearly delineate different parts of the prompt (like instructions, premise, hypothesis).
*   **Output Format:** JSON was chosen for the output structure (`{"thought_process": "...", "predicted_label": ...}`) to allow for easy programmatic parsing. Providing a few-shot example within the prompt proved crucial for ensuring the model consistently adhered to this JSON format and maintained a consistent reasoning style.
*   **CoT Guidance:** Light research into effective strategies for solving NLI problems informed the step-by-step instructions within the CoT prompt. Clear, simple, and direct instructions yielded the best results in guiding the model's reasoning process.
*   **Experimentation Scope:** These prompt variations were tested on a sub-sample of 1000 examples, with the total cost for this experimentation phase estimated at around £5.

This iterative process led to the final prompt structures used to generate the CoT data for fine-tuning.

---

## Training Evolution & QLoRA Parameter Rationale

The fine-tuning process wasn't linear and involved several iterations based on observed results:

1.  **Initial Attempt (Correct Predictions Only):**
    *   The first fine-tuning runs used only the subset of the data where the initial `open-mistral-7b` CoT generation produced the *correct* label (~65-70% of the data).
    *   **QLoRA Settings:** Started with commonly recommended settings for instruction tuning: `r=16`, `lora_alpha=32`, `dropout=0.05`.
    *   **Outcome:** This approach suffered significantly. The dataset was heavily biased towards the initially correct predictions (often simpler examples or one particular class like no-entailment). The model overfit very quickly (often before completing a single epoch) and converged to suboptimal performance.

2.  **Problem Diagnosis & The Reflection Idea:**
    *   Analysis pointed towards data bias and the inherent subjectivity of some NLI labels. Attempts to automatically score or "gold-standardize" the generated thoughts using stronger models proved very expensive (estimated >£100 in API costs) and often resulted in models getting stuck, unable to improve reasoning further.
    *   This led to the core idea of **reflection**: Instead of trying to fix the *thoughts* for correct labels, focus on fixing the *reasoning* for incorrect predictions by providing the *true* label and prompting a stronger model (`open-mistral-nemo`) to generate a valid reasoning path.

3.  **Second Attempt (Combined Correct + Reflected Data):**
    *   The fine-tuning dataset was recreated using the strategy outlined earlier: combining the initially correct examples with the newly generated reflected examples for the initially incorrect ones.
    *   **Impact:** This effectively doubled the usable training data, created a more balanced class distribution, and introduced more diverse and complex reasoning paths (especially for the harder examples the initial model failed on).
    *   **QLoRA Settings & Outcome:** Using the same initial QLoRA settings (`r=16`, `lora_alpha=32`, `dropout=0.05`), the model now showed much better training dynamics. It continued to converge and improve validation loss well past the first epoch, sustaining learning for up to 3 epochs.

4.  **Experimenting with Higher Rank (Addressing Complexity Concerns):**
    *   There was a hypothesis that the added complexity from the reflected data (potentially introducing reasoning styles conflicting with the initial model's) might benefit from increased capacity in the LoRA adapters.
    *   An ablation was run using **doubled rank and alpha (`r=32`, `lora_alpha=64`)**, along with adjustments like increased batch size and gradient clipping, to see if this captured the complexity better.
    *   **Outcome:** While this configuration also trained successfully, it didn't significantly outperform the simpler `r=16`/`alpha=32` setup. It suggested the initial, standard settings were sufficient to capture the nuances introduced by the reflected data, despite the added complexity.
    *   **(Note:** The final configuration documented in the main README settled on `r=32`, `lora_alpha=64`. This suggests that while the simpler settings were *sufficient*, the higher rank might have been chosen for the final models for robustness or slight edge, even if the difference wasn't dramatic in initial tests).

This iterative process demonstrates the importance of data quality and diversity. Addressing the shortcomings of the initial data generation (via reflection) was more impactful than simply increasing model capacity through higher LoRA ranks.

5.  **Revisiting Larger Scale with Stability (Ongoing Success):**
    *   **Hypothesis Revisited:** Based on the complexity introduced by the reflected data (diverse, potentially conflicting reasoning) and the observation that even simpler models kept improving past epoch 1, the hypothesis that more capacity (`r=32/a=64`) and longer training were needed gained strength.
    *   **Key Insight:** The initial failure with larger batch sizes/higher rank likely stemmed from instability caused by the noisy/complex nature of the combined dataset.
    *   **The Fix:** The larger scale experiment was re-run, but this time incorporating crucial stability measures:
        *   **Increased Effective Batch Size (e.g., to 64):** To average out noise from diverse examples within each batch.
        *   **Gradient Clipping:** To prevent gradient explosions.
        *   **Lower Learning Rate:** Tuned appropriately for the larger batch size and potential noise.
    *   **Outcome:** This refined, stabilized approach with higher rank (`r=32/a=64`) and larger effective batch size is currently yielding the best training results, validating the hypothesis that the complex reflected dataset benefits from more capacity *when training is properly stabilized*.
    *   **(Note:** This clarifies the final choice of `r=32/a=64` documented elsewhere - it wasn't just arbitrary but the result of iterative refinement and adding necessary stability controls).

6.  **The Critical Impact of Epoch Count on Training Dynamics:**
    *   A crucial lesson learned was that changing the number of epochs doesn't just affect training duration - it fundamentally transforms the entire learning process.
    *   **Warmup Ratio Interaction:** When using `warmup_ratio` instead of fixed `warmup_steps`, doubling the epochs halves the effective warmup period relative to total training time. This can dramatically change how the model begins learning.
    *   **Learning Rate Decay Effects:** With schedulers like cosine decay, changing epoch count stretches or compresses the decay curve. Adding epochs to an existing training run doesn't simply continue the previous trajectory - it redefines the entire learning rate path.
    *   **Practical Impact:** Attempts to extend promising but prematurely ended ablations by adding more epochs often failed because the additional epochs altered the careful balance of warmup and decay that had made the original configuration successful.
    *   **Best Practice:** Start with conservative epoch estimates and incrementally add epochs in new training runs rather than dramatically increasing them. This preserves the warmup/decay balance that worked in the initial configuration.

This interplay between epoch count, learning rate scheduling, and training dynamics highlights the complex interdependencies in the fine-tuning process. What appears to be a simple duration parameter actually reshapes the entire optimization landscape.

### Navigating Label Disagreement: A Core Challenge

A significant challenge arose from the ~30% of examples where the initial `open-mistral-7b` model's prediction disagreed with the provided dataset label. This led to a critical decision point with several options:

1.  **Keep Original:** Trust the model's reasoning (which often seemed plausible) and keep the original prediction/thought, despite disagreeing with the label.
2.  **Omit:** Discard these disagreeing examples entirely.
3.  **Correct Naturally:** Attempt to guide the model towards generating a new thought process that naturally leads to the *correct* label, without explicitly forcing it.
4.  **Correct Forcefully:** Provide the model with the *correct* label and instruct it to generate a reasoning path for it.

Initially, options 1 and 2 were appealing, as personal review often sided with the model's logic over the dataset label, suggesting potential label subjectivity or ambiguity. However, acknowledging the coursework requirement to treat the dataset as the **"gold standard"** and maximize performance on its hidden test set, a pragmatic decision was made. The objective shifted from finding absolute NLI truth to effectively learning the dataset's specific (potentially imperfect) mapping. This converted a subjective debate into a tractable ML problem for the task context.

An attempt was made to implement **Option 3** using an iterative self-scoring and self-improvement pipeline. The idea was to use automated scoring and repeated prompting (with increasingly powerful models) to gently nudge the reasoning towards the correct label. This proved **prohibitively expensive and often ineffective**. For many examples, even powerful models struggled to find a coherent, natural reasoning path to the target label, reinforcing suspicions about label quality and highlighting the limits of purely automated refinement in such cases.

This led to the **final reflection strategy (`scripts/generate_thoughts_reflected.py`)**, which represents a **hybrid of Options 3 and 4**. It *forcefully* provides the target label (acknowledging the gold standard constraint) but *also* provides the original flawed reasoning as context (aiming for a more natural, contextual correction). This proved to be a more effective and efficient compromise, allowing the project to leverage the full dataset while attempting to generate the most plausible reasoning *given* the target label.

The process of identifying this issue, experimenting with solutions, and adapting the strategy based on empirical results (including the costly failure of Option 3) was a key learning experience, ultimately leading to a more robust data generation pipeline better suited to addressing the early overfitting issues encountered with the initial, smaller dataset.

### Reflection Prompting Deep Dive: Guiding Nemo

Choosing *how* to prompt the reflection model (`open-mistral-nemo`) was a critical decision point. Two main options were considered:

1.  **Simple Correction:** Provide the premise, hypothesis, and the *true label*, asking for a new thought process similar to the initial generation.
2.  **Guided Reflection:** Provide the premise, hypothesis, true label, *and the original incorrect thought process* from `open-mistral-7b`. Explicitly ask the model to analyze the initial mistake and generate a corrected reasoning path.

Option 2 was chosen for several reasons:

*   **Addressing Subjectivity:** Given that the dataset labels sometimes felt subjective (and the initial 7B reasoning often seemed plausible even when wrong), merely providing the true label risked the reflection model generating forced, unnatural reasoning just to match the target. We often agreed more with the 7B model's initial prediction than the label itself.
*   **Correcting Logic, Not Just Label:** By showing Nemo the flawed 7B logic, the prompt encouraged it to identify *where* the reasoning failed and construct a fundamentally sounder path to the correct label, rather than just any path.
*   **Leveraging Nemo's Strength for the Reflection Task:** This complex task of analyzing another model's error and performing a grounded correction required a stronger reasoning engine. The act of reflection itself demands a capacity to understand diverse viewpoints and identify subtle logical gaps, necessitating a model like `open-mistral-nemo` (12B) with a broader knowledge base and more flexible reasoning capabilities. It needed the "open-mindedness" to see paths the smaller 7B model might have missed. It wasn't just about general strength, but suitability for this specific meta-reasoning task. Nemo wasn't used for initial generation precisely because its more detailed reasoning often led to lower alignment with the dataset's simpler logic preference. Using it here allowed its power to be focused on the most challenging examples, guided by the 7B's prior attempt.
*   **Mimicking Style:** The hope was that by seeing the 7B's (flawed) attempt, Nemo could use its sophistication to generate a corrected reasoning chain that still somewhat mimicked the desired concise style, rather than defaulting to an overly complex explanation.

This approach represents a deliberate attempt to inject higher-quality reasoning specifically where the initial, simpler model failed, tackling the difficult tail end of the data distribution.

### The Interplay of LR, Batch Size, and Warmup

Further tuning revealed significant instability and sensitivity related to the learning rate schedule, particularly the warmup phase, when adjusting other parameters like batch size or total training duration:

*   **Warmup Ratio Sensitivity:** Initial successful runs (like the default configuration) used `warmup_ratio = 0.03`. For a 2-epoch run with the default dataset size (~2224 steps), this translated to roughly 67 warmup steps. This seemed to be a sweet spot *for that specific training length*.
*   **Dependency on Total Steps:** The core issue with `warmup_ratio` is that the *absolute number* of warmup steps changes if the total number of training steps changes (due to more epochs, different batch sizes, or different dataset sizes). A ratio that works well for a short run might be too short or too long for a longer run, potentially leading to instability or slow convergence.
*   **Batch Size Interaction:** Experiments that varied the effective batch size (e.g., Ablation 1 using 32 vs. Ablation 0 using 16) showed poor results when other parameters like learning rate weren't adjusted accordingly. A learning rate suitable for a small batch size (like 2e-4 for 16) could become unstable with a larger batch size (32) without a corresponding reduction or a different warmup profile. This highlighted the need to co-tune LR and batch size.
*   **Resuming Challenges:** Resuming training often failed to correctly restore the learning rate scheduler's state, leading to the scheduler restarting its warmup phase inappropriately mid-training. This caused sudden learning rate spikes, loss increases, and accuracy drops.

**Decision:** To decouple the warmup phase from the total training length and enable more consistent comparisons across experiments with varying epochs or batch sizes, the approach was shifted from using `warmup_ratio` to specifying a **fixed number of `warmup_steps`**. This provides more direct control and makes tuning the initial learning phase more predictable and reproducible, especially when resuming training or adjusting the total training duration. Experimentally, the successful larger-scale runs (`r=32/a=64`, effective batch size 64) performed best with `warmup_steps` around **150**, while earlier, smaller-scale runs were more stable with `warmup_steps` in the **50-75** range. It also simplifies diagnosing issues related to the learning rate schedule.

---

## Sequence Length Optimization: A Double Win

Reducing the maximum sequence length processed by the model during fine-tuning and inference was a key optimization, driven by both performance observations and efficiency needs:

*   **Accuracy Correlation:** Initial analysis of the generated CoT data revealed a correlation between the length of the `thought_process` and the accuracy of the `predicted_label`. Thought chains resulting in a total input+output length of roughly 250-400 tokens showed the highest accuracy (around 85-95% correct predictions within this range). While longer chains (e.g., 500-950 tokens) sometimes produced correct predictions, they were less frequent and accuracy was slightly lower (70-80s%). <!-- TODO: Add note about lost Jupyter notebook containing detailed analysis correlating steps/length/accuracy. -->
*   **Prompt Refinement for Brevity:** Recognizing this correlation, the prompt engineering process was specifically tuned to encourage the models (`open-mistral-7b` and `open-mistral-nemo`) to produce more concise yet effective reasoning chains. This involved:
    *   Adding explicit instructions like "Keep your reasoning brief."
    *   Analyzing successful responses and identifying a common pattern of 3 core reasoning steps. The prompt structure was then modified to explicitly request exactly 3 steps, guiding the model towards this effective pattern.
    *   Implementing a separate (though ultimately not used for final data) scoring and self-improvement prompt loop where brevity and conciseness were criteria for high-scoring thoughts, reinforcing the desired output style during experimentation.
    *   **Impact:** This prompt refinement *itself* led to an improvement in the accuracy of the initial thought generation (using `open-mistral-7b`), boosting it from ~65% to the 70-75% range.
*   **Setting the Max Length (512 tokens) - Post-Analysis Optimization:** Crucially, the `max_seq_length` was NOT set arbitrarily before generation. After generating the *entire* dataset using the refined, brevity-focused prompts, analysis confirmed that *no generated example exceeded 400 tokens*. Therefore, setting `max_seq_length=512` for fine-tuning and inference was a safe and highly effective optimization. It capitalized on the successful prompt engineering to drastically reduce computational requirements, rather than forcefully truncating potentially valid reasoning.
*   **The "Double Win":**
    1.  **Improved Accuracy:** Guiding the model towards conciseness via prompting (especially the 3-step structure) enhanced the quality and accuracy of the generated reasoning.
    2.  **Massive Efficiency Gains:** Capping the sequence length at 512 (justified by the observed max length <400) dramatically reduced computational load compared to the default 2048.
        *   *Why?* Transformer attention complexity scales quadratically with sequence length. Shorter sequences mean significantly fewer calculations per token.
        *   *Memory:* Shorter sequences require much less GPU VRAM to store activations and the Key-Value (KV) cache during generation.
*   **Practical Benefits:** This optimization made training faster, critically freed up GPU memory enabling exploration of other hyperparameters (larger batch sizes, higher LoRA rank/alpha) that would otherwise have been infeasible, reduced inference time significantly (contributing to the ~3x speedup mentioned in the README), and increased the overall feasibility of running the project on available hardware.

This strategic reduction in sequence length, enabled by successful prompt engineering for conciseness, was therefore crucial for both model performance and project feasibility.

---

## Docker for Reproducibility and Portability

The extensive use of Docker throughout the project, particularly for training and inference, stemmed from the need for a consistent and portable environment, especially when dealing with GPU dependencies:

*   **Initial Phase (Local Feasibility):** The initial data generation step, which involved calling external APIs (like Mistral's) to generate CoT data, was computationally less demanding and could be feasibly run locally using CPU-based parallel processing.
*   **The GPU Dependency Challenge:** Fine-tuning the Mistral-7B model and running optimized inference required specific GPU hardware and, critically, a complex stack of compatible software dependencies. This included precise versions of CUDA, PyTorch, and specialized libraries like `transformers`, `peft`, `trl`, and `bitsandbytes` (for quantization). Ensuring this exact environment could be replicated reliably across different potential execution platforms (personal workstations with GPUs, cloud services like Runpod, university clusters like Kilburn CSF) became a major hurdle – the classic "dependency hell".
*   **Docker as the Solution:** Docker was adopted as the solution to guarantee that the training and inference code would run consistently regardless of the underlying host system. By containerizing the application and all its dependencies (including CUDA toolkit versions managed within the image), the setup became portable and reproducible.
*   **Scope of Docker Usage:**
    *   **Essential:** Docker is strictly necessary for running the `run_training.sh` and `run_inference.sh` scripts, as these rely on the containerized GPU environment.
    *   **Convenience:** While not technically required for the API-based data generation scripts (`generate_thoughts.py`, `generate_thoughts_reflected.py`), the data and model download scripts (`data/download_data.py`, `models/download_model.py`) were also provided with Docker wrappers. This allows a user to set up the *entire* project environment and download all necessary artifacts using only Docker commands, streamlining the setup process even if they choose to run generation locally later.

---

## Configuration System for Training (NanoGPT-inspired)

The choice to implement a Python-based configuration system (`train/configs/`, `train/config_loader.py`) for managing training parameters, rather than relying solely on CLI arguments or bash variables, was driven by prior experience and the need for better experiment management:

*   **Learning from Past Projects:** Experience gained from a previous project (pre-training Chess-playing LLMs using NanoGPT) highlighted the challenges of managing numerous hyperparameter tuning experiments using only bash scripts and command-line parsing. This approach quickly became unstructured and difficult to track.
*   **Pain Point Addressed:** The primary goal was to move away from potentially long, complex, and error-prone bash commands for launching training runs with different settings.
*   **Configs as Self-Documenting Experiments:** Using separate Python configuration files (e.g., `train/configs/ablation1.py`, `train/configs/default.py`) provides a clear, readable, and version-controllable record of each experimental setup. This makes it easy to revisit, understand, and reproduce specific hyperparameter combinations later.
*   **Improved Readability & Maintainability:** Python configuration files are generally more readable and easier to maintain than complex bash logic for parameter handling. Comments can be added easily to explain specific choices.
*   **Flexibility and Control:** This system allows for easy modification of various parameters (learning rate, batch size, LoRA settings, GPU IDs, distributed training flags, output directories, wandb usage, etc.) within the config files, while still allowing for quick overrides via the command line for specific tweaks.
*   **Simplified Entry Point:** It enables a much cleaner main training script (`run_training.sh`). This script's primary responsibility becomes launching the Docker container and passing the chosen config file path (and any CLI overrides) to the main Python training script (`train/train_sft.py`). The Python script then handles the logic of loading the base config, applying overrides, and using the final parameters.

---

## Appendix: COMP34812 Coursework Task & Marking Scheme (AY 2024-25)

*This section includes the relevant parts of the official coursework description and marking rubric to provide context for the project's goals and evaluation criteria.*

### **Introduction**
- **Task**: Shared task on pairwise sequence classification.
- **Mode**: Closed (only provided datasets allowed).
- **Track Chosen**: Natural Language Inference (NLI)
  - **Description**: Given a premise and a hypothesis, determine if the hypothesis is true based on the premise.
  - **Data**: >24K training pairs, >6K validation pairs (hidden test set for final evaluation).

### **Solutions Requirements**
- Develop **two** solutions from different categories (A: Traditional ML, B: Non-Transformer DL, C: Transformer DL).
- *Note: This repository focuses solely on **one solution** falling under **Category C (Transformer DL)**, specifically fine-tuning Mistral-7B with Chain-of-Thought. The second solution has been completed by my partner*
- Aim to outperform provided baseline methods.

### **Marking Rubric (40 marks total)**

*The following highlights categories directly relevant to the single Transformer-based solution developed in this project.*

| Category | Subcategory | Description | Marks | Relevance Notes |
|----------|-------------|-------------|-------|-----------------|
| **System Predictions** | Competitive performance (Solution C) | Statistically significant improvement over baseline. | 2 | Primary goal: High accuracy on hidden test set. |
| **Implementation** | Organisation and documentation | Well-documented and structured code. | 2 | **Key focus of current documentation effort.** |
|  | Completeness and Reproducibility | All resources provided for reproducibility. | 2 | Docker usage, download scripts, config system contribute here. |
|  | Soundness (Solution C) | Technically sound design. | 2 | CoT, reflection, QLoRA, optimizations aim for this. |
|  | Creativity (Solution C) | Adventurous and state-of-the-art approach. | 2 | CoT for NLI, reflection mechanism target this. |
| **Evaluation** |  | Effort beyond the supporting benchmarking tool (Codabench). | 2 | Potentially analyzing reasoning quality, error types, etc. |
| **Model Cards** | Formatting | Correct format for the model card. | 2 | Deliverable needed. |
|  | Informativeness (Solution C) | Sufficient description for reuse. | 2 | Deliverable needed. |
|  | Accurate representation | Accurately represents the implemented solution. | 2 | Deliverable needed. |
| **Flash Presentation** | Live demo (Solution C) | Demo code works out of the box. | 2 | `run_inference.sh` with sample data should support this. |
|  | Poster content | Informative and stand-alone. | 2 | Deliverable needed. |
|  | Poster aesthetics | Visually appealing with good use of visuals. | 2 | Deliverable needed. |
|  | Poster presentation | Engaging and concise explanation. | 2 | Presentation skill. |
|  | Q&A (Solution C) | Satisfactory answers to questions. | 2 | Depends on understanding documented here. |

*(Note: Marks related to a second solution, Q&A for it, etc., are omitted for clarity as they are not covered by this specific project focus.)*

---

## Evaluation Strategy Notes

### Qualitative Analysis of Thoughts

While the generated `thought_process` offers a potential window into the model's reasoning, performing rigorous *qualitative* analysis (e.g., categorizing reasoning types, identifying common flaws) is complex and currently considered out-of-scope for the primary project goals. Instead, the focus is on using the CoT generation as a mechanism to **improve quantitative metrics** (primarily accuracy on the NLI task). These quantitative metrics serve as an indirect proxy for the quality of the underlying reasoning.

An **LLM-as-a-judge** scoring mechanism *was* developed (`scripts/score_thoughts.py`) to automatically assess thought quality based on criteria like coherence, correctness, and brevity. While experiments showed this could be effective to some degree, results must be interpreted cautiously due to the inherent biases and potential inconsistencies of using LLMs for evaluation, especially given the subjective nature of some NLI examples encountered.

### Robust Prediction Extraction

Early evaluation attempts were hindered by flawed logic for extracting the final `predicted_label` from the model's generated output string, which included the JSON CoT. A significant issue arose from setting `tokenizer.pad_token = tokenizer.eos_token` during training/generation. This configuration sometimes caused the model to generate repetitive, non-JSON text after the intended JSON output, simply to fill the sequence up to the maximum length before the EOS token. The initial parsing logic failed to handle these malformed strings correctly, leading to inaccurate performance reporting.

The parsing logic in the evaluation scripts (`evaluate/sample_model.py` and helpers) was subsequently **corrected** to robustly identify and extract the intended JSON structure, even in the presence of such trailing padding or other potential generation artifacts. This fix was crucial for obtaining reliable performance metrics for the CoT-generating models.

### Training Metrics vs. Final Evaluation

During fine-tuning, the `SFTTrainer` in `train_sft.py` primarily monitors **training loss**, **validation loss**, and **token accuracy**. It does not currently implement a custom `compute_metrics` function to calculate classification metrics like Precision, Recall, and F1 during the training loop. The definitive evaluation of NLI classification performance (Accuracy, P/R/F1 if labels are available) relies on running the dedicated inference script (`run_inference.sh`) on the relevant dataset (e.g., dev set, hidden test set).

---

## Next Steps, Loose Ends, and Future Directions

Reflecting on the project's progress and findings, several potential avenues for improvement, necessary wrap-up tasks, and future explorations emerge:

### Potential Next Steps & Improvements

*   **Agentic Data Synthesis Pipeline:** Explore transforming the current sequential data generation process into a more integrated, self-improving loop. This could involve:
    *   Further refining the LLM-as-a-judge scorer (`scripts/score_thoughts.py`).
    *   Integrating automated scoring and reflection *during* the initial thought generation phase, allowing for iterative self-correction until reasoning meets certain quality criteria (e.g., coherence, brevity, logical soundness).
    *   The goal would be a self-sufficient pipeline aiming to maximize the quality of the generated CoT data, potentially leading to better downstream classification performance and lower training loss.
*   **Deeper Qualitative Analysis:** If time permits, perform a more structured qualitative analysis on a sample of the generated `thought_process` outputs (both original and reflected) to categorize common reasoning patterns, error types, and assess the impact of reflection more granularly.

### Alternative Reasoning Paradigms (Beyond Current Scope)

*   **Multi-Path Reasoning:** Investigate generating multiple, diverse reasoning chains for each NLI example, particularly for those identified as subjective or ambiguous during initial analysis. This could potentially improve model robustness and generalization.
*   **Advanced Reasoning Frameworks:** Explore connections to or implementations of frameworks like Tree-of-Thoughts (ToT) for exploring branching reasoning paths, or employ modern Reinforcement Learning techniques (e.g., approaches similar to GRPO or others focused on optimizing reasoning policies) as a post-training refinement step for the generated thoughts.

### Loose Ends (Pre-Submission Checklist)

*   **Results Collation:** Systematically gather and organize all quantitative results from various experiments:
    *   Model performance metrics (Accuracy, Loss) for different ablations/checkpoints on dev/test sets.
    *   Data augmentation statistics (e.g., number of examples generated, number reflected, average thought length).
    *   Findings from any data analysis (even if anecdotal due to lost artifacts).
    *   Results from the LLM-as-a-judge scorer, if used for analysis.
*   **Visualization for Communication:** Create clear and informative visualizations:
    *   Tables comparing model performance across different configurations.
    *   Graphs showing training/validation loss curves, accuracy trends.
    *   Diagrams illustrating the key processes: the two-stage CoT generation pipeline (initial + reflection), the QLoRA fine-tuning setup, the evaluation workflow.

### Future Research Directions

*   Currently, the focus remains on completing the coursework objectives. However, the effectiveness of the reflection mechanism and the challenges of subjective labels in NLI datasets could inspire future investigations.

--- 