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

### Model Selection for CoT Generation & Reflection

A two-stage approach was adopted for generating the CoT data used in fine-tuning:

1.  **Initial Thought Generation:** `open-mistral-7b` was used to generate the first pass of `thought_process` and `predicted_label` for the training/dev data.
    *   **Rationale:** Analysis suggested this 7B model was less prone to "overthinking" on this specific dataset. Simpler reasoning paths often correlated with the *correct* label according to the (sometimes subjective) ground truth. There was a strong correlation observed: fewer generated tokens often meant higher accuracy.
    *   **Goal:** Maximize alignment with the provided "gold standard" labels, even if some labels seemed questionable upon human review or when compared to stronger models.

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

### Training Script Cleanup

*   `run_training.sh` is the current, correct script for launching training.
*   `train.sh` is outdated and should be ignored/removed.

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

### The Interplay of LR, Batch Size, and Warmup

Further tuning revealed significant instability and sensitivity related to the learning rate schedule, particularly the warmup phase, when adjusting other parameters like batch size or total training duration:

*   **Warmup Ratio Sensitivity:** Initial successful runs (like the default configuration) used `warmup_ratio = 0.03`. For a 2-epoch run with the default dataset size (~2224 steps), this translated to roughly 67 warmup steps. This seemed to be a sweet spot *for that specific training length*.
*   **Dependency on Total Steps:** The core issue with `warmup_ratio` is that the *absolute number* of warmup steps changes if the total number of training steps changes (due to more epochs, different batch sizes, or different dataset sizes). A ratio that works well for a short run might be too short or too long for a longer run, potentially leading to instability or slow convergence.
*   **Batch Size Interaction:** Experiments that varied the effective batch size (e.g., Ablation 1 using 32 vs. Ablation 0 using 16) showed poor results when other parameters like learning rate weren't adjusted accordingly. A learning rate suitable for a small batch size (like 2e-4 for 16) could become unstable with a larger batch size (32) without a corresponding reduction or a different warmup profile. This highlighted the need to co-tune LR and batch size.
*   **Resuming Challenges:** Resuming training often failed to correctly restore the learning rate scheduler's state, leading to the scheduler restarting its warmup phase inappropriately mid-training. This caused sudden learning rate spikes, loss increases, and accuracy drops.

**Decision:** To decouple the warmup phase from the total training length and enable more consistent comparisons across experiments with varying epochs or batch sizes, the approach was shifted from using `warmup_ratio` to specifying a **fixed number of `warmup_steps`**. This provides more direct control and makes tuning the initial learning phase more predictable and reproducible, especially when resuming training or adjusting the total training duration. It also simplifies diagnosing issues related to the learning rate schedule.

---

## Sequence Length Optimization: A Double Win

Reducing the maximum sequence length processed by the model during fine-tuning and inference was a key optimization, driven by both performance observations and efficiency needs:

*   **Accuracy Correlation:** Initial analysis of the generated CoT data revealed a correlation between the length of the `thought_process` and the accuracy of the `predicted_label`. Thought chains resulting in a total input+output length of roughly 250-400 tokens showed the highest accuracy (around 85-95% correct predictions within this range). While longer chains (e.g., 500-950 tokens) sometimes produced correct predictions, they were less frequent and accuracy was slightly lower (70-80s%).
*   **Prompt Refinement for Brevity:** Recognizing this correlation, the prompt engineering process was specifically tuned to encourage the models (`open-mistral-7b` and `open-mistral-nemo`) to produce more concise yet effective reasoning chains. This involved refining instructions to favor shorter, direct logical steps.
    *   **Impact:** This prompt refinement *itself* led to an improvement in the accuracy of the initial thought generation (using `open-mistral-7b`), boosting it from ~65% to the 70-75% range.
*   **Setting the Max Length (512 tokens):** Since the refined prompts naturally led to shorter outputs without sacrificing (and in fact, improving) quality, it became feasible to set a maximum sequence length of 512 tokens for fine-tuning and inference. This comfortably accommodated the typical input lengths (premise + hypothesis + instruction prompt) and the desired concise thought processes (target range ~250-400 tokens total).
*   **The "Double Win":**
    1.  **Improved Accuracy:** Guiding the model towards conciseness via prompting enhanced the quality and accuracy of the generated reasoning.
    2.  **Massive Efficiency Gains:** Capping the sequence length at 512 (down from a potential default of 2048 or higher) dramatically reduced computational load.
        *   *Why?* Transformer attention complexity scales quadratically with sequence length. Shorter sequences mean significantly fewer calculations per token.
        *   *Memory:* Shorter sequences require much less GPU VRAM to store activations and the Key-Value (KV) cache during generation.
*   **Practical Benefits:** This optimization made training faster, allowed for larger batch sizes within the same VRAM budget, reduced inference time significantly (contributing to the ~3x speedup mentioned in the README), and increased the feasibility of running the process on available hardware.

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