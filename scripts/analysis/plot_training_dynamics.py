#!/usr/bin/env python3
"""
Plot real training dynamics from the Hugging Face Trainer logs.

Each training run saved a `trainer_state.json` in its checkpoint directories.
Those files are public in the model repository jd0g/Mistral-7B-NLI-v0.3. This
script downloads one per run, and plots validation loss and validation
mean-token-accuracy (the token-level accuracy reported by SFTTrainer, not the
classification accuracy) against training step.

The default runs are the ones behind the published test results:
  ablation 0 -> mistral-thinking-abl0-ext/checkpoint-3402
  ablation 1 -> mistral-thinking-default-epochs2/checkpoint-2000
  ablation 2 -> mistral-thinking-abl3/checkpoint-1275

Usage (no token needed, the repository is public):
    python scripts/analysis/plot_training_dynamics.py
    python scripts/analysis/plot_training_dynamics.py --output metrics/training_dynamics.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

REPO_ID = "jd0g/Mistral-7B-NLI-v0.3"
RUNS = {
    "Ablation 0 (mistral-thinking-abl0-ext)": "mistral-thinking-abl0-ext/checkpoint-3402/trainer_state.json",
    "Ablation 1 (mistral-thinking-default-epochs2)": "mistral-thinking-default-epochs2/checkpoint-2000/trainer_state.json",
    "Ablation 2 (mistral-thinking-abl3)": "mistral-thinking-abl3/checkpoint-1275/trainer_state.json",
}
COLORS = ["#7B61FF", "#4EAEFF", "#E91E63"]


def load_state(path_in_repo):
    local = hf_hub_download(repo_id=REPO_ID, filename=path_in_repo, repo_type="model")
    with open(local, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot training dynamics from trainer_state.json logs.")
    parser.add_argument("--output", default="metrics/training_dynamics.png")
    args = parser.parse_args()

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
    for (label, path), color in zip(RUNS.items(), COLORS):
        state = load_state(path)
        evals = [e for e in state["log_history"] if "eval_loss" in e]
        steps = [e["step"] for e in evals]
        ax_loss.plot(steps, [e["eval_loss"] for e in evals], marker="o", ms=3, color=color, label=label)
        ax_acc.plot(steps, [100 * e["eval_mean_token_accuracy"] for e in evals], marker="o", ms=3, color=color, label=label)
        print(f"{label}: {len(evals)} evaluations, final eval_loss={evals[-1]['eval_loss']:.4f}, "
              f"final eval_mean_token_accuracy={100 * evals[-1]['eval_mean_token_accuracy']:.2f}%")

    ax_loss.set_title("Validation loss")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    ax_acc.set_title("Validation mean token accuracy")
    ax_acc.set_xlabel("Training step")
    ax_acc.set_ylabel("Token accuracy (%)")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(fontsize=8, loc="lower right")
    fig.suptitle("Training dynamics from trainer_state.json (dev set, evaluated every 250 steps)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
