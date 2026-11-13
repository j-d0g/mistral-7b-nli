#!/usr/bin/env python3
"""
Recompute the headline classification metrics from the result files.

Reads the per-example result JSON files produced by evaluate/sample_model.py
(downloaded with results/download_results.py from the public dataset
jd0g/nlistral-7b-results) and prints accuracy, macro-averaged precision,
recall and F1, and Matthews correlation coefficient (MCC) for each file.

It also checks that no (premise, hypothesis) pair in the evaluated split also
appears in the training CSV, so the reported numbers are on held-out data.

Usage (from the project root, after downloading data and results):
    python scripts/analysis/compute_test_metrics.py
    python scripts/analysis/compute_test_metrics.py --results results/nlistral-ablation1-test-labelled.json
"""
import argparse
import glob
import json
import os

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

DEFAULT_RESULTS = [
    "results/mistral-base-test-labelled.json",
    "results/nlistral-ablation0-test-labelled.json",
    "results/nlistral-ablation1-test-labelled.json",
    "results/nlistral-ablation2og-test-labelled.json",
    "results/nlistral-ablation0-dev-labelled.json",
    "results/nlistral-ablation1-dev-labelled.json",
    "results/nlistral-ablation2-dev-labelled.json",
]


def pair_key(premise, hypothesis):
    return (str(premise).strip(), str(hypothesis).strip())


def load_pairs(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, keep_default_na=False)
    return {pair_key(p, h) for p, h in zip(df["premise"], df["hypothesis"])}


def metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "precision_macro": 100 * precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": 100 * recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": 100 * f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": 100 * f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Recompute metrics from result files.")
    parser.add_argument("--results", nargs="*", default=None, help="Result JSON files (default: the published set)")
    parser.add_argument("--train-csv", default="data/original_data/train.csv")
    args = parser.parse_args()

    files = args.results or [f for f in DEFAULT_RESULTS if os.path.exists(f)]
    if not files:
        files = sorted(glob.glob("results/*-labelled.json"))
    train_pairs = load_pairs(args.train_csv)

    for path in files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data["results"]
        y_true = [int(r["true_label"]) for r in rows]
        y_pred = [int(r["predicted_label"]) for r in rows]
        m = metrics(y_true, y_pred)
        print(f"\n{path}")
        print(f"  model directory recorded in file: {data.get('model')}")
        print(f"  rows: {m['n']}")
        if train_pairs is not None:
            overlap = sum(pair_key(r["premise"], r["hypothesis"]) in train_pairs for r in rows)
            print(f"  rows whose (premise, hypothesis) also appears in {args.train_csv}: {overlap}")
        print(f"  accuracy        : {m['accuracy']:.2f}%")
        print(f"  precision (macro): {m['precision_macro']:.2f}%")
        print(f"  recall (macro)   : {m['recall_macro']:.2f}%")
        print(f"  F1 (macro)       : {m['f1_macro']:.2f}%")
        print(f"  F1 (weighted)    : {m['f1_weighted']:.2f}%")
        print(f"  MCC              : {m['mcc']:.3f}")


if __name__ == "__main__":
    main()
