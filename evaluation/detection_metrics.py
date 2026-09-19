#!/usr/bin/env python3
"""Compute label-separated ACC, macro-F1, ROC AUC, and EER from JSONL scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def roc_metrics(labels: list[int], scores: list[float]) -> tuple[float, float, float]:
    order = np.argsort(-np.asarray(scores)); y = np.asarray(labels, dtype=int)[order]
    s = np.asarray(scores, dtype=float)[order]
    positives, negatives = max(1, int(y.sum())), max(1, len(y) - int(y.sum()))
    tp = fp = 0; points = [(0.0, 0.0, float("inf"))]
    for index, (label, score) in enumerate(zip(y, s)):
        tp += int(label); fp += 1 - int(label)
        if index == len(y) - 1 or score != s[index + 1]:
            points.append((fp / negatives, tp / positives, float(score)))
    fpr = np.asarray([x[0] for x in points]); tpr = np.asarray([x[1] for x in points])
    fnr = 1.0 - tpr; position = int(np.argmin(np.abs(fpr - fnr)))
    return float(np.trapezoid(tpr, fpr)), float((fpr[position] + fnr[position]) / 2), points[position][2]


def macro_f1(truth: list[str], predictions: list[str]) -> float:
    scores = []
    for label in ("genuine", "spoof"):
        tp = sum(a == b == label for a, b in zip(truth, predictions))
        fp = sum(a != label and b == label for a, b in zip(truth, predictions))
        fn = sum(a == label and b != label for a, b in zip(truth, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True,
                        help="JSONL with file_id and spoof_score only")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Private JSONL with file_id and label; joined only after inference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = {r["file_id"]: r["label"] for r in map(json.loads, args.labels.open())}
    rows = [json.loads(line) for line in args.scores.open() if line.strip()]
    if any(key in row for row in rows for key in ("label", "attack_type")):
        raise ValueError("Inference score file must not contain labels or attack types")
    truth = [labels[row["file_id"]] for row in rows]
    numeric = [int(value == "spoof") for value in truth]
    scores = [float(row["spoof_score"]) for row in rows]
    auc, eer, threshold = roc_metrics(numeric, scores)
    predicted = ["spoof" if score >= threshold else "genuine" for score in scores]
    report = {
        "n": len(rows), "accuracy": float(np.mean(np.asarray(truth) == predicted)),
        "macro_f1": macro_f1(truth, predicted), "roc_auc": auc, "eer": eer,
        "threshold": threshold, "threshold_note": "Select on development data for test use",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
