"""Recompute immunogenicity metrics from the saved test scores and labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA = Path(__file__).resolve().parent / "datasets" / "immunogenicity" / "test_scores.json"


def precision_curve(labels, scores):
    """Expected precision at ranks 1 to n, where n is the positive count."""
    positives = int(labels.sum())
    if not positives:
        return np.array([], dtype=float)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(labels)]
    expected_hits = np.empty(len(labels), dtype=float)
    hits_before = 0.0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        hits = float(sorted_labels[start:stop].sum())
        expected_hits[start:stop] = hits_before + np.arange(1, stop - start + 1) * hits / (stop - start)
        hits_before += hits
    return expected_hits[:positives] / np.arange(1, positives + 1)


def evaluate(rows):
    labels = np.asarray([row["label"] for row in rows])
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    if not np.isin(labels, (0, 1)).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    positives = int(labels.sum())
    result = {"n": len(rows), "positive": positives, "negative": len(rows) - positives}
    if not rows:
        return result
    curve = precision_curve(labels, scores)
    calls = scores >= 0.5
    both_classes = 0 < positives < len(labels)
    result.update(
        auroc=float(roc_auc_score(labels, scores)) if both_classes else None,
        auprc=float(average_precision_score(labels, scores)) if both_classes else None,
        ppvn=float(curve[-1]) if len(curve) else None,
        ppvn_n=positives,
        mean_ppvn=float(curve.mean()) if len(curve) else None,
        ppvn_tie_policy="expected_uniform",
        accuracy=float(accuracy_score(labels, calls)),
        precision=float(precision_score(labels, calls, zero_division=0)),
        recall=float(recall_score(labels, calls, zero_division=0)),
        f1=float(f1_score(labels, calls, zero_division=0)),
        mcc=float(matthews_corrcoef(labels, calls)),
        brier=float(brier_score_loss(labels, scores)),
        threshold=0.5,
        prevalence=float(labels.mean()),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        cohorts = json.load(handle)
    result = {}
    for cohort, rows in cohorts.items():
        if any(row["in_train"] not in (0, 1) for row in rows):
            raise ValueError(f"Invalid training membership flag in {cohort}")
        result[cohort] = {
            "all": evaluate(rows),
            "overlap_train": evaluate([row for row in rows if row["in_train"]]),
            "nonoverlap_train": evaluate([row for row in rows if not row["in_train"]]),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
