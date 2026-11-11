"""Statistics for the strict holdout and fivefold specificity evaluations."""

import numpy as np

from .training.metrics import compute_binary_metrics, mean_median_of_finite, per_group_aupr


def ppvn(labels, scores):
    """Precision at the number of positives, with fractional credit for ties."""
    n = int(labels.sum())
    if not n:
        return None
    cutoff = np.sort(scores)[-n]
    above, equal = scores > cutoff, scores == cutoff
    return float((labels[above].sum() + (n - above.sum()) * labels[equal].mean()) / n)


def expected_calibration_error(labels, scores):
    bins = np.minimum((scores * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(scores[mask].mean() - labels[mask].mean())
    return float(ece)


def evaluate_predictions(frame):
    labels = frame.y_true.to_numpy(int)
    scores = frame.y_prob.to_numpy(float)
    metrics = compute_binary_metrics(labels, scores)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    predicted_positive = scores >= 0.5
    specificity = float((~predicted_positive[labels == 0]).mean()) if negatives else None
    metrics.update(
        n=len(labels),
        positives=positives,
        negatives=negatives,
        specificity=specificity,
        balanced_accuracy=(metrics["recall"] + specificity) / 2 if positives and negatives else None,
        ppvn=ppvn(labels, scores),
        brier=float(np.square(scores - labels).mean()),
        ece_10=expected_calibration_error(labels, scores),
    )

    for name, groups in (("epitope", "Peptide"), ("phla", ["Peptide", "HLA"])):
        values = per_group_aupr(frame, groups, "y_true", "y_prob", min_pos=3)
        mean, median = mean_median_of_finite(values)
        metrics[f"mean_per_{name}_auprc"] = mean
        metrics[f"median_per_{name}_auprc"] = median
    return {
        key: value if value is None or np.isfinite(value) else None
        for key, value in metrics.items()
    }


def summarize_folds(folds):
    rows = list(folds)
    metrics = (
        "auprc", "auroc", "balanced_accuracy", "precision", "recall",
        "specificity", "ppvn", "acc", "f1", "mcc", "brier", "ece_10",
    )
    summary = {}
    for metric in metrics:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n_folds": len(values),
        }
    return summary
