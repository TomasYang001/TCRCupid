"""Normalize dumped specificity scores within biologically meaningful groups.

The script consumes the CSV emitted by :mod:`predict_specificity` and writes
ranking metrics after group-wise logit normalization. It is intentionally
post-hoc: no labels from the evaluated split are used unless the caller passes
that split as ``--fit-predictions`` for alpha selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..training.metrics import (
    compute_binary_metrics,
    mean_median_of_finite,
    per_group_aupr,
)

logger = logging.getLogger(__name__)

GROUP_PRESETS = {
    "HLA": ["HLA"],
    "Peptide": ["Peptide"],
    "pHLA": ["Peptide", "HLA"],
}


def _logit_prob(values: pd.Series) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return pd.Series(np.log(arr / (1.0 - arr)), index=values.index)


def _sigmoid(values: pd.Series) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    out = 1.0 / (1.0 + np.exp(-arr))
    return pd.Series(out, index=values.index)


def _group_columns(group_by: str) -> List[str]:
    if group_by not in GROUP_PRESETS:
        raise ValueError(f"unknown group_by={group_by!r}; choices={sorted(GROUP_PRESETS)}")
    return GROUP_PRESETS[group_by]


def normalize_scores(
    df: pd.DataFrame,
    *,
    mode: str,
    group_by: str,
    alpha: float = 1.0,
    input_col: str = "y_prob",
) -> pd.Series:
    """Return a monotonic score for downstream AUPRC/AUROC computation."""
    if input_col not in df.columns:
        raise KeyError(f"missing score column {input_col!r}")
    group_cols = _group_columns(group_by)
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise KeyError(f"missing group columns {missing}")

    logits = _logit_prob(df[input_col])
    grouped = logits.groupby([df[col] for col in group_cols], dropna=False)

    if mode == "raw":
        return logits
    if mode == "group_center":
        means = grouped.transform("mean")
        return logits - float(alpha) * means
    if mode == "group_zscore":
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0.0, np.nan).fillna(1.0)
        return (logits - means) / stds
    if mode == "group_rank":
        return grouped.rank(method="average", pct=True)

    raise ValueError(
        "mode must be one of raw, group_center, group_zscore, group_rank; "
        f"got {mode!r}"
    )


def _same_hla_mask(df: pd.DataFrame) -> pd.Series:
    mask = df["y_true"].astype(int) == 1
    if "negative_type" in df.columns:
        mask = mask | (df["negative_type"].fillna("") == "same_hla_reshuffle")
    if "hla_match_level" in df.columns:
        mask = mask | (df["hla_match_level"].fillna("") == "same_hla")
    return mask


def score_metrics(df: pd.DataFrame, score: pd.Series) -> Dict[str, float]:
    work = df.copy()
    work["normalized_score"] = np.asarray(score, dtype=np.float64)
    work["normalized_prob"] = _sigmoid(work["normalized_score"])

    metrics = compute_binary_metrics(work["y_true"], work["normalized_prob"])
    pep = per_group_aupr(work, "Peptide", "y_true", "normalized_prob")
    phla = per_group_aupr(work, ["Peptide", "HLA"], "y_true", "normalized_prob")
    pep_mean, pep_median = mean_median_of_finite(pep)
    phla_mean, phla_median = mean_median_of_finite(phla)
    metrics.update({
        "mean_per_epitope_auprc": pep_mean,
        "median_per_epitope_auprc": pep_median,
        "mean_per_phla_auprc": phla_mean,
        "median_per_phla_auprc": phla_median,
    })

    same_mask = _same_hla_mask(work)
    same = work.loc[same_mask]
    metrics.update({
        "same_hla_n": float(len(same)),
        "same_hla_pos": float((same["y_true"].astype(int) == 1).sum()),
        "same_hla_neg": float((same["y_true"].astype(int) == 0).sum()),
    })
    if len(same) > 0 and same["y_true"].nunique() >= 2:
        same_metrics = compute_binary_metrics(same["y_true"], same["normalized_prob"])
        metrics["same_hla_auprc"] = same_metrics["auprc"]
        metrics["same_hla_auroc"] = same_metrics["auroc"]
    else:
        metrics["same_hla_auprc"] = float("nan")
        metrics["same_hla_auroc"] = float("nan")
    return metrics


def _alpha_grid(min_value: float, max_value: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError(f"alpha step must be > 0; got {step}")
    n = int(math.floor((max_value - min_value) / step)) + 1
    return min_value + np.arange(n, dtype=np.float64) * step


def select_alpha(
    fit_df: pd.DataFrame,
    *,
    group_by: str,
    input_col: str,
    alpha_values: Sequence[float],
) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score

    best_alpha = None
    best_auprc = -float("inf")
    for alpha in alpha_values:
        score = normalize_scores(
            fit_df,
            mode="group_center",
            group_by=group_by,
            alpha=float(alpha),
            input_col=input_col,
        )
        auprc = float(average_precision_score(fit_df["y_true"].astype(int), _sigmoid(score)))
        if auprc > best_auprc:
            best_auprc = auprc
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("alpha selection failed")
    return {"alpha": best_alpha, "fit_auprc": float(best_auprc)}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:  # pragma: no cover - CLI
    p = argparse.ArgumentParser(prog="normalize_specificity_scores")
    p.add_argument("--predictions", required=True)
    p.add_argument("--mode", choices=["raw", "group_center", "group_zscore", "group_rank"], required=True)
    p.add_argument("--group-by", choices=sorted(GROUP_PRESETS), default="HLA")
    p.add_argument("--input-col", default="y_prob")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--fit-predictions", default=None)
    p.add_argument("--alpha-min", type=float, default=0.0)
    p.add_argument("--alpha-max", type=float, default=2.0)
    p.add_argument("--alpha-step", type=float, default=0.01)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-predictions", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    df = pd.read_csv(args.predictions)
    alpha = float(args.alpha)
    fit_summary = None
    if args.fit_predictions:
        if args.mode != "group_center":
            raise ValueError("--fit-predictions is only supported for mode=group_center")
        fit_df = pd.read_csv(args.fit_predictions)
        fit_summary = select_alpha(
            fit_df,
            group_by=args.group_by,
            input_col=args.input_col,
            alpha_values=_alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step),
        )
        alpha = float(fit_summary["alpha"])

    score = normalize_scores(
        df,
        mode=args.mode,
        group_by=args.group_by,
        alpha=alpha,
        input_col=args.input_col,
    )
    metrics = score_metrics(df, score)
    result: Dict[str, Any] = {
        "status": "ok",
        "predictions": args.predictions,
        "mode": args.mode,
        "group_by": args.group_by,
        "alpha": alpha,
        "fit": fit_summary,
        "metrics": metrics,
        "n_samples": int(len(df)),
    }

    if args.out_predictions:
        out_df = df.copy()
        out_df["normalized_score"] = np.asarray(score, dtype=np.float64)
        out_df["normalized_prob"] = np.asarray(_sigmoid(score), dtype=np.float64)
        out_path = Path(args.out_predictions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        logger.info("[normalize_specificity_scores] wrote %s", out_path)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info("[normalize_specificity_scores] wrote %s", out_json)

    logger.info("[normalize_specificity_scores] metrics: %s", result["metrics"])
    return result


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
