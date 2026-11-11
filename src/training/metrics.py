"""Binary + per-group classification metrics (step 14).

Public API
----------
- :func:`compute_binary_metrics(y_true, y_prob, threshold=0.5)` →
    ``{"auroc", "auprc", "acc", "precision", "recall", "f1", "mcc", "threshold"}``
- :func:`per_group_aupr(df, group_col, y_true_col, y_prob_col, min_pos=3)` →
  ``pd.Series`` (groups with ``#pos < min_pos`` or a single-class split are
  recorded as ``NaN``, never dropped — the caller sees the full group set).
- :func:`mean_median_of_finite(series)` → ``(mean, median)`` over finite values
  (``(NaN, NaN)`` if no finite entry).

``group_col`` accepts either a single column name (e.g. ``"Peptide"``) or a
list of names (e.g. ``["Peptide", "HLA"]``) for the ``Peptide||HLA`` slice.

Dependency discipline
~~~~~~~~~~~~~~~~~~~~~
This module **does not import** ``tpmhc.training.*``. The AUPR logic mirrors
``tpmhc.training.group_aupr`` in intent (low-positive groups → NaN, mean/median
over finite only) but is reimplemented from scratch to preserve independence
(see DATA_CONTRACT §9 / project rule "no reverse imports from tpmhc").
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "compute_binary_metrics",
    "per_group_aupr",
    "mean_median_of_finite",
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _to_numpy(x: Any, *, dtype: np.dtype, name: str) -> np.ndarray:
    """Best-effort conversion to a 1-D ``np.ndarray``.

    Accepts:
        - ``np.ndarray``
        - ``torch.Tensor`` (detached, moved to CPU)
        - ``pd.Series`` / ``list`` / ``tuple``
    """
    if isinstance(x, np.ndarray):
        arr = x
    elif hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr).astype(dtype, copy=False).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


# ---------------------------------------------------------------------------
# compute_binary_metrics.
# ---------------------------------------------------------------------------


def compute_binary_metrics(
    y_true: Any,
    y_prob: Any,
    *,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Return ``{"auroc", "auprc", "acc", "f1", "threshold"}``.

    Parameters
    ----------
    y_true
        0/1 integer labels; accepted in any numpy / torch / list-like form.
    y_prob
        Probabilities in ``[0, 1]``. If you have logits, apply ``sigmoid`` first.
    threshold
        Decision threshold for ``acc`` / ``f1``. Default 0.5 (the project
        convention — DATA_CONTRACT §9).

    Returns
    -------
    Dict[str, float]
        - ``auroc``  — ``NaN`` if only one class is present in ``y_true``.
        - ``auprc``  — ``NaN`` if only one class is present.
        - ``acc``    — always finite (single-class inputs still defined).
    - ``precision`` / ``recall`` — decision metrics at ``threshold``.
    - ``f1``     — ``NaN`` if precision == recall == 0 (degenerate).
    - ``mcc``    — Matthews correlation coefficient; ``NaN`` if denominator is 0.
        - ``threshold`` — echoes back the input for logging convenience.

    Notes
    -----
    When ``sklearn`` is unavailable at runtime, ``auroc`` / ``auprc`` fall back
    to ``NaN`` with a single warning. This mirrors the behaviour of
    ``compute_binary_metrics`` in the reference repo but is reimplemented
    locally per the no-reverse-import rule.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must lie in [0, 1]; got {threshold}")

    y_t = _to_numpy(y_true, dtype=np.int64, name="y_true")
    y_p = _to_numpy(y_prob, dtype=np.float64, name="y_prob")

    if y_t.shape != y_p.shape:
        raise ValueError(
            f"y_true / y_prob shape mismatch: {y_t.shape} vs {y_p.shape}"
        )
    unique_labels = np.unique(y_t)
    illegal = unique_labels[(unique_labels != 0) & (unique_labels != 1)]
    if illegal.size:
        raise ValueError(
            f"y_true must contain only {{0, 1}}; found extraneous labels {illegal.tolist()}"
        )

    preds = (y_p >= float(threshold)).astype(np.int64)
    tp = int(((preds == 1) & (y_t == 1)).sum())
    tn = int(((preds == 0) & (y_t == 0)).sum())
    fp = int(((preds == 1) & (y_t == 0)).sum())
    fn = int(((preds == 0) & (y_t == 1)).sum())

    n = len(y_t)
    acc = (tp + tn) / max(n, 1)

    denom_pr = (tp + fp) if (tp + fp) > 0 else 0
    denom_re = (tp + fn) if (tp + fn) > 0 else 0
    prec = (tp / denom_pr) if denom_pr > 0 else 0.0
    rec = (tp / denom_re) if denom_re > 0 else 0.0
    if prec + rec <= 0.0:
        # Degenerate (no positives predicted or no positive targets).
        # Use NaN rather than 0 to signal it cleanly in logs.
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)

    mcc_denom = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = ((tp * tn - fp * fn) / mcc_denom) if mcc_denom > 0.0 else float("nan")

    auroc = float("nan")
    auprc = float("nan")
    if unique_labels.size >= 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            auroc = float(roc_auc_score(y_t, y_p))
            auprc = float(average_precision_score(y_t, y_p))
        except Exception as exc:  # pragma: no cover - sklearn missing
            logger.warning(
                "[compute_binary_metrics] sklearn metric computation failed: %s", exc
            )

    return {
        "auroc": auroc,
        "auprc": auprc,
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
        "threshold": float(threshold),
    }


# ---------------------------------------------------------------------------
# per_group_aupr.
# ---------------------------------------------------------------------------


def _resolve_group_keys(
    group_col: Union[str, Sequence[str]],
) -> List[str]:
    if isinstance(group_col, str):
        return [group_col]
    if isinstance(group_col, Sequence) and all(isinstance(c, str) for c in group_col):
        keys = list(group_col)
        if not keys:
            raise ValueError("group_col list must be non-empty")
        return keys
    raise TypeError(
        f"group_col must be str or sequence of str; got {type(group_col).__name__}"
    )


def per_group_aupr(
    df: pd.DataFrame,
    group_col: Union[str, Sequence[str]],
    y_true: str,
    y_prob: str,
    *,
    min_pos: int = 3,
) -> pd.Series:
    """Per-group AUPR, with low-positive groups marked ``NaN``.

    Parameters
    ----------
    df
        DataFrame containing at least ``group_col`` + ``y_true`` + ``y_prob``.
    group_col
        ``"Peptide"`` or ``["Peptide", "HLA"]`` — the latter produces a
        ``MultiIndex`` series keyed by ``(peptide, hla)``.
    y_true
        Column name of 0/1 integer labels.
    y_prob
        Column name of float probabilities in ``[0, 1]``.
    min_pos
        Groups with fewer than ``min_pos`` positive examples **or** only one
        class are recorded as ``NaN``. They are never dropped from the Series
        — downstream calls to :func:`mean_median_of_finite` will skip them.

    Returns
    -------
    pd.Series
        Named ``aupr@{group_col}`` (``||``-joined for a list). Index name is
        the group column(s). Values: float AUPR or ``NaN``.

    Warns
    -----
    Logs a single warning with the count / total of groups that were dropped
    into NaN so the operator can spot evaluation coverage issues.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df must be a pandas DataFrame; got {type(df).__name__}"
        )
    if min_pos < 1:
        raise ValueError(f"min_pos must be >= 1; got {min_pos}")

    group_keys = _resolve_group_keys(group_col)
    missing = [c for c in group_keys + [y_true, y_prob] if c not in df.columns]
    if missing:
        raise KeyError(
            f"per_group_aupr: df is missing columns {missing}"
        )

    # Lazy-import sklearn so that importing this module never triggers it.
    try:
        from sklearn.metrics import average_precision_score
    except Exception as exc:  # pragma: no cover - sklearn missing
        raise ImportError(
            "per_group_aupr requires scikit-learn; "
            "install via requirements.txt"
        ) from exc

    # Stable alphabetical group order to keep regression tests deterministic.
    groupby_target = group_keys[0] if len(group_keys) == 1 else group_keys
    grouped = df.groupby(groupby_target, dropna=False, sort=True)

    keys: List[Any] = []
    values: List[float] = []
    dropped_low_pos = 0
    dropped_one_class = 0

    for key, sub in grouped:
        y_t = np.asarray(sub[y_true]).astype(np.int64).reshape(-1)
        y_p = np.asarray(sub[y_prob]).astype(np.float64).reshape(-1)
        n_pos = int((y_t == 1).sum())
        n_neg = int((y_t == 0).sum())

        if n_pos < min_pos:
            keys.append(key)
            values.append(float("nan"))
            dropped_low_pos += 1
            continue
        if n_neg == 0:
            # All positives: AUPR is undefined (degenerate).
            keys.append(key)
            values.append(float("nan"))
            dropped_one_class += 1
            continue

        keys.append(key)
        values.append(float(average_precision_score(y_t, y_p)))

    total = len(keys)
    if dropped_low_pos:
        logger.warning(
            "[per_group_aupr] %d/%d groups had #pos < min_pos=%d → NaN",
            dropped_low_pos,
            total,
            min_pos,
        )
    if dropped_one_class:
        logger.warning(
            "[per_group_aupr] %d/%d groups had only one class → NaN",
            dropped_one_class,
            total,
        )

    suffix = "||".join(group_keys)
    series_name = f"aupr@{suffix}"

    if len(group_keys) == 1:
        idx = pd.Index(keys, name=group_keys[0])
    else:
        # ``groupby`` yields tuples when grouping by multiple columns.
        idx = pd.MultiIndex.from_tuples(keys, names=group_keys)

    return pd.Series(values, index=idx, name=series_name, dtype=float)


# ---------------------------------------------------------------------------
# mean_median_of_finite.
# ---------------------------------------------------------------------------


def mean_median_of_finite(series: Any) -> Tuple[float, float]:
    """Return ``(mean, median)`` over finite values.

    Parameters
    ----------
    series
        Anything convertible via ``np.asarray(...)`` — ``pd.Series``, list,
        ndarray, etc.

    Returns
    -------
    Tuple[float, float]
        ``(mean, median)`` over the values passing ``np.isfinite``. If no
        finite values exist, returns ``(NaN, NaN)`` and emits a warning.
    """
    if isinstance(series, pd.Series):
        arr = series.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(series, dtype=np.float64)
    arr = arr.reshape(-1)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        logger.warning(
            "[mean_median_of_finite] all %d entries are non-finite; returning (NaN, NaN)",
            arr.size,
        )
        return float("nan"), float("nan")

    return float(np.mean(finite)), float(np.median(finite))
