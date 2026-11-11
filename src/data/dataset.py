"""``TCRPMHCDataset`` + ``_normalize_columns`` + Held-out views.

Implements DATA_CONTRACT v0.1 §2–§5. See ``../DATA_CONTRACT.md`` for the rules
this module enforces.

Principles (must not be violated by subclasses / callers):

1. **Tokenization does not happen here.** ``__getitem__`` returns plain Python
   ``str``/``float``/``dict`` values; the heavy lifting (ESM-C / CDR tokenisers)
   is deferred to the collate stage.
2. **Column names follow the real CSV convention** (``Va / Vb / Ja / Jb / label``).
   ``_normalize_columns`` is the *only* allowed place to touch column names;
   any future legacy aliases should go through ``LEGACY_RENAME_MAP`` rather
   than being handled ad-hoc in calling code.
3. **Class II is dropped** in v0.1 (see DATA_CONTRACT §1).
4. Every drop is accounted for in ``drop_report`` — this becomes
   ``data_drop_report.json`` at training time.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..config import LengthConfig
from .mhc_pseudo import PseudoInfo, classify_hla, fill_pseudo

logger = logging.getLogger(__name__)

DIAGNOSTIC_META_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "negative_type",
    "hla_match_level",
    "peptide_levenshtein_distance",
    "cdr3a_length_diff",
    "cdr3b_length_diff",
    "receptor_id",
    "target_peptide",
    "donor_peptide",
    "target_hla",
    "donor_hla",
    "scan_id",
    "wt_peptide",
    "mut_peptide",
    "is_wt",
    "n_substitutions",
    "mutation_position",
    "wt_aa",
    "mut_aa",
    "mut_aa_type",
    "mutation_id",
    "activation_label",
)


__all__ = [
    "STANDARD_COLUMNS",
    "DIAGNOSTIC_META_COLUMNS",
    "LEGACY_RENAME_MAP",
    "DropReport",
    "TCRPMHCDataset",
    "PMHCBindingDataset",
    "EpitopeHeldOutView",
    "PhlaHeldOutView",
]


# ---------------------------------------------------------------------------
# Column schema (DATA_CONTRACT §2.1).
# ---------------------------------------------------------------------------

STANDARD_COLUMNS: Tuple[str, ...] = (
    "Peptide",
    "HLA",
    "MHC_pseudo_seq",
    "CDR3b",
    "Vb",
    "Jb",
    "CDR1b",
    "CDR2b",
    "CDR3a",
    "Va",
    "Ja",
    "CDR1a",
    "CDR2a",
    "label",
)

# Future-proofing hook (DATA_CONTRACT §2.2): populated empty by default.
LEGACY_RENAME_MAP: Dict[str, str] = {
    # tcrlm / PairedTCR ``preprocessed_unlabelled`` exports (TRAV, CDR3B, …).
    "TRAV": "Va",
    "TRBV": "Vb",
    "TRAJ": "Ja",
    "TRBJ": "Jb",
    "CDR3A": "CDR3a",
    "CDR3B": "CDR3b",
    "Epitope": "Peptide",
    "MHCA": "HLA",
}

# Per-chain CDR alias expansion (not strictly needed for the four-split CSVs
# but useful for future sources that ship short column names).
_OPTIONAL_ALIASES: Dict[str, str] = {
    "peptide": "Peptide",
    "hla": "HLA",
    "mhc_pseudo": "MHC_pseudo_seq",
    "MHC_PSEUDO_SEQ": "MHC_pseudo_seq",
}

_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_STANDARD_AA_SET = frozenset(_STANDARD_AA)
_NON_AA_RE = re.compile(rf"[^{_STANDARD_AA}]")


# ---------------------------------------------------------------------------
# Drop report container.
# ---------------------------------------------------------------------------


def _report_json_value(value: Any) -> Any:
    """Keep numeric report fields numeric while making pandas/NumPy JSON-safe."""
    if isinstance(value, dict):
        return {key: _report_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_report_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class DropReport:
    """Accounting structure for rows removed during normalization / validation."""

    reasons: Counter = field(default_factory=Counter)
    examples: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    total_input: int = 0
    total_kept: int = 0

    def record(self, reason: str, row: Mapping[str, Any]) -> None:
        self.reasons[reason] += 1
        if len(self.examples[reason]) < 5:
            self.examples[reason].append(dict(row))

    def to_dict(self) -> Dict[str, Any]:
        dropped_total = sum(self.reasons.values())
        return _report_json_value({
            "total_input": self.total_input,
            "total_kept": self.total_kept,
            "total_dropped": dropped_total,
            "drop_rate": (dropped_total / self.total_input) if self.total_input else 0.0,
            "reasons": dict(self.reasons),
            "examples": {k: list(v) for k, v in self.examples.items()},
        })


# ---------------------------------------------------------------------------
# Normalisation helpers.
# ---------------------------------------------------------------------------


def _strip_upper(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", "null"):
        return ""
    return s


def _coerce_label(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return None
            if float(value) not in (0.0, 1.0):
                return None
            return int(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "pos", "positive"):
            return 1
        if s in ("0", "false", "f", "no", "n", "neg", "negative"):
            return 0
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _check_aa(seq: str) -> bool:
    """True iff seq is a non-empty string over the 20 standard AAs."""
    if not seq:
        return False
    return _NON_AA_RE.search(seq.upper()) is None


def _check_optional_aa(seq: str) -> bool:
    """Empty string is OK; otherwise must be AA-only."""
    if not seq:
        return True
    return _NON_AA_RE.search(seq.upper()) is None


def _normalize_columns(
    df: pd.DataFrame,
    report: Optional[DropReport] = None,
    *,
    passthrough_columns: Tuple[str, ...] = (),
) -> pd.DataFrame:
    """Rename legacy aliases, enforce standard columns, and log dropped extras.

    **Does not** perform row-level validation — that happens in
    :func:`_validate_and_fill`. This split keeps the column-level transform
    idempotent and easy to reason about.
    """
    if report is None:
        report = DropReport()

    df = df.copy()

    # 1. Apply legacy/optional renames.
    applied: Dict[str, str] = {}
    for src, dst in list(LEGACY_RENAME_MAP.items()) + list(_OPTIONAL_ALIASES.items()):
        if src in df.columns and dst not in df.columns:
            applied[src] = dst
    if applied:
        df = df.rename(columns=applied)
        logger.info("[dataset] applied legacy renames: %s", applied)

    # 2. Drop the "Task" column outright (DATA_CONTRACT: never keep Task).
    if "Task" in df.columns:
        logger.info("[dataset] dropping 'Task' column")
        df = df.drop(columns=["Task"])

    # 3. Track and drop non-standard columns.
    standard_set = set(STANDARD_COLUMNS)
    passthrough_set = {c for c in passthrough_columns if c in df.columns}
    extras = [c for c in df.columns if c not in standard_set and c not in passthrough_set]
    if extras:
        logger.info("[dataset] discarding %d non-standard columns: %s", len(extras), extras)
        df = df.drop(columns=extras)

    # 4. Ensure every standard column exists (fill missing with defaults).
    #    MHC_pseudo_seq is special — it may be absent; it's filled in
    #    _validate_and_fill after HLA is validated.
    defaults: Dict[str, Any] = {
        "Peptide": "",
        "HLA": "",
        "MHC_pseudo_seq": "",
        "CDR3b": "",
        "Vb": "unknown",
        "Jb": "unknown",
        "CDR1b": "",
        "CDR2b": "",
        "CDR3a": "",
        "Va": "unknown",
        "Ja": "unknown",
        "CDR1a": "",
        "CDR2a": "",
        "label": None,
    }
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = defaults[col]

    # 5. Reorder to STANDARD_COLUMNS for determinism.
    return df[list(STANDARD_COLUMNS) + [c for c in passthrough_columns if c in df.columns]]


def _validate_and_fill(
    df: pd.DataFrame,
    length_cfg: LengthConfig,
    report: DropReport,
) -> pd.DataFrame:
    """Apply DATA_CONTRACT §2.3 + §3 row-level validation, filling MHC pseudo.

    Returns a **new** DataFrame containing only the rows that passed.
    """
    kept_idx: List[int] = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        peptide = _strip_upper(row["Peptide"]).upper()
        hla = _strip_upper(row["HLA"])
        cdr3b = _strip_upper(row["CDR3b"]).upper()
        mhc_pseudo = _strip_upper(row["MHC_pseudo_seq"])

        cdr1a = _strip_upper(row["CDR1a"]).upper()
        cdr2a = _strip_upper(row["CDR2a"]).upper()
        cdr3a = _strip_upper(row["CDR3a"]).upper()
        cdr1b = _strip_upper(row["CDR1b"]).upper()
        cdr2b = _strip_upper(row["CDR2b"]).upper()

        label = _coerce_label(row["label"])

        raw_meta = {"Peptide": peptide, "HLA": hla, "CDR3b": cdr3b, "label": row["label"]}

        if not peptide or not _check_aa(peptide):
            report.record("peptide_invalid_aa", raw_meta)
            continue
        if len(peptide) > length_cfg.peptide:
            report.record("peptide_too_long", raw_meta)
            continue

        if not hla:
            report.record("hla_missing", raw_meta)
            continue

        hla_class = classify_hla(hla)
        if hla_class == "classII":
            report.record("hla_class_ii_dropped", raw_meta)
            continue
        if hla_class == "unknown":
            report.record("hla_unrecognized", raw_meta)
            continue

        # Fill MHC_pseudo_seq when missing.
        if not mhc_pseudo:
            info: Optional[PseudoInfo] = fill_pseudo(hla)
            if info is None or info.dtype != "classI" or info.sequence is None:
                report.record("hla_not_mapped", raw_meta)
                continue
            mhc_pseudo = info.sequence
        if not mhc_pseudo or len(mhc_pseudo) > length_cfg.mhc_pseudo:
            report.record("mhc_pseudo_bad_length", raw_meta)
            continue
        if not _check_aa(mhc_pseudo):
            report.record("mhc_pseudo_bad_aa", raw_meta)
            continue

        if not cdr3b or not _check_aa(cdr3b):
            report.record("cdr3b_invalid_aa", raw_meta)
            continue
        if len(cdr3b) > length_cfg.cdr:
            report.record("cdr3b_too_long", raw_meta)
            continue

        # Optional CDR fields: silently blank out non-AA contamination (warn).
        def _sanitize_opt(name: str, value: str) -> str:
            if not value:
                return value
            if _check_aa(value):
                if len(value) > length_cfg.cdr:
                    return "__TOO_LONG__"
                return value
            logger.warning("[dataset] row %d: field %s has non-AA chars → blanked", idx, name)
            return ""

        cdr1a_clean = _sanitize_opt("CDR1a", cdr1a)
        cdr2a_clean = _sanitize_opt("CDR2a", cdr2a)
        cdr3a_clean = _sanitize_opt("CDR3a", cdr3a)
        cdr1b_clean = _sanitize_opt("CDR1b", cdr1b)
        cdr2b_clean = _sanitize_opt("CDR2b", cdr2b)

        if "__TOO_LONG__" in (cdr1a_clean, cdr2a_clean, cdr3a_clean, cdr1b_clean, cdr2b_clean):
            report.record("cdr_too_long", raw_meta)
            continue

        # Per-chain CDR concatenation budget (applies to non-empty triples).
        def _chain_len(*segs: str) -> int:
            return sum(len(s) for s in segs)

        if _chain_len(cdr1b_clean, cdr2b_clean, cdr3b) > length_cfg.chain_cdr_concat:
            report.record("chain_b_over_budget", raw_meta)
            continue
        if _chain_len(cdr1a_clean, cdr2a_clean, cdr3a_clean) > length_cfg.chain_cdr_concat:
            report.record("chain_a_over_budget", raw_meta)
            continue

        if label is None or label not in (0, 1):
            report.record("label_invalid", raw_meta)
            continue

        # Commit normalised values back into the frame.
        df.at[idx, "Peptide"] = peptide
        df.at[idx, "HLA"] = hla
        df.at[idx, "MHC_pseudo_seq"] = mhc_pseudo
        df.at[idx, "CDR3b"] = cdr3b
        df.at[idx, "CDR1b"] = cdr1b_clean
        df.at[idx, "CDR2b"] = cdr2b_clean
        df.at[idx, "CDR3a"] = cdr3a_clean
        df.at[idx, "CDR1a"] = cdr1a_clean
        df.at[idx, "CDR2a"] = cdr2a_clean
        df.at[idx, "label"] = int(label)

        kept_idx.append(idx)

    kept = df.iloc[kept_idx].copy().reset_index(drop=True)
    kept["label"] = kept["label"].astype(int)
    report.total_kept = len(kept)
    return kept


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------


class TCRPMHCDataset(Dataset):
    """CSV-backed dataset producing DATA_CONTRACT §2 dict samples.

    Parameters
    ----------
    csv_path
        Path to a ``combined_*.csv`` (or any CSV with compatible columns).
    length_cfg
        Length bounds to enforce (DATA_CONTRACT §3). Defaults to ``LengthConfig()``.
    label_col
        Expected label column name in the **already-normalized** frame; must be
        ``"label"`` (default) under DATA_CONTRACT v0.1.
    preload
        If True, materialize all ``__getitem__`` dicts at construction time into
        ``self._cache`` for faster iteration (use for small splits).
    report_json
        Optional path; when provided, the ``DropReport.to_dict()`` is written
        there after ``__init__`` completes. Independent of training-time
        ``data_drop_report.json``.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        length_cfg: Optional[LengthConfig] = None,
        label_col: str = "label",
        preload: bool = False,
        *,
        report_json: Optional[Union[str, Path]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.length_cfg = length_cfg or LengthConfig()
        if label_col != "label":
            # Deliberate strictness: DATA_CONTRACT §2.1 forbids aliases.
            raise ValueError(
                f"label_col must be 'label' per DATA_CONTRACT v0.1; got {label_col!r}. "
                "Translate your CSV via _normalize_columns / LEGACY_RENAME_MAP instead."
            )
        self.label_col = label_col
        self.preload = preload

        raw = pd.read_csv(self.csv_path, low_memory=False)
        raw_source_idx_col = "__source_row_idx"
        raw[raw_source_idx_col] = np.arange(len(raw), dtype=np.int64)
        diagnostic_cols = tuple(c for c in DIAGNOSTIC_META_COLUMNS if c in raw.columns)

        self.report = DropReport(total_input=len(raw))
        df = _normalize_columns(
            raw,
            self.report,
            passthrough_columns=(raw_source_idx_col,),
        )
        df = _validate_and_fill(df, self.length_cfg, self.report)

        source_idx = (
            df[raw_source_idx_col].astype(int).to_numpy()
            if raw_source_idx_col in df.columns
            else np.arange(len(df), dtype=np.int64)
        )
        self._diagnostic_meta: List[Dict[str, Any]] = []
        for raw_idx in source_idx:
            meta: Dict[str, Any] = {}
            for col in diagnostic_cols:
                val = raw.iloc[int(raw_idx)][col]
                if pd.isna(val):
                    val = ""
                meta[col] = str(val)
            self._diagnostic_meta.append(meta)

        if raw_source_idx_col in df.columns:
            df = df.drop(columns=[raw_source_idx_col])
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self._cache: Optional[List[Dict[str, Any]]] = None
        if self.preload:
            self._cache = [self._build_sample(i) for i in range(len(self.df))]

        if report_json is not None:
            import json

            path = Path(report_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(
            "[dataset] %s: kept=%d / input=%d (drop_rate=%.3f)",
            self.csv_path.name,
            self.report.total_kept,
            self.report.total_input,
            self.report.to_dict()["drop_rate"],
        )

    # ----- Standard dataset API -----

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self._build_sample(idx)

    # ----- Helpers -----

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "peptide": str(row["Peptide"]),
            "mhc_pseudo": str(row["MHC_pseudo_seq"]),
            "cdr_alpha": {
                "cdr1": str(row["CDR1a"]),
                "cdr2": str(row["CDR2a"]),
                "cdr3": str(row["CDR3a"]),
            },
            "cdr_beta": {
                "cdr1": str(row["CDR1b"]),
                "cdr2": str(row["CDR2b"]),
                "cdr3": str(row["CDR3b"]),
            },
            "label": float(row["label"]),
            # Diagnostic passthrough; collate may inspect this but must not
            # feed it into the model. Includes ``Peptide`` so Stage-3 per-epitope
            # metrics (step 17) can index into the batch without re-reading the CSV.
            "_meta": {
                "Peptide": str(row["Peptide"]),
                "HLA": str(row["HLA"]),
                "Va": str(row["Va"]),
                "Ja": str(row["Ja"]),
                "Vb": str(row["Vb"]),
                "Jb": str(row["Jb"]),
                **(self._diagnostic_meta[idx] if idx < len(self._diagnostic_meta) else {}),
            },
        }

    # ----- Convenience -----

    def compute_pos_weight(self) -> float:
        """Return #neg / max(#pos, 1) for BCEWithLogitsLoss."""
        labels = self.df[self.label_col].astype(int).to_numpy()
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        return float(neg) / max(pos, 1)

    def unique_epitopes(self) -> Set[str]:
        return set(self.df["Peptide"].astype(str).unique())

    def unique_phlas(self) -> Set[Tuple[str, str]]:
        return set(zip(self.df["Peptide"].astype(str), self.df["HLA"].astype(str)))


# ---------------------------------------------------------------------------
# Stage-1 pMHC binding dataset (no TCR columns required).
# ---------------------------------------------------------------------------


def _normalize_pmhc_columns(
    df: pd.DataFrame,
    report: DropReport,
) -> pd.DataFrame:
    """Rename aliases + drop extras, but only enforce the pMHC subset.

    Unlike :func:`_normalize_columns`, this keeps only
    ``{Peptide, HLA, MHC_pseudo_seq, label}`` — pMHC binding data has no TCR
    fields and we refuse to silently fabricate them.
    """
    df = df.copy()

    applied: Dict[str, str] = {}
    for src, dst in list(LEGACY_RENAME_MAP.items()) + list(_OPTIONAL_ALIASES.items()):
        if src in df.columns and dst not in df.columns:
            applied[src] = dst
    if applied:
        df = df.rename(columns=applied)
        logger.info("[pmhc_binding] applied legacy renames: %s", applied)

    if "Task" in df.columns:
        logger.info("[pmhc_binding] dropping 'Task' column")
        df = df.drop(columns=["Task"])

    required_subset = ("Peptide", "HLA", "MHC_pseudo_seq", "label")
    extras = [c for c in df.columns if c not in required_subset]
    if extras:
        logger.info(
            "[pmhc_binding] discarding %d non-pMHC columns: %s",
            len(extras),
            extras,
        )
        df = df.drop(columns=extras)

    defaults: Dict[str, Any] = {
        "Peptide": "",
        "HLA": "",
        "MHC_pseudo_seq": "",
        "label": None,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    return df[list(required_subset)]


def _validate_pmhc_rows(
    df: pd.DataFrame,
    length_cfg: LengthConfig,
    report: DropReport,
) -> pd.DataFrame:
    """Row-level validation for pMHC binding data (peptide + HLA + pseudo + label)."""
    kept_idx: List[int] = []

    for idx in range(len(df)):
        row = df.iloc[idx]

        peptide = _strip_upper(row["Peptide"]).upper()
        hla = _strip_upper(row["HLA"])
        mhc_pseudo = _strip_upper(row["MHC_pseudo_seq"])
        label = _coerce_label(row["label"])

        raw_meta = {"Peptide": peptide, "HLA": hla, "label": row["label"]}

        if not peptide or not _check_aa(peptide):
            report.record("peptide_invalid_aa", raw_meta)
            continue
        if len(peptide) > length_cfg.peptide:
            report.record("peptide_too_long", raw_meta)
            continue

        if not hla:
            report.record("hla_missing", raw_meta)
            continue

        hla_class = classify_hla(hla)
        if hla_class == "classII":
            report.record("hla_class_ii_dropped", raw_meta)
            continue
        if hla_class == "unknown":
            report.record("hla_unrecognized", raw_meta)
            continue

        if not mhc_pseudo:
            info: Optional[PseudoInfo] = fill_pseudo(hla)
            if info is None or info.dtype != "classI" or info.sequence is None:
                report.record("hla_not_mapped", raw_meta)
                continue
            mhc_pseudo = info.sequence
        if not mhc_pseudo or len(mhc_pseudo) > length_cfg.mhc_pseudo:
            report.record("mhc_pseudo_bad_length", raw_meta)
            continue
        if not _check_aa(mhc_pseudo):
            report.record("mhc_pseudo_bad_aa", raw_meta)
            continue

        if label is None or label not in (0, 1):
            report.record("label_invalid", raw_meta)
            continue

        df.at[idx, "Peptide"] = peptide
        df.at[idx, "HLA"] = hla
        df.at[idx, "MHC_pseudo_seq"] = mhc_pseudo
        df.at[idx, "label"] = int(label)

        kept_idx.append(idx)

    kept = df.iloc[kept_idx].copy().reset_index(drop=True)
    kept["label"] = kept["label"].astype(int)
    report.total_kept = len(kept)
    return kept


class PMHCBindingDataset(Dataset):
    """CSV-backed dataset for Stage-1 pMHC binding pretraining.

    CSV must contain at least ``Peptide`` + ``HLA`` + ``label`` columns;
    ``MHC_pseudo_seq`` is optional and filled from
    :func:`src.data.mhc_pseudo.fill_pseudo` when missing.
    TCR columns (``CDR1a`` / ... / ``Vb`` / etc.) are **ignored** — this
    dataset intentionally refuses to fabricate TCR data so Stage-1 never
    contaminates Stage-3 with synthetic TCR tokens.

    Each sample is:

    .. code-block:: python

        {
            "peptide":      str,
            "mhc_pseudo":   str,
            "label":        float,
            "_meta":        {"HLA": str},
        }

    Parameters
    ----------
    csv_path
        Path to a Stage-1 CSV. Extra columns are silently dropped.
    length_cfg
        Length budget (``peptide`` / ``mhc_pseudo``). Defaults to
        :class:`LengthConfig`.
    label_col
        Must be ``"label"`` under DATA_CONTRACT v0.1.
    preload
        Cache all samples in memory.
    report_json
        Optional ``DropReport`` dump path.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        length_cfg: Optional[LengthConfig] = None,
        label_col: str = "label",
        preload: bool = False,
        *,
        report_json: Optional[Union[str, Path]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.length_cfg = length_cfg or LengthConfig()
        if label_col != "label":
            raise ValueError(
                f"label_col must be 'label' per DATA_CONTRACT v0.1; got {label_col!r}"
            )
        self.label_col = label_col
        self.preload = preload

        raw = pd.read_csv(self.csv_path, low_memory=False)
        self.report = DropReport(total_input=len(raw))
        df = _normalize_pmhc_columns(raw, self.report)
        df = _validate_pmhc_rows(df, self.length_cfg, self.report)

        self.df: pd.DataFrame = df.reset_index(drop=True)
        self._cache: Optional[List[Dict[str, Any]]] = None
        if self.preload:
            self._cache = [self._build_sample(i) for i in range(len(self.df))]

        if report_json is not None:
            import json

            path = Path(report_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(
            "[pmhc_binding] %s: kept=%d / input=%d (drop_rate=%.3f)",
            self.csv_path.name,
            self.report.total_kept,
            self.report.total_input,
            self.report.to_dict()["drop_rate"],
        )

    # ----- Dataset API -----

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._cache is not None:
            return self._cache[idx]
        return self._build_sample(idx)

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "peptide": str(row["Peptide"]),
            "mhc_pseudo": str(row["MHC_pseudo_seq"]),
            "label": float(row["label"]),
            "_meta": {"HLA": str(row["HLA"])},
        }

    # ----- Convenience -----

    def compute_pos_weight(self) -> float:
        """Return ``#neg / max(#pos, 1)`` (matches :meth:`TCRPMHCDataset`)."""
        labels = self.df[self.label_col].astype(int).to_numpy()
        pos = int((labels == 1).sum())
        neg = int((labels == 0).sum())
        return float(neg) / max(pos, 1)


# ---------------------------------------------------------------------------
# Held-out views (for use when an in-memory split is more convenient than
# materialising separate CSVs — e.g. sanity-checking, sub-sampling, or future
# cross-validation tooling; the canonical four CSVs are still loaded directly).
# ---------------------------------------------------------------------------


class _IndexView(Dataset):
    """Thin view that re-indexes into a base TCRPMHCDataset."""

    def __init__(self, base: TCRPMHCDataset, indices: List[int], tag: str) -> None:
        self.base = base
        self.indices = list(indices)
        self.tag = tag

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.base[int(self.indices[i])]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} tag={self.tag!r} n={len(self)}>"


class EpitopeHeldOutView(_IndexView):
    """Filter a ``TCRPMHCDataset`` by a set of held-out epitopes.

    Parameters
    ----------
    base
        Parent dataset.
    held_out_epitopes
        Peptides to treat as the held-out set.
    mode
        ``"exclude"`` (default) — keep rows whose ``Peptide`` is **not** in
        ``held_out_epitopes`` (i.e. train split);
        ``"include"`` — keep only rows in ``held_out_epitopes`` (i.e. test split).
    """

    def __init__(
        self,
        base: TCRPMHCDataset,
        held_out_epitopes: Set[str],
        mode: str = "exclude",
    ) -> None:
        if mode not in ("include", "exclude"):
            raise ValueError(f"mode must be 'include' or 'exclude'; got {mode!r}")
        peptides = base.df["Peptide"].astype(str).to_numpy()
        mask = np.isin(peptides, list(held_out_epitopes))
        if mode == "exclude":
            mask = ~mask
        indices = np.nonzero(mask)[0].tolist()
        super().__init__(base, indices, tag=f"epitope_{mode}")


class PhlaHeldOutView(_IndexView):
    """Filter by a set of (Peptide, HLA) tuples."""

    def __init__(
        self,
        base: TCRPMHCDataset,
        held_out_phlas: Set[Tuple[str, str]],
        mode: str = "exclude",
    ) -> None:
        if mode not in ("include", "exclude"):
            raise ValueError(f"mode must be 'include' or 'exclude'; got {mode!r}")
        pep = base.df["Peptide"].astype(str).to_numpy()
        hla = base.df["HLA"].astype(str).to_numpy()
        phla_set = set(held_out_phlas)
        mask = np.array([(p, h) in phla_set for p, h in zip(pep, hla)])
        if mode == "exclude":
            mask = ~mask
        indices = np.nonzero(mask)[0].tolist()
        super().__init__(base, indices, tag=f"phla_{mode}")
