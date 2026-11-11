"""HLA allele → 34aa class-I pseudo sequence lookup.

Ports the PUFFIN pseudo dictionary from ``tpmhc/mhc_poseudo.py`` (see
``_mhc_pseudo_class_i.MHC_PSEUDO_CLASS_I``) and adds the allele-normalization
probing rules from ``PYfile/MHC_Coverter.py``.

Per ``DATA_CONTRACT`` v0.1:
- class I: returns 34aa string;
- class II (``HLA-DR/DQ/DP``): returns a placeholder ``PseudoInfo`` with
  ``dtype="classII"`` and ``sequence=None``; caller (dataset layer) must drop
  these rows for v0.1.
- unknown: returns ``None``; caller must drop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from ._mhc_pseudo_class_i import MHC_PSEUDO_CLASS_I

__all__ = [
    "MHC_PSEUDO_CLASS_I",
    "PseudoInfo",
    "classify_hla",
    "normalize_hla_i",
    "fill_pseudo",
    "has_pseudo",
]


HLAClass = Literal["classI", "classII", "unknown"]


@dataclass(frozen=True)
class PseudoInfo:
    """Result of a pseudo-sequence lookup."""

    dtype: HLAClass                 # "classI" / "classII" / "unknown"
    sequence: Optional[str]         # class I: 34aa string; class II / unknown: None
    canonical: Optional[str] = None # normalized allele when lookup succeeded


# ---------------------------------------------------------------------------
# Class discrimination & normalization.
# ---------------------------------------------------------------------------

_CLASS_I_GENES = frozenset({"A", "B", "C", "E", "F", "G"})
_CLASS_II_PREFIXES = ("DRA", "DRB", "DQA", "DQB", "DPA", "DPB", "DMA", "DMB", "DOA", "DOB")

_HLA_STRIP_RE = re.compile(r"\s+")


def classify_hla(allele: str) -> HLAClass:
    """Classify an HLA string into class I / class II / unknown."""
    if allele is None:
        return "unknown"
    s = _HLA_STRIP_RE.sub("", str(allele).upper()).replace("_", "*")
    if s == "" or s == "NAN" or s == "UNKNOWN":
        return "unknown"

    # Strip leading "HLA-"
    stem = s[4:] if s.startswith("HLA-") else s

    # Class II: starts with one of the D-family locus tokens.
    if any(stem.startswith(p) for p in _CLASS_II_PREFIXES):
        return "classII"

    # Class I: single-letter gene + optional * + digits.
    if len(stem) >= 1 and stem[0] in _CLASS_I_GENES:
        return "classI"

    return "unknown"


def normalize_hla_i(allele: str) -> Optional[str]:
    """Canonicalize a class-I allele to ``HLA-X*NN:NN`` (4-digit).

    Returns ``None`` if the string is not a class I form.
    Accepts inputs like ``HLA-A*01:01:48`` / ``A01:01`` / ``HLA-A_01:01``.
    """
    if allele is None:
        return None
    s = _HLA_STRIP_RE.sub("", str(allele).upper()).replace("_", "*")
    if not s:
        return None

    # Only probe A / B / C here to match PYfile/MHC_Coverter.py behaviour
    # (E / F / G exist in the table but in practice PUFFIN covers them via
    # exact-match candidates further down; we leave normalize_hla_i strict).
    m = re.match(
        r"^(?:HLA-)?([ABC])\*?(\d{2,3}):(\d{2,3})(?::\d{2,3})?(?::\d{2,3})?[NLSCAQ]?$",
        s,
    )
    if not m:
        return None

    gene, f1, f2 = m.group(1), m.group(2), m.group(3)
    # PUFFIN 字典里的 key 统一用 2 位冒号数字（会少量溢出，但足够覆盖我们现有数据）
    return f"HLA-{gene}*{f1.zfill(2)}:{f2.zfill(2)}"


def _probe_candidates(canon: str) -> list[str]:
    """Build lookup candidates (aligned with PYfile/MHC_Coverter.py)."""
    return [
        canon,
        canon.replace("HLA-", ""),                       # "A*01:01"
        canon.replace("*", ""),                          # "HLA-A01:01"
        canon.replace("HLA-", "").replace("*", ""),      # "A01:01"
    ]


# ---------------------------------------------------------------------------
# Lookup API.
# ---------------------------------------------------------------------------


def fill_pseudo(hla: str) -> Optional[PseudoInfo]:
    """Return ``PseudoInfo`` describing an HLA allele, or ``None`` if unknown.

    Semantics (DATA_CONTRACT v0.1):
    - ``classI`` hit              → ``PseudoInfo(dtype='classI', sequence=<34aa>, canonical=...)``
    - ``classII``                 → ``PseudoInfo(dtype='classII', sequence=None,  canonical=None)``
    - class-I shaped but no entry → ``None`` (caller should drop the row)
    - malformed / empty           → ``None``
    """
    if hla is None:
        return None

    cls = classify_hla(hla)
    if cls == "classII":
        return PseudoInfo(dtype="classII", sequence=None, canonical=None)
    if cls == "unknown":
        return None

    canon = normalize_hla_i(hla)
    if canon is None:
        # class-I-ish prefix (E/F/G) that normalize_hla_i rejects (strict to A/B/C).
        # Try one more direct hit on the raw string before giving up.
        raw = _HLA_STRIP_RE.sub("", str(hla).upper()).replace("_", "*")
        if raw in MHC_PSEUDO_CLASS_I:
            return PseudoInfo(dtype="classI", sequence=MHC_PSEUDO_CLASS_I[raw], canonical=raw)
        return None

    for cand in _probe_candidates(canon):
        seq = MHC_PSEUDO_CLASS_I.get(cand)
        if seq is not None:
            return PseudoInfo(dtype="classI", sequence=seq, canonical=canon)
    return None


def has_pseudo(hla: str) -> bool:
    """True iff ``fill_pseudo(hla)`` yields a class-I hit with a 34aa sequence."""
    info = fill_pseudo(hla)
    return info is not None and info.dtype == "classI" and info.sequence is not None
