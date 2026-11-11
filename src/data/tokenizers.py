"""Two tokenizers: ``PMHCESMTokenizer`` (pMHC side) + placeholder for ``CdrTcrTokenizer``.

步骤 5 实装 ``PMHCESMTokenizer``；步骤 6 再补 ``CdrTcrTokenizer``。

Design goals (DATA_CONTRACT §7.1 + step 5 prompt):

1. **Backbone API compatibility**: different ``esm`` versions expose different
   tokeniser interfaces. Primary path is
   ``backbone.tokenizer(seqs, padding=True, return_tensors="pt")`` →
   ``{"input_ids", "attention_mask"}``; fallback path is single-sequence
   ``backbone.encode(ESMProtein(sequence=s)).sequence`` + manual padding.
2. **Dynamic special ids**: ``pad / cls(=bos) / eos`` are read from the
   tokenizer via ``getattr`` — never hard-coded.
3. **Singleton by convention**: caller should build **one** instance per
   backbone and reuse across the entire DataLoader / train loop. A tiny
   module-level cache (:func:`get_pmhc_tokenizer`) is provided for this.
4. **CPU-only tokenisation**: output tensors live on CPU; the caller is
   responsible for moving to GPU. The tokenizer itself must never appear
   inside a forward pass.
5. **Testable without real ESM-C**: the backbone module can be any object
   that *either* exposes ``.tokenizer(seqs, padding=True, return_tensors="pt")``
   *or* ``.encode(protein_obj)`` — stubs in unit tests exploit this.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "PMHCESMTokenizer",
    "SpecialTokenIds",
    "get_pmhc_tokenizer",
    "reset_pmhc_tokenizer_cache",
    "TcrTokenIndex",
    "TcrCompartmentIndex",
    "AMINO_ACID_ORDER",
    "CdrTcrTokenizer",
]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecialTokenIds:
    """Resolved special-token ids for the current backbone."""

    pad: Optional[int]
    bos: Optional[int]  # CLS for ESM / BOS alias
    eos: Optional[int]

    def as_dict(self) -> Dict[str, Optional[int]]:
        return {"pad": self.pad, "bos": self.bos, "eos": self.eos}


def _default_protein_wrap(seq: str):
    """Default ``backbone.encode`` argument factory — uses ``ESMProtein``.

    Imported lazily so that environments without ``esm`` installed can still
    exercise the *primary* path in unit tests.
    """
    from esm.sdk.api import ESMProtein  # local import on purpose

    return ESMProtein(sequence=seq)


def _resolve_special_ids_from_hf(tok: Any) -> SpecialTokenIds:
    """Read pad / cls(bos) / eos from a HuggingFace-style tokenizer."""
    pad = getattr(tok, "pad_token_id", None)
    bos = getattr(tok, "cls_token_id", None)
    if bos is None:
        bos = getattr(tok, "bos_token_id", None)
    eos = getattr(tok, "eos_token_id", None)
    return SpecialTokenIds(pad=pad, bos=bos, eos=eos)


_ESM2_ALIASES: Dict[str, str] = {
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2-650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_650": "facebook/esm2_t33_650M_UR50D",
    "esm2-650": "facebook/esm2_t33_650M_UR50D",
    "esm2_t33_650m_ur50d": "facebook/esm2_t33_650M_UR50D",
}
_ESM3_ALIASES: Dict[str, str] = {
    "esm3_1_4b": "esm3_sm_open_v1",
    "esm3-1.4b": "esm3_sm_open_v1",
    "esm3_sm": "esm3_sm_open_v1",
    "esm3-small": "esm3_sm_open_v1",
    "esm3_sm_open": "esm3_sm_open_v1",
    "esm3-sm-open-v1": "esm3_sm_open_v1",
}


def _canonical_backbone_name(name: str) -> str:
    key = str(name).strip()
    lower = key.lower()
    return _ESM2_ALIASES.get(lower, _ESM3_ALIASES.get(lower, key))


def _is_hf_esm2_name(name: str) -> bool:
    resolved = _canonical_backbone_name(name).lower()
    return resolved.startswith("facebook/esm2_") or "/esm2_" in resolved


def _is_esm3_name(name: str) -> bool:
    resolved = _canonical_backbone_name(name).lower()
    return resolved.startswith("esm3") or "esm3" in resolved


class _HFTokenizerBackbone:
    """Minimal object matching the tokenizer-wrapper duck type."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer


# ---------------------------------------------------------------------------
# Core class.
# ---------------------------------------------------------------------------


class PMHCESMTokenizer:
    """ESM-C sequence tokeniser wrapper for pMHC inputs.

    Parameters
    ----------
    backbone_module
        Any object satisfying the duck-typed interface: must expose **either**

        - ``.tokenizer`` (HuggingFace-style, callable with
          ``(seqs, padding=True, return_tensors='pt')``), **or**
        - ``.encode(protein_obj) -> object with .sequence : LongTensor``.

        For real usage this is an ``esm.models.esmc.ESMC`` instance.
    backbone_name
        Optional human-readable identifier (used for logging / cache keys).
    allow_fallback
        If ``True`` (default) and ``.tokenizer`` is missing, fall back to the
        per-sequence ``.encode`` path. ``False`` raises immediately.
    pad_id_override
        Optional explicit pad id for the fallback path when the backbone
        doesn't expose a tokenizer attribute (some esm builds).
    protein_wrap
        Callable ``str -> protein_obj`` used by the fallback path. Defaults
        to building ``esm.sdk.api.ESMProtein``; tests may pass an identity
        lambda to avoid the ``esm`` import.
    """

    def __init__(
        self,
        backbone_module: Any,
        *,
        backbone_name: Optional[str] = None,
        allow_fallback: bool = True,
        pad_id_override: Optional[int] = None,
        protein_wrap: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if backbone_module is None:
            raise ValueError("backbone_module must not be None")

        self.backbone = backbone_module
        self.backbone_name = backbone_name or type(backbone_module).__name__

        self._hf_tok = getattr(backbone_module, "tokenizer", None)
        if self._hf_tok is None:
            tokenizers = getattr(backbone_module, "tokenizers", None)
            self._hf_tok = getattr(tokenizers, "sequence", None)
        if self._hf_tok is not None and not callable(self._hf_tok):
            logger.warning(
                "[pmhc-tok] backbone.tokenizer exists but is not callable; "
                "treating as fallback-only"
            )
            self._hf_tok = None

        self.use_fallback = self._hf_tok is None
        if self.use_fallback and not allow_fallback:
            raise RuntimeError(
                f"backbone {self.backbone_name!r} has no '.tokenizer' attribute "
                "and allow_fallback=False"
            )

        self._protein_wrap = protein_wrap or _default_protein_wrap
        self.special = self._resolve_special_ids(pad_id_override)

        logger.info(
            "[pmhc-tok] backbone=%s path=%s special=%s",
            self.backbone_name,
            "hf" if not self.use_fallback else "fallback",
            self.special.as_dict(),
        )

    # ----- Construction helpers -----

    @classmethod
    def from_backbone_name(
        cls,
        name: str,
        *,
        device: str = "cpu",
        dtype: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> "PMHCESMTokenizer":
        """Lazy-load a pMHC backbone tokenizer and wrap it.

        ``device`` defaults to CPU because the tokenizer itself never needs
        GPU; the caller can move the actual backbone elsewhere independently.
        """
        resolved_name = _canonical_backbone_name(name)
        if _is_hf_esm2_name(resolved_name):
            from transformers import AutoTokenizer  # lazy import

            tokenizer = AutoTokenizer.from_pretrained(resolved_name)
            return cls(
                _HFTokenizerBackbone(tokenizer),
                backbone_name=resolved_name,
                allow_fallback=False,
            )
        if _is_esm3_name(resolved_name):
            from esm.models.esm3 import ESM3  # lazy import

            torch_device = torch.device(device) if device is not None else None
            backbone = ESM3.from_pretrained(resolved_name, device=torch_device)
            if dtype:
                torch_dtype = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }[dtype]
                backbone = backbone.to(dtype=torch_dtype)
            return cls(backbone, backbone_name=resolved_name, allow_fallback=allow_fallback)

        from esm.models.esmc import ESMC  # lazy import

        backbone = ESMC.from_pretrained(resolved_name)
        if dtype:
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype]
            backbone = backbone.to(dtype=torch_dtype)
        backbone = backbone.to(device)
        return cls(backbone, backbone_name=resolved_name, allow_fallback=allow_fallback)

    # ----- Special ids resolution -----

    def _resolve_special_ids(
        self,
        pad_id_override: Optional[int],
    ) -> SpecialTokenIds:
        if self._hf_tok is not None:
            special = _resolve_special_ids_from_hf(self._hf_tok)
            if pad_id_override is not None and special.pad is None:
                special = SpecialTokenIds(
                    pad=pad_id_override, bos=special.bos, eos=special.eos
                )
            return special

        # Fallback path: probe BOS/EOS from the shortest legal sequence.
        try:
            probe = self.backbone.encode(self._protein_wrap("A"))
            probe_ids = getattr(probe, "sequence", probe)
            if torch.is_tensor(probe_ids) and probe_ids.dim() == 2:
                probe_ids = probe_ids[0]
            ids_list: List[int] = (
                probe_ids.tolist() if torch.is_tensor(probe_ids) else list(probe_ids)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[pmhc-tok] probe encode failed: %s", exc)
            ids_list = []

        bos = ids_list[0] if len(ids_list) >= 1 else None
        eos = ids_list[-1] if len(ids_list) >= 2 else None
        return SpecialTokenIds(pad=pad_id_override, bos=bos, eos=eos)

    # ----- Public encode API -----

    def encode_batch(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenise a list of sequences.

        Returns
        -------
        input_ids   : ``LongTensor [B, L]`` on CPU
        pad_mask    : ``BoolTensor [B, L]`` on CPU — **True = ignore**
        """
        if not seqs:
            return (
                torch.zeros(0, 0, dtype=torch.long),
                torch.zeros(0, 0, dtype=torch.bool),
            )

        seqs_list = [self._normalize_seq(s) for s in seqs]
        if not self.use_fallback:
            return self._encode_via_hf(seqs_list)
        return self._encode_via_backbone(seqs_list)

    def encode_pair(
        self,
        peptides: Sequence[str],
        mhc_pseudos: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Tokenise paired ``peptide / mhc_pseudo`` lists into batch tensors.

        Output keys (DATA_CONTRACT §6):
            ``antigen_tokens / antigen_pad_mask / mhc_tokens / mhc_pad_mask``.
        """
        if len(peptides) != len(mhc_pseudos):
            raise ValueError(
                f"peptides and mhc_pseudos must have the same length; "
                f"got {len(peptides)} vs {len(mhc_pseudos)}"
            )
        ant_ids, ant_pad = self.encode_batch(peptides)
        mhc_ids, mhc_pad = self.encode_batch(mhc_pseudos)
        return {
            "antigen_tokens": ant_ids,
            "antigen_pad_mask": ant_pad,
            "mhc_tokens": mhc_ids,
            "mhc_pad_mask": mhc_pad,
        }

    # ----- Internal encoders -----

    def _encode_via_hf(
        self, seqs: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self._hf_tok(seqs, padding=True, return_tensors="pt")
        if not isinstance(out, dict):
            out = {k: v for k, v in out.items()}  # BatchEncoding → dict-like
        input_ids = out["input_ids"].to(torch.long).cpu()
        attn = out.get("attention_mask")
        if attn is None:
            # No attention_mask returned; derive it from pad_id if available.
            if self.special.pad is None:
                # No way to tell pad positions; assume everything is valid.
                pad_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            else:
                pad_mask = input_ids.eq(self.special.pad)
        else:
            pad_mask = attn.to(torch.bool).cpu().logical_not()
        return input_ids, pad_mask

    def _encode_via_backbone(
        self, seqs: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.special.pad is None:
            raise RuntimeError(
                "fallback path requires a known pad_id; backbone exposes no "
                "tokenizer and no pad_id_override was provided. Pass "
                "pad_id_override=<int> when constructing PMHCESMTokenizer."
            )
        per_seq: List[torch.Tensor] = []
        for s in seqs:
            encoded = self.backbone.encode(self._protein_wrap(s))
            ids = getattr(encoded, "sequence", encoded)
            if not torch.is_tensor(ids):
                ids = torch.as_tensor(ids)
            if ids.dim() == 2:
                ids = ids[0]
            per_seq.append(ids.to(dtype=torch.long).cpu())

        max_len = max(int(x.size(0)) for x in per_seq)
        B = len(per_seq)
        out = torch.full((B, max_len), fill_value=int(self.special.pad), dtype=torch.long)
        pad_mask = torch.ones((B, max_len), dtype=torch.bool)  # True = ignore
        for i, ids in enumerate(per_seq):
            L = int(ids.size(0))
            out[i, :L] = ids
            pad_mask[i, :L] = False
        return out, pad_mask

    # ----- Misc helpers -----

    @staticmethod
    def _normalize_seq(s: str) -> str:
        if s is None:
            return ""
        return str(s).strip().upper()

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.special.pad

    @property
    def cls_token_id(self) -> Optional[int]:
        return self.special.bos

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.special.bos

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.special.eos

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<PMHCESMTokenizer backbone={self.backbone_name} "
            f"path={'hf' if not self.use_fallback else 'fallback'} "
            f"pad={self.special.pad} cls={self.special.bos} eos={self.special.eos}>"
        )


# ---------------------------------------------------------------------------
# Process-local singleton cache.
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_TOKENIZER_CACHE: Dict[Tuple[str, str, str], PMHCESMTokenizer] = {}


def get_pmhc_tokenizer(
    backbone_name: str,
    *,
    device: str = "cpu",
    dtype: Optional[str] = None,
) -> PMHCESMTokenizer:
    """Return a cached :class:`PMHCESMTokenizer`, creating on first use.

    Caching key: ``(backbone_name, device, dtype)``. Each DataLoader worker
    process has its own cache; that's intentional — we don't want to ship an
    ESM-C model across multiprocessing boundaries.
    """
    key = (backbone_name, device, dtype or "default")
    with _CACHE_LOCK:
        tok = _TOKENIZER_CACHE.get(key)
        if tok is None:
            tok = PMHCESMTokenizer.from_backbone_name(
                backbone_name, device=device, dtype=dtype
            )
            _TOKENIZER_CACHE[key] = tok
        return tok


def reset_pmhc_tokenizer_cache() -> None:
    """Drop every cached tokenizer (useful in tests)."""
    with _CACHE_LOCK:
        _TOKENIZER_CACHE.clear()


# ---------------------------------------------------------------------------
# CdrTcrTokenizer — SCEPTR-style four-tuple tokenizer (step 6).
# ---------------------------------------------------------------------------

# Contract: every TCR position is a LongTensor of shape [4] layed out as
#   [aa_id, within_compartment_pos, compartment_total_len, compartment_id]
# matching libtcrlm's CdrEmbedder so a future port of SCEPTR weights can
# consume our tensors without any index re-mapping.

from enum import IntEnum  # noqa: E402  — kept close to its only consumers


class TcrTokenIndex(IntEnum):
    """Token vocabulary for the TCR tower. Must match ``ModelConfig.tcr_vocab_size``."""

    NULL = 0   # padding / missing residue
    MASK = 1   # MLM masked residue
    CLS = 2    # sequence prefix

    # 20 standard amino acids in the canonical libtcrlm ``AminoAcid`` order
    # (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
    A = 3
    C = 4
    D = 5
    E = 6
    F = 7
    G = 8
    H = 9
    I = 10
    K = 11
    L = 12
    M = 13
    N = 14
    P = 15
    Q = 16
    R = 17
    S = 18
    T = 19
    V = 20
    W = 21
    Y = 22


class TcrCompartmentIndex(IntEnum):
    """Compartment ids. Must match ``ModelConfig.tcr_n_compartments``."""

    NULL = 0   # padding / unused slot
    CLS = 1
    CDR1a = 2
    CDR2a = 3
    CDR3a = 4
    CDR1b = 5
    CDR2b = 6
    CDR3b = 7


AMINO_ACID_ORDER: str = "ACDEFGHIKLMNPQRSTVWY"
assert len(AMINO_ACID_ORDER) == 20


# Build the AA → vocab id lookup once at module import time.
_AA_TO_ID: Dict[str, int] = {c: TcrTokenIndex[c].value for c in AMINO_ACID_ORDER}


class CdrTcrTokenizer:
    """Paired αβ CDR tokenizer producing ``[B, Lt, 4]`` LongTensors.

    Layout (DATA_CONTRACT §7.2)::

        [CLS, αCDR1, αCDR2, αCDR3, βCDR1, βCDR2, βCDR3]

    Every CDR slot is **fixed width** equal to its budget
    (``cdr1_budget / cdr2_budget / cdr3_budget`` from :class:`LengthConfig`).
    Shorter CDRs are padded with all-zero (NULL) four-tuples — this guarantees
    every sample in a batch has identical ``sequence_length`` and keeps the
    sinusoidal / segment embeddings aligned.  Longer CDRs are **truncated**
    (dataset-level validation already drops rows that exceed budgets, so
    truncation is only a defensive fallback).

    Parameters
    ----------
    length_cfg
        Source of per-chain CDR budgets. Defaults to :class:`LengthConfig()`.

    Attributes
    ----------
    sequence_length
        ``1 + 2 * (cdr1_budget + cdr2_budget + cdr3_budget)``.
    special_ids
        Dict mapping ``{"NULL","MASK","CLS"}`` and ``compartments`` to their
        integer ids. Consumers (collate / embedder) must use these instead of
        hard-coding indices.
    """

    def __init__(self, length_cfg: Optional[Any] = None) -> None:
        # Lazy-import to avoid a config ↔ data cycle during packaging.
        if length_cfg is None:
            from ..config import LengthConfig  # local import

            length_cfg = LengthConfig()
        self.length_cfg = length_cfg

        self._segment_layout: List[Tuple[TcrCompartmentIndex, int]] = [
            (TcrCompartmentIndex.CDR1a, int(length_cfg.cdr1_budget)),
            (TcrCompartmentIndex.CDR2a, int(length_cfg.cdr2_budget)),
            (TcrCompartmentIndex.CDR3a, int(length_cfg.cdr3_budget)),
            (TcrCompartmentIndex.CDR1b, int(length_cfg.cdr1_budget)),
            (TcrCompartmentIndex.CDR2b, int(length_cfg.cdr2_budget)),
            (TcrCompartmentIndex.CDR3b, int(length_cfg.cdr3_budget)),
        ]
        self.sequence_length = 1 + sum(b for _, b in self._segment_layout)
        if self.sequence_length > int(length_cfg.tcr_tokens):
            raise ValueError(
                f"layout length {self.sequence_length} exceeds "
                f"LengthConfig.tcr_tokens={length_cfg.tcr_tokens}"
            )

        self._truncated_warned: set = set()

    # ----- Metadata -----

    @property
    def vocab_size(self) -> int:
        return len(TcrTokenIndex)

    @property
    def n_compartments(self) -> int:
        return len(TcrCompartmentIndex)

    @property
    def special_ids(self) -> Dict[str, Any]:
        return {
            "NULL": int(TcrTokenIndex.NULL),
            "MASK": int(TcrTokenIndex.MASK),
            "CLS": int(TcrTokenIndex.CLS),
            "compartments": {c.name: int(c.value) for c in TcrCompartmentIndex},
            "segment_layout": [
                {"compartment": c.name, "budget": b} for c, b in self._segment_layout
            ],
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "n_compartments": self.n_compartments,
        }

    # ----- Core encoding -----

    def encode_one(
        self,
        cdr_alpha: Optional[Dict[str, str]] = None,
        cdr_beta: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Encode one paired-αβ sample into ``[sequence_length, 4]``.

        ``cdr_alpha`` / ``cdr_beta`` are dicts with keys ``"cdr1", "cdr2", "cdr3"``;
        either may be ``None`` or contain empty strings (handled as NULL-padding
        within the corresponding compartment slot).
        """
        cdr_alpha = cdr_alpha or {}
        cdr_beta = cdr_beta or {}
        segments_text: List[str] = [
            self._norm_cdr(cdr_alpha.get("cdr1")),
            self._norm_cdr(cdr_alpha.get("cdr2")),
            self._norm_cdr(cdr_alpha.get("cdr3")),
            self._norm_cdr(cdr_beta.get("cdr1")),
            self._norm_cdr(cdr_beta.get("cdr2")),
            self._norm_cdr(cdr_beta.get("cdr3")),
        ]

        tokens = torch.zeros(self.sequence_length, 4, dtype=torch.long)

        # CLS at position 0.
        tokens[0, 0] = int(TcrTokenIndex.CLS)
        tokens[0, 1] = 0
        tokens[0, 2] = 1
        tokens[0, 3] = int(TcrCompartmentIndex.CLS)

        cursor = 1
        for (compartment, budget), cdr_str in zip(self._segment_layout, segments_text):
            actual = min(len(cdr_str), budget)
            if len(cdr_str) > budget:
                key = (compartment.name, len(cdr_str))
                if key not in self._truncated_warned:
                    logger.warning(
                        "[cdr-tok] truncating %s from len=%d to budget=%d (once per length)",
                        compartment.name, len(cdr_str), budget,
                    )
                    self._truncated_warned.add(key)
            for i in range(actual):
                c = cdr_str[i]
                aa_id = _AA_TO_ID.get(c, int(TcrTokenIndex.NULL))
                tokens[cursor + i, 0] = aa_id
                tokens[cursor + i, 1] = i
                tokens[cursor + i, 2] = actual
                tokens[cursor + i, 3] = int(compartment)
            # Remaining [cursor + actual : cursor + budget] stays all-zero (NULL).
            cursor += budget

        return tokens

    def encode_batch(
        self,
        samples: Sequence[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of dataset-style dicts.

        Each item must expose ``cdr_alpha`` and ``cdr_beta`` dicts matching
        :meth:`TCRPMHCDataset.__getitem__`. Returns

        - ``tcr_tokens: LongTensor [B, Lt, 4]``
        - ``tcr_pad_mask: BoolTensor [B, Lt]`` — **True = ignore**, computed
          strictly as ``tcr_tokens[:, :, 0] == NULL``.
        """
        B = len(samples)
        tokens = torch.zeros(B, self.sequence_length, 4, dtype=torch.long)
        if B == 0:
            pad_mask = torch.zeros(0, self.sequence_length, dtype=torch.bool)
            return tokens, pad_mask

        for i, sample in enumerate(samples):
            tokens[i] = self.encode_one(
                cdr_alpha=sample.get("cdr_alpha"),
                cdr_beta=sample.get("cdr_beta"),
            )

        pad_mask = tokens[:, :, 0].eq(int(TcrTokenIndex.NULL))
        return tokens, pad_mask

    # ----- Helpers -----

    @staticmethod
    def _norm_cdr(s: Optional[str]) -> str:
        if s is None:
            return ""
        return str(s).strip().upper()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<CdrTcrTokenizer seq_len={self.sequence_length} "
            f"vocab={self.vocab_size} compartments={self.n_compartments} "
            f"budget={self.length_cfg.per_chain_budget}>"
        )
