"""tcr_epi_collate_fn — produce the canonical Batch Contract.

==================================================================
**Mask semantics铁律（违反等同 bug，所有调用方必须遵守）：**

- 所有 ``*_pad_mask`` 输出均为 ``BoolTensor``，语义 **True = ignore**，
  可以直接喂 ``nn.MultiheadAttention(key_padding_mask=...)``
  与 ``nn.TransformerEncoder(src_key_padding_mask=...)``。

- 所有 ``*_valid_mask`` 输出均为 ``BoolTensor``，语义 **True = valid**，
  专门用于 **masked mean pool**（在 pool 氨基酸侧表征时排除 PAD / CLS / EOS / BOS 等控制 token）。

- **禁止** 下游模块自行从 ``*_pad_mask`` 推导 ``*_valid_mask`` 或反之；必须使用
  collate 提供的字段。两者在 CLS / EOS 位置上的取值**不相等**——`pad_mask=False`
  但 `valid_mask=False` 表示"该位不是 pad，但也不应被 mean pool 消费"。
==================================================================

Batch Contract（DATA_CONTRACT §6）::

    antigen_tokens       LongTensor [B, La]       ESM-C ids
    antigen_pad_mask     BoolTensor [B, La]       True = ignore
    antigen_valid_mask   BoolTensor [B, La]       True = valid（排除 PAD/CLS/EOS）

    mhc_tokens           LongTensor [B, Lm]
    mhc_pad_mask         BoolTensor [B, Lm]
    mhc_valid_mask       BoolTensor [B, Lm]

    tcr_tokens           LongTensor [B, Lt, 4]
    tcr_pad_mask         BoolTensor [B, Lt]       = tcr_tokens[..., 0] == NULL
    tcr_valid_mask       BoolTensor [B, Lt]       True = valid（排除 PAD + CLS）

    label                FloatTensor [B]          {0.0, 1.0}
    meta                 list[dict]               仅 diagnostic（HLA / Va / Vb …）

==================================================================

DataLoader 多 worker 约束：

- collate 必须在 **CPU** 端运行；任何 ``.cuda()`` / ``.to(device)`` 都是禁止的。
- tokenizer 实例由 collate 的调用者（``TcrEpiCollator`` 或 ``functools.partial``）
  注入；worker 进程在 ``fork`` 启动方式下直接继承，``spawn`` 启动方式下要求
  tokenizer 可 pickle（对 ESM-C backbone 可能需要先 .cpu()）。
- 若要并行，使用 ``TcrEpiCollator(pmhc_tok, tcr_tok)`` 作为 ``DataLoader(collate_fn=...)``
  的 value——函数句柄比 ``partial`` 更稳地在各平台 pickle 成功。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .tokenizers import CdrTcrTokenizer, PMHCESMTokenizer, TcrCompartmentIndex, TcrTokenIndex

logger = logging.getLogger(__name__)

__all__ = [
    "BATCH_KEYS",
    "PMHC_BINDING_BATCH_KEYS",
    "tcr_epi_collate_fn",
    "TcrEpiCollator",
    "pmhc_binding_collate_fn",
    "PMHCBindingCollator",
]


BATCH_KEYS = (
    "antigen_tokens", "antigen_pad_mask", "antigen_valid_mask",
    "mhc_tokens", "mhc_pad_mask", "mhc_valid_mask",
    "tcr_tokens", "tcr_pad_mask", "tcr_valid_mask",
    "label", "meta",
)


PMHC_BINDING_BATCH_KEYS = (
    "antigen_tokens", "antigen_pad_mask", "antigen_valid_mask",
    "mhc_tokens", "mhc_pad_mask", "mhc_valid_mask",
    "label", "meta",
)
"""Batch keys for the Stage-1 pMHC binding collate — no TCR fields (the
:class:`PMHCBindingDataset` never carries CDRs). Same mask semantics as
:data:`BATCH_KEYS` (``*_pad_mask`` → True = ignore,
``*_valid_mask`` → True = valid, CLS/BOS/EOS excluded)."""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _pmhc_valid_mask(
    tokens: torch.Tensor,
    pad_mask: torch.Tensor,
    tokenizer: PMHCESMTokenizer,
) -> torch.Tensor:
    """Build a True=valid mask that additionally excludes PAD / CLS / EOS.

    Defensive-or-in the pad id because some ESM-C versions emit
    ``attention_mask`` with all-ones for every token (including pad) when the
    batch happens to contain equal-length sequences — the redundancy guards
    against this edge case.
    """
    valid = pad_mask.logical_not()

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id

    # All three ids should come from the tokenizer attribute plumbing.
    if pad_id is not None:
        valid = valid & tokens.ne(int(pad_id))
    if cls_id is not None:
        valid = valid & tokens.ne(int(cls_id))
    if eos_id is not None:
        valid = valid & tokens.ne(int(eos_id))

    # Log once per process if every special id is missing — this is a serious
    # config smell (valid_mask ≡ ~pad_mask) but not fatal on its own.
    if pad_id is None and cls_id is None and eos_id is None:
        _warn_once_missing_special_ids(tokenizer.backbone_name)

    return valid


_WARNED_BACKBONES: set = set()


def _warn_once_missing_special_ids(backbone_name: str) -> None:
    if backbone_name in _WARNED_BACKBONES:
        return
    _WARNED_BACKBONES.add(backbone_name)
    logger.warning(
        "[collate] PMHCESMTokenizer(%s) has no pad/cls/eos ids resolved; "
        "antigen/mhc_valid_mask will degrade to '~pad_mask' (CLS/EOS may "
        "leak into mean pool). Provide pad_id_override or upgrade esm.",
        backbone_name,
    )


# ---------------------------------------------------------------------------
# Core collate function.
# ---------------------------------------------------------------------------


def tcr_epi_collate_fn(
    batch: Sequence[Mapping[str, Any]],
    pmhc_tokenizer: PMHCESMTokenizer,
    tcr_tokenizer: CdrTcrTokenizer,
) -> Dict[str, Any]:
    """Collate a list of dataset-style dicts into the Batch Contract.

    Parameters
    ----------
    batch
        Each element must match :meth:`TCRPMHCDataset._build_sample` output:
        ``{"peptide", "mhc_pseudo", "cdr_alpha", "cdr_beta", "label", "_meta"}``.
    pmhc_tokenizer
        Shared :class:`PMHCESMTokenizer` instance (CPU).
    tcr_tokenizer
        Shared :class:`CdrTcrTokenizer` instance (CPU).

    Returns
    -------
    dict
        Keys listed in :data:`BATCH_KEYS`. Every tensor lives on CPU.
    """
    if len(batch) == 0:
        raise ValueError("tcr_epi_collate_fn received an empty batch")

    # ---- Labels -----------------------------------------------------------
    labels = torch.tensor(
        [float(item["label"]) for item in batch],
        dtype=torch.float32,
    )

    # ---- pMHC tokenise ----------------------------------------------------
    peptides = [str(item["peptide"]) for item in batch]
    mhc_pseudos = [str(item["mhc_pseudo"]) for item in batch]
    pmhc_out = pmhc_tokenizer.encode_pair(peptides, mhc_pseudos)
    antigen_tokens: torch.Tensor = pmhc_out["antigen_tokens"]
    antigen_pad_mask: torch.Tensor = pmhc_out["antigen_pad_mask"]
    mhc_tokens: torch.Tensor = pmhc_out["mhc_tokens"]
    mhc_pad_mask: torch.Tensor = pmhc_out["mhc_pad_mask"]

    antigen_valid_mask = _pmhc_valid_mask(antigen_tokens, antigen_pad_mask, pmhc_tokenizer)
    mhc_valid_mask = _pmhc_valid_mask(mhc_tokens, mhc_pad_mask, pmhc_tokenizer)

    # ---- TCR tokenise -----------------------------------------------------
    tcr_tokens, tcr_pad_mask = tcr_tokenizer.encode_batch(list(batch))
    # tcr_pad_mask already == (aa_id == NULL)。
    cls_compartment = int(TcrCompartmentIndex.CLS)
    tcr_valid_mask = tcr_pad_mask.logical_not() & tcr_tokens[..., 3].ne(cls_compartment)

    # Optional defensive sanity check — NULL token-id positions must agree.
    # (Catches the case where someone edits the tokenizer in the future.)
    assert torch.equal(
        tcr_pad_mask, tcr_tokens[..., 0].eq(int(TcrTokenIndex.NULL))
    ), "tcr_pad_mask contract broken: must equal (aa_id == NULL)"

    # ---- Meta passthrough -------------------------------------------------
    meta: List[Dict[str, Any]] = [dict(item.get("_meta") or {}) for item in batch]

    out: Dict[str, Any] = {
        "antigen_tokens": antigen_tokens,
        "antigen_pad_mask": antigen_pad_mask,
        "antigen_valid_mask": antigen_valid_mask,
        "mhc_tokens": mhc_tokens,
        "mhc_pad_mask": mhc_pad_mask,
        "mhc_valid_mask": mhc_valid_mask,
        "tcr_tokens": tcr_tokens,
        "tcr_pad_mask": tcr_pad_mask,
        "tcr_valid_mask": tcr_valid_mask,
        "label": labels,
        "meta": meta,
    }
    return out


# ---------------------------------------------------------------------------
# Class wrapper for DataLoader(num_workers>0, pickle=True).
# ---------------------------------------------------------------------------


class TcrEpiCollator:
    """Callable wrapper over :func:`tcr_epi_collate_fn`.

    Using a module-level class (rather than ``functools.partial``) ensures
    that ``DataLoader(collate_fn=TcrEpiCollator(...))`` can be pickled under
    the ``spawn`` start method on macOS / Windows, provided the underlying
    tokenizers are picklable.

    Parameters
    ----------
    pmhc_tokenizer
        :class:`PMHCESMTokenizer`.
    tcr_tokenizer
        :class:`CdrTcrTokenizer`.
    """

    def __init__(
        self,
        pmhc_tokenizer: PMHCESMTokenizer,
        tcr_tokenizer: CdrTcrTokenizer,
    ) -> None:
        self.pmhc_tokenizer = pmhc_tokenizer
        self.tcr_tokenizer = tcr_tokenizer

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return tcr_epi_collate_fn(batch, self.pmhc_tokenizer, self.tcr_tokenizer)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<TcrEpiCollator pmhc={self.pmhc_tokenizer!r} "
            f"tcr={self.tcr_tokenizer!r}>"
        )


# ---------------------------------------------------------------------------
# Stage-1 pMHC binding collate (no TCR).
# ---------------------------------------------------------------------------


def pmhc_binding_collate_fn(
    batch: Sequence[Mapping[str, Any]],
    pmhc_tokenizer: PMHCESMTokenizer,
) -> Dict[str, Any]:
    """Collate for :class:`PMHCBindingDataset` → Stage-1 batch contract.

    Parameters
    ----------
    batch
        Each element must match :meth:`PMHCBindingDataset._build_sample`
        output: ``{"peptide", "mhc_pseudo", "label", "_meta"}``.
    pmhc_tokenizer
        Shared :class:`PMHCESMTokenizer` instance (CPU).

    Returns
    -------
    dict
        Keys listed in :data:`PMHC_BINDING_BATCH_KEYS`. No TCR tensors.
    """
    if len(batch) == 0:
        raise ValueError("pmhc_binding_collate_fn received an empty batch")

    labels = torch.tensor(
        [float(item["label"]) for item in batch],
        dtype=torch.float32,
    )

    peptides = [str(item["peptide"]) for item in batch]
    mhc_pseudos = [str(item["mhc_pseudo"]) for item in batch]
    pmhc_out = pmhc_tokenizer.encode_pair(peptides, mhc_pseudos)
    antigen_tokens: torch.Tensor = pmhc_out["antigen_tokens"]
    antigen_pad_mask: torch.Tensor = pmhc_out["antigen_pad_mask"]
    mhc_tokens: torch.Tensor = pmhc_out["mhc_tokens"]
    mhc_pad_mask: torch.Tensor = pmhc_out["mhc_pad_mask"]

    antigen_valid_mask = _pmhc_valid_mask(antigen_tokens, antigen_pad_mask, pmhc_tokenizer)
    mhc_valid_mask = _pmhc_valid_mask(mhc_tokens, mhc_pad_mask, pmhc_tokenizer)

    meta: List[Dict[str, Any]] = [dict(item.get("_meta") or {}) for item in batch]

    return {
        "antigen_tokens": antigen_tokens,
        "antigen_pad_mask": antigen_pad_mask,
        "antigen_valid_mask": antigen_valid_mask,
        "mhc_tokens": mhc_tokens,
        "mhc_pad_mask": mhc_pad_mask,
        "mhc_valid_mask": mhc_valid_mask,
        "label": labels,
        "meta": meta,
    }


class PMHCBindingCollator:
    """Picklable wrapper over :func:`pmhc_binding_collate_fn`."""

    def __init__(self, pmhc_tokenizer: PMHCESMTokenizer) -> None:
        self.pmhc_tokenizer = pmhc_tokenizer

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return pmhc_binding_collate_fn(batch, self.pmhc_tokenizer)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<PMHCBindingCollator pmhc={self.pmhc_tokenizer!r}>"


# Optional debug helper: print a one-line contract snapshot.
def _collate_snapshot(out: Dict[str, Any]) -> str:
    """Pretty-print the shape/dtype of a collated batch. Debug only."""
    return " | ".join(
        f"{k}={tuple(v.shape)}:{v.dtype}" if torch.is_tensor(v)
        else f"{k}=<{type(v).__name__} n={len(v)}>"
        for k, v in out.items()
    )
