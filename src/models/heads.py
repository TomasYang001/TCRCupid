"""Prediction heads + shared pooling helper.

Implementations:

- :func:`masked_mean_pool` — token-level average with a True=valid mask.
- :class:`MLPClassifier`    — configurable MLP tower ending in 1-or-more logits.
- :class:`PMHCBindingHead`  — Stage-1 pMHC binding head (antigen side mean
  pool + small MLP → 1 logit).

Step 13's ``TCREpitopeSpecificityModel`` will reuse :func:`masked_mean_pool`
and build another :class:`MLPClassifier` with 4D concat-diff-mul input.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "masked_mean_pool",
    "GatedAttentionPool",
    "MLPClassifier",
    "PMHCBindingHead",
]


# ---------------------------------------------------------------------------
# Masked mean pool.
# ---------------------------------------------------------------------------


def masked_mean_pool(
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute ``sum(x * valid) / sum(valid)`` along the sequence axis.

    Parameters
    ----------
    x
        ``[B, L, D]`` token-level features.
    valid_mask
        ``[B, L]`` **bool** tensor, ``True`` = valid (per DATA_CONTRACT §6).
        Must be the collate-side ``*_valid_mask``, **not** ``~pad_mask``.
    eps
        Numerical floor for the denominator. Setting to a tiny positive keeps
        rows with zero-valid samples from producing NaN; such rows will emit
        an all-zero pooled vector (callers should avoid relying on this).

    Returns
    -------
    ``[B, D]`` pooled representation in the dtype of ``x``.
    """
    if x.dim() != 3:
        raise ValueError(f"masked_mean_pool expects x of shape [B, L, D]; got {tuple(x.shape)}")
    if valid_mask.dim() != 2:
        raise ValueError(
            f"masked_mean_pool expects valid_mask of shape [B, L]; got {tuple(valid_mask.shape)}"
        )
    if valid_mask.shape != x.shape[:2]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} must match x[:, :, :] prefix "
            f"{tuple(x.shape[:2])}"
        )
    if valid_mask.dtype != torch.bool:
        # Defensive: accept 0/1 ints but warn (silent casting would hide bugs).
        logger.warning(
            "[masked_mean_pool] valid_mask dtype=%s is not bool; casting (please fix at collate)",
            valid_mask.dtype,
        )
        valid_mask = valid_mask.to(torch.bool)

    w = valid_mask.unsqueeze(-1).to(x.dtype)     # [B, L, 1]
    denom = w.sum(dim=1).clamp_min(eps)          # [B, 1]
    numer = (x * w).sum(dim=1)                   # [B, D]
    return numer / denom


# ---------------------------------------------------------------------------
# GatedAttentionPool.
# ---------------------------------------------------------------------------


class GatedAttentionPool(nn.Module):
    """Mask-aware learnable pooling for sparse residue-level signals.

    The module scores each token independently, masks invalid positions, then
    returns a weighted sum over valid tokens. It is intentionally small: the
    cross-attention stack still performs interaction modelling, while this
    layer only decides which contextualised residues should dominate the
    sequence-level representation.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0; got {d_model}")
        if hidden_dim is None:
            hidden_dim = d_model
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0; got {hidden_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"GatedAttentionPool expects [B, L, D]; got {tuple(x.shape)}")
        if valid_mask.shape != x.shape[:2]:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} must match x prefix "
                f"{tuple(x.shape[:2])}"
            )
        if valid_mask.dtype != torch.bool:
            logger.warning(
                "[GatedAttentionPool] valid_mask dtype=%s is not bool; casting",
                valid_mask.dtype,
            )
            valid_mask = valid_mask.to(torch.bool)

        scores = self.net(x).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~valid_mask, -1.0e4)
        weights = torch.softmax(scores.float(), dim=1).to(dtype=x.dtype)
        weights = weights * valid_mask.to(dtype=x.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


# ---------------------------------------------------------------------------
# MLPClassifier.
# ---------------------------------------------------------------------------


class MLPClassifier(nn.Module):
    """Configurable MLP tower: ``(LN?) → [Linear, act, Dropout] × N → Linear(out)``.

    Parameters
    ----------
    in_dim
        Input feature dimension.
    hidden_dims
        Sequence of hidden widths. Length-0 is allowed → ``Linear(in_dim, out_dim)``
        directly (useful for tiny heads / ablation).
    out_dim
        Output dim, typically 1 for binary BCE.
    dropout
        Applied after each activation (not after the final Linear).
    activation
        ``"gelu"`` (default) or ``"relu"``.
    pre_norm
        When ``True`` (default), add a LayerNorm on the input.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (),
        *,
        out_dim: int = 1,
        dropout: float = 0.2,
        activation: Literal["gelu", "relu"] = "gelu",
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0; got {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0; got {out_dim}")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims must be all positive; got {list(hidden_dims)}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)

        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]

        layers: list[nn.Module] = []
        if pre_norm:
            layers.append(nn.LayerNorm(in_dim))

        prev = in_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PMHCBindingHead.
# ---------------------------------------------------------------------------


class PMHCBindingHead(nn.Module):
    """Stage-1 binding classifier: antigen mean pool → MLP → 1 logit.

    Input contract (DATA_CONTRACT §6):
        ``ant_tokens_repr``   : ``[B, La, D]``  — output of :class:`PMHCTower`.
        ``antigen_valid_mask``: ``[B, La]``     — True=valid (excludes CLS/EOS/PAD).

    The head is **Stage-1 only**. Stage-3 specificity uses a different head
    (built in step 13) and should never load this module's weights — the
    checkpoint format described in ``ckpt_adapter/load_pmhc_ckpt.py`` ensures
    the binding head is **not** persisted alongside the tower.
    """

    def __init__(
        self,
        d_model: int,
        *,
        hidden_mult: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0; got {d_model}")
        if hidden_mult <= 0:
            raise ValueError(f"hidden_mult must be > 0; got {hidden_mult}")

        self.d_model = int(d_model)
        h1 = d_model * hidden_mult
        h2 = d_model * max(1, hidden_mult // 2)
        self.classifier = MLPClassifier(
            in_dim=d_model,
            hidden_dims=(h1, h2),
            out_dim=1,
            dropout=dropout,
            activation="gelu",
            pre_norm=True,
        )

    def forward(
        self,
        ant_tokens_repr: torch.Tensor,
        antigen_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B]`` logits."""
        pooled = masked_mean_pool(ant_tokens_repr, antigen_valid_mask)  # [B, D]
        return self.classifier(pooled).squeeze(-1)  # [B]

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"d_model={self.d_model}"
