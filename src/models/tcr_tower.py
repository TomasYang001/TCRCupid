"""``TcrBert`` — tcrlm-style TCR tower, fully re-implemented (no libtcrlm import).

Three-layer composition (step 10 prompt):

1. :class:`CdrEmbedder`
   Sum of ``(aa, within-compartment sin-position, compartment)`` embeddings,
   scaled by ``sqrt(d_model)``. Mirrors ``libtcrlm.token_embedder.cdr_embedder.CdrEmbedder``.
2. :class:`SelfAttentionStack` (ABC)
   Abstract base pulled out so that future Roformer / Longformer variants
   can be swapped in without touching :class:`TcrBert`. The concrete
   :class:`TorchTransformerSelfAttentionStack` is the default backend
   (``nn.TransformerEncoder`` over ``nn.TransformerEncoderLayer``).
   The stack exposes the three behaviours that tcrlm-style pre-training relies on:

   - ``forward(x, padding_mask)``                              → all-layer output
   - ``get_token_embeddings_at_penultimate_layer(x, pad_mask)`` → last layer skipped
   - ``set_fine_tuning_mode(turn_on)``                         → freeze 0..N-2

3. :class:`TcrBert`
   Binds the embedder + stack. Extends tcrlm's ``Bert`` with:

   - :meth:`TcrBert.get_token_embeddings_of` — **new**; ``[B, L, D]``
     contextualised token embeddings for the downstream cross-attention
     (Step 12 / 13). No L2 norm. No mean-pool.
   - :meth:`TcrBert.get_vector_representations_of` — **optional**; tcrlm-style
     penultimate-layer + masked average pool + L2 norm, for contrastive
     pre-training (Stage-2 variants).

Padding mask convention (DATA_CONTRACT §6): ``tcr_pad_mask = tokens[:, :, 0] == NULL``
with ``True = ignore``, in sync with the collate.

Extensibility hook: pass any :class:`SelfAttentionStack` subclass into
:class:`TcrBert` and the rest of the package just works; see
``TcrBert.from_config`` for how the default stack is wired.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..data.tokenizers import TcrCompartmentIndex, TcrTokenIndex

logger = logging.getLogger(__name__)

__all__ = [
    "SinPositionEmbedding",
    "CdrEmbedder",
    "SelfAttentionStack",
    "TorchTransformerSelfAttentionStack",
    "RoFormerSelfAttentionStack",
    "ReformerSelfAttentionStack",
    "TcrBert",
    "DEFAULT_MAX_POSITION_LEN",
    "LOCATION_OF_CLS_TOKEN",
]


DEFAULT_MAX_POSITION_LEN: int = 100
"""Matches ``libtcrlm.token_embedder.cdr_embedder.MAX_PLAUSIBLE_CDR_LENGTH``."""

LOCATION_OF_CLS_TOKEN: int = 0
"""The fixed slot inside ``tcr_tokens`` that holds the CLS four-tuple (see
CdrTcrTokenizer §2.1 layout). Used in masked-mean-pool to exclude CLS."""


# ---------------------------------------------------------------------------
# SinPositionEmbedding.
# ---------------------------------------------------------------------------


class SinPositionEmbedding(nn.Module):
    """Non-parametric sinusoidal position embedding (tcrlm reference).

    The lookup buffer has ``num_embeddings + 1`` rows:

    - Row ``0`` is all zeros — used for pad / CLS / unused slots (callers
      are responsible for shifting *real* residue positions by ``+1`` before
      indexing).
    - Rows ``1..num_embeddings`` hold the sin/cos pair for positions
      ``0..num_embeddings - 1`` (i.e. 0-based residue positions shifted to
      1-based matrix indices).

    Parameters
    ----------
    num_embeddings
        Number of *real* residue positions supported (sin/cos rows stored
        are ``num_embeddings``, one extra pad row adds up to ``num_embeddings + 1``).
    embedding_dim
        Must be even. Sin takes the even halves, cos takes the odd halves.
    sin_scale_factor
        Wavelength base; defaults to tcrlm's ``30``. Smaller = faster
        frequency sweep, 30 is well-tested for short CDR sequences.
    """

    def __init__(
        self,
        num_embeddings: int = DEFAULT_MAX_POSITION_LEN,
        embedding_dim: int = 64,
        sin_scale_factor: float = 30.0,
    ) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even; got {embedding_dim}")
        if num_embeddings < 1:
            raise ValueError(f"num_embeddings must be >= 1; got {num_embeddings}")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.sin_scale_factor = float(sin_scale_factor)

        matrix = self._compute_matrix(
            self.num_embeddings, self.embedding_dim, self.sin_scale_factor
        )
        # persistent=True to stay byte-compatible with tcrlm state_dicts.
        self.register_buffer("position_embedding_matrix", matrix, persistent=True)

    @staticmethod
    def _compute_matrix(
        num_embeddings: int, embedding_dim: int, sin_scale_factor: float
    ) -> torch.Tensor:
        rows = torch.zeros(num_embeddings + 1, embedding_dim)
        pos_col = torch.arange(0, num_embeddings, dtype=torch.float32).unsqueeze(1)
        scaler = torch.pow(
            torch.tensor(sin_scale_factor, dtype=torch.float32),
            torch.arange(0, embedding_dim, 2, dtype=torch.float32) / embedding_dim,
        )
        rows[1:, 0::2] = torch.sin(pos_col / scaler)
        rows[1:, 1::2] = torch.cos(pos_col / scaler)
        return rows

    def forward(self, shifted_indices: torch.Tensor) -> torch.Tensor:
        """Look up rows. ``shifted_indices`` must already be in ``[0, num_embeddings]``."""
        return self.position_embedding_matrix[shifted_indices]


# ---------------------------------------------------------------------------
# CdrEmbedder.
# ---------------------------------------------------------------------------


class CdrEmbedder(nn.Module):
    """Sum of aa + sin-position + compartment embeddings, scaled by √d.

    Input shape is ``[B, L, 4]`` with channels
    ``[aa_id, within_compartment_pos, compartment_total_len, compartment_id]``
    (DATA_CONTRACT §7.2). ``within_compartment_pos`` is **0-based** per our
    tokenizer contract; the shift to tcrlm's 1-based sin table happens
    locally inside this module.

    Padding / CLS handling:

    - A slot is considered a "real residue" iff ``aa_id != NULL`` **and**
      ``compartment_id != CLS``. Real residues contribute their
      ``sin[pos_raw + 1]`` row.
    - Pad (``aa_id == NULL``) and CLS (``compartment_id == CLS``) slots
      look up the zero row, matching tcrlm's behaviour where ``padding_idx=NULL``
      zeros out both aa and compartment embeddings — so the three components
      *all* become zero on those slots.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_compartments: int,
        *,
        max_position_len: int = DEFAULT_MAX_POSITION_LEN,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 2 != 0:
            raise ValueError(f"d_model must be a positive even int; got {d_model}")
        self._d_model = int(d_model)
        self._max_pos = int(max_position_len)
        self._null_aa_id = int(TcrTokenIndex.NULL)
        self._cls_compartment_id = int(TcrCompartmentIndex.CLS)

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=int(TcrTokenIndex.NULL),
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=self._max_pos,
            embedding_dim=d_model,
        )
        self.compartment_embedding = nn.Embedding(
            num_embeddings=n_compartments,
            embedding_dim=d_model,
            padding_idx=int(TcrCompartmentIndex.NULL),
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, tcr_tokens: torch.Tensor) -> torch.Tensor:
        """``[B, L, 4]`` → ``[B, L, D]`` embeddings × √d."""
        if tcr_tokens.dim() != 3 or tcr_tokens.size(-1) != 4:
            raise ValueError(
                f"tcr_tokens must have shape [B, L, 4]; got {tuple(tcr_tokens.shape)}"
            )
        if tcr_tokens.dtype not in (torch.long, torch.int64, torch.int32):
            raise TypeError(
                f"tcr_tokens must be integer LongTensor; got dtype={tcr_tokens.dtype}"
            )

        aa_id = tcr_tokens[..., 0]
        pos_raw = tcr_tokens[..., 1]
        compartment = tcr_tokens[..., 3]

        is_real_residue = (aa_id != self._null_aa_id) & (
            compartment != self._cls_compartment_id
        )
        pos_shifted = torch.where(
            is_real_residue, pos_raw + 1, torch.zeros_like(pos_raw)
        )
        pos_shifted = pos_shifted.clamp(min=0, max=self._max_pos)

        token_c = self.token_embedding(aa_id)
        position_c = self.position_embedding(pos_shifted)
        compart_c = self.compartment_embedding(compartment)

        summed = token_c + position_c + compart_c
        return summed * math.sqrt(self._d_model)


# ---------------------------------------------------------------------------
# SelfAttentionStack — abstract base + torch default.
# ---------------------------------------------------------------------------


class SelfAttentionStack(nn.Module, ABC):
    """Abstract TCR self-attention stack.

    Subclass to swap in Roformer / Longformer / LLaMA-style attention blocks.
    Any subclass MUST preserve the four-method surface below so that
    :class:`TcrBert` and the Stage-2 pre-training delegates keep working.
    """

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @abstractmethod
    def forward(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run all layers. ``padding_mask`` = ``True`` means ignore that token."""

    @abstractmethod
    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run ``num_layers - 1`` layers (skip the last block). Only meaningful
        when ``num_layers >= 2``; when ``num_layers == 1`` the input is returned."""

    @abstractmethod
    def set_fine_tuning_mode(self, turn_on: bool) -> None:
        """``turn_on=True`` freezes layers ``[0 .. num_layers-2]`` and keeps the
        last layer trainable. ``turn_on=False`` unfreezes everything."""


class TorchTransformerSelfAttentionStack(SelfAttentionStack):
    """Default stack — PyTorch ``TransformerEncoder``."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0; got {d_model}")
        if nhead <= 0 or d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self._d_model = int(d_model)
        self._num_layers = int(num_layers)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        # Name mirrors tcrlm's ``self._self_attention_stack`` so ckpt-adapter
        # renames in step 11 can point directly at ``stack.layers.*``.
        self.stack = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.stack(token_embeddings, src_key_padding_mask=padding_mask)

    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = token_embeddings
        penultimate = self._num_layers - 1
        for layer in self.stack.layers[:penultimate]:
            out = layer(src=out, src_key_padding_mask=padding_mask)
        return out

    def set_fine_tuning_mode(self, turn_on: bool) -> None:
        want_grad = not bool(turn_on)
        penultimate = self._num_layers - 1
        for i, layer in enumerate(self.stack.layers):
            if i < penultimate:
                layer.requires_grad_(want_grad)
            else:
                layer.requires_grad_(True)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _rope_cache(
    *,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim; got {head_dim}")
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        float(base)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.einsum("l,d->ld", pos, inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def _apply_rope(x: torch.Tensor, *, base: float) -> torch.Tensor:
    # x: [B, H, L, Dh]
    cos, sin = _rope_cache(
        seq_len=x.size(-2),
        head_dim=x.size(-1),
        device=x.device,
        dtype=x.dtype,
        base=base,
    )
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class RoFormerEncoderLayer(nn.Module):
    """Transformer encoder layer with rotary position embedding in self-attention."""

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        head_dim = d_model // nhead
        if head_dim % 2 != 0:
            raise ValueError(f"RoFormer head_dim must be even; got {head_dim}")
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = int(head_dim)
        self.rope_base = float(rope_base)

        self.ln_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        h = self.ln_attn(x)
        q = _apply_rope(self._split_heads(self.q_proj(h)), base=self.rope_base)
        k = _apply_rope(self._split_heads(self.k_proj(h)), base=self.rope_base)
        v = self._split_heads(self.v_proj(h))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(
                src_key_padding_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        x = x + self.resid_drop(self.out_proj(self._merge_heads(out)))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class RoFormerSelfAttentionStack(SelfAttentionStack):
    """RoFormer-style TCR stack using RoPE instead of additive position codes in attention."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self._d_model = int(d_model)
        self._num_layers = int(num_layers)
        layers = nn.ModuleList(
            [
                RoFormerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        # Compatibility for existing freeze logic/tests that inspect
        # ``self_attention_stack.stack.layers`` on the default transformer stack.
        self.stack = nn.Module()
        self.stack.layers = layers

    @property
    def layers(self) -> nn.ModuleList:
        return self.stack.layers

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = token_embeddings
        for layer in self.layers:
            out = layer(src=out, src_key_padding_mask=padding_mask)
        return out

    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = token_embeddings
        for layer in self.layers[: self._num_layers - 1]:
            out = layer(src=out, src_key_padding_mask=padding_mask)
        return out

    def set_fine_tuning_mode(self, turn_on: bool) -> None:
        want_grad = not bool(turn_on)
        penultimate = self._num_layers - 1
        for i, layer in enumerate(self.layers):
            if i < penultimate:
                layer.requires_grad_(want_grad)
            else:
                layer.requires_grad_(True)


class ReformerLSHEncoderLayer(nn.Module):
    """Reformer-style encoder layer using LSH-masked self-attention.

    This keeps the same external contract as the local TCR-BERT stack while
    replacing dense self-attention with content buckets generated from random
    signed projections. For the short paired-CDR sequences used here the memory
    gain is modest, but the inductive bias is the relevant ablation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        bucket_size: int,
        num_hashes: int,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be > 0; got {bucket_size}")
        if num_hashes <= 0:
            raise ValueError(f"num_hashes must be > 0; got {num_hashes}")

        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.head_dim = int(d_model // nhead)
        self.bucket_size = int(bucket_size)
        self.num_hashes = int(num_hashes)

        self.ln_attn = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Persisted so Stage-3 uses the same LSH bucket projections learned
        # around during Stage-2 pretraining.
        hash_proj = torch.randn(num_hashes, nhead, self.head_dim) / math.sqrt(self.head_dim)
        self.register_buffer("hash_proj", hash_proj, persistent=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def _lsh_allowed_mask(
        self,
        q: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # q: [B, H, L, Dh]. Output [B, H, L, L], True means attention is allowed.
        q_hash = F.normalize(q.detach().float(), p=2, dim=-1)
        target_buckets = max(2, math.ceil(q.size(-2) / self.bucket_size))
        n_bits = min(self.num_hashes, max(1, math.ceil(math.log2(target_buckets))))
        proj = torch.einsum(
            "bhld,rhd->bhlr",
            q_hash,
            self.hash_proj[:n_bits].to(device=q.device, dtype=torch.float32),
        )
        bits = proj > 0
        powers = (2 ** torch.arange(n_bits, device=q.device)).view(1, 1, 1, -1)
        bucket_ids = (bits.to(torch.long) * powers).sum(dim=-1)
        allowed = bucket_ids.unsqueeze(-1).eq(bucket_ids.unsqueeze(-2))

        eye = torch.eye(q.size(-2), device=q.device, dtype=torch.bool)
        allowed = allowed | eye.view(1, 1, q.size(-2), q.size(-2))

        if padding_mask is not None:
            valid_keys = (~padding_mask).view(padding_mask.size(0), 1, 1, padding_mask.size(1))
            allowed = allowed & valid_keys
            # A fully masked bucket would produce all -inf and NaNs. Fall back
            # to all valid keys only for those rows.
            has_key = allowed.any(dim=-1, keepdim=True)
            allowed = torch.where(has_key, allowed, valid_keys.expand_as(allowed))

        return allowed

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        h = self.ln_attn(x)
        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self._lsh_allowed_mask(q, src_key_padding_mask)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        x = x + self.resid_drop(self.out_proj(self._merge_heads(out)))
        x = x + self.ffn(self.ln_ffn(x))
        return x


class ReformerSelfAttentionStack(SelfAttentionStack):
    """Reformer-style TCR stack with LSH self-attention blocks."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        bucket_size: int = 16,
        num_hashes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}")
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self._d_model = int(d_model)
        self._num_layers = int(num_layers)
        layers = nn.ModuleList(
            [
                ReformerLSHEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    bucket_size=bucket_size,
                    num_hashes=num_hashes,
                )
                for _ in range(num_layers)
            ]
        )
        self.stack = nn.Module()
        self.stack.layers = layers

    @property
    def layers(self) -> nn.ModuleList:
        return self.stack.layers

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = token_embeddings
        for layer in self.layers:
            out = layer(src=out, src_key_padding_mask=padding_mask)
        return out

    def get_token_embeddings_at_penultimate_layer(
        self, token_embeddings: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = token_embeddings
        for layer in self.layers[: self._num_layers - 1]:
            out = layer(src=out, src_key_padding_mask=padding_mask)
        return out

    def set_fine_tuning_mode(self, turn_on: bool) -> None:
        want_grad = not bool(turn_on)
        penultimate = self._num_layers - 1
        for i, layer in enumerate(self.layers):
            if i < penultimate:
                layer.requires_grad_(want_grad)
            else:
                layer.requires_grad_(True)


# ---------------------------------------------------------------------------
# TcrBert.
# ---------------------------------------------------------------------------


class TcrBert(nn.Module):
    """TCR encoder with the two public surfaces the rest of the package relies on.

    Parameters
    ----------
    token_embedder
        A :class:`CdrEmbedder` (or any compatible ``nn.Module`` that accepts
        ``[B, L, 4]`` and returns ``[B, L, D]``).
    self_attention_stack
        Any :class:`SelfAttentionStack` subclass.

    Keeping both arguments injectable makes it trivial to substitute a
    ``RoformerSelfAttentionStack`` later without touching Stage-2 / Stage-3
    code.
    """

    def __init__(
        self,
        token_embedder: nn.Module,
        self_attention_stack: SelfAttentionStack,
    ) -> None:
        super().__init__()
        if not isinstance(self_attention_stack, SelfAttentionStack):
            raise TypeError(
                "self_attention_stack must subclass tcr_tower.SelfAttentionStack; "
                f"got {type(self_attention_stack).__name__}"
            )
        if hasattr(token_embedder, "d_model") and (
            token_embedder.d_model != self_attention_stack.d_model
        ):
            raise ValueError(
                f"d_model mismatch: token_embedder={token_embedder.d_model} vs "
                f"stack={self_attention_stack.d_model}"
            )

        self.token_embedder = token_embedder
        self.self_attention_stack = self_attention_stack
        self._null_aa_id = int(TcrTokenIndex.NULL)

    # ----- Public properties -----

    @property
    def d_model(self) -> int:
        return self.self_attention_stack.d_model

    @property
    def num_layers(self) -> int:
        return self.self_attention_stack.num_layers

    # ----- Primary downstream API (step 12 cross-attention) -----

    def get_token_embeddings_of(self, tcr_tokens: torch.Tensor) -> torch.Tensor:
        """Return contextualised ``[B, Lt, D]`` embeddings.

        No L2 norm. No mean-pool. This is the exact tensor the cross-interaction
        module consumes. Padding positions will still be ignored downstream
        via the ``tcr_pad_mask`` from collate.
        """
        padding_mask = self._get_padding_mask(tcr_tokens)
        raw = self.token_embedder(tcr_tokens)
        return self.self_attention_stack(raw, padding_mask)

    # ----- Optional pre-training API -----

    def get_vector_representations_of(self, tcr_tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[B, D]`` L2-normalised sequence vectors (tcrlm CL-style).

        Runs to the **penultimate** layer of the stack and does masked mean
        pool that excludes both padding **and** the CLS slot, then
        L2-normalises. Kept for Stage-2 contrastive pre-training parity
        with tcrlm-main's ``AveragePoolVectorRepresentationDelegate``.
        """
        padding_mask = self._get_padding_mask(tcr_tokens)
        raw = self.token_embedder(tcr_tokens)
        penult = self.self_attention_stack.get_token_embeddings_at_penultimate_layer(
            raw, padding_mask
        )

        valid_mask = ~padding_mask  # True = keep
        if valid_mask.size(1) > LOCATION_OF_CLS_TOKEN:
            valid_mask = valid_mask.clone()
            valid_mask[:, LOCATION_OF_CLS_TOKEN] = False
        w = valid_mask.unsqueeze(-1).to(penult.dtype)
        denom = w.sum(dim=1).clamp_min(1e-6)
        pooled = (penult * w).sum(dim=1) / denom
        return F.normalize(pooled, p=2, dim=1)

    def get_cls_vector_representations_of(self, tcr_tokens: torch.Tensor) -> torch.Tensor:
        """Return ``[B, D]`` L2-normalised **CLS** embeddings (final layer)."""
        padding_mask = self._get_padding_mask(tcr_tokens)
        raw = self.token_embedder(tcr_tokens)
        hidden = self.self_attention_stack(raw, padding_mask)
        cls_vec = hidden[:, LOCATION_OF_CLS_TOKEN, :]
        return F.normalize(cls_vec, p=2, dim=1)

    # ----- Freeze mode -----

    def set_fine_tuning_mode(self, turn_on: bool) -> None:
        """Delegates to the underlying :class:`SelfAttentionStack`. Does **not**
        freeze the embedder — tcrlm's behaviour."""
        self.self_attention_stack.set_fine_tuning_mode(turn_on)

    # ----- Helpers -----

    def _get_padding_mask(self, tcr_tokens: torch.Tensor) -> torch.Tensor:
        """``tokens[:, :, 0] == NULL`` per DATA_CONTRACT §6."""
        if tcr_tokens.dim() != 3 or tcr_tokens.size(-1) != 4:
            raise ValueError(
                f"tcr_tokens must be [B, L, 4]; got {tuple(tcr_tokens.shape)}"
            )
        return tcr_tokens[..., 0] == self._null_aa_id

    # ----- Construction helpers -----

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        *,
        max_position_len: int = DEFAULT_MAX_POSITION_LEN,
    ) -> "TcrBert":
        """Build the default (TransformerEncoder) TCR tower from a :class:`ModelConfig`."""
        if model_cfg.tcr_stack_type == "sceptr_hybrid":
            from .tcr_sceptr_adapter import SceptrSpecificityTcr
            return SceptrSpecificityTcr(model_cfg)
        embedder = CdrEmbedder(
            d_model=model_cfg.d_model,
            vocab_size=model_cfg.tcr_vocab_size,
            n_compartments=model_cfg.tcr_n_compartments,
            max_position_len=max_position_len,
        )
        if model_cfg.tcr_stack_type == "roformer":
            stack = RoFormerSelfAttentionStack(
                num_layers=model_cfg.tcr_encoder_layers,
                d_model=model_cfg.d_model,
                nhead=model_cfg.tcr_encoder_nhead,
                dim_feedforward=model_cfg.tcr_encoder_ff,
                dropout=model_cfg.tcr_encoder_dropout,
                rope_base=model_cfg.tcr_rope_base,
            )
        elif model_cfg.tcr_stack_type == "reformer":
            stack = ReformerSelfAttentionStack(
                num_layers=model_cfg.tcr_encoder_layers,
                d_model=model_cfg.d_model,
                nhead=model_cfg.tcr_encoder_nhead,
                dim_feedforward=model_cfg.tcr_encoder_ff,
                dropout=model_cfg.tcr_encoder_dropout,
                bucket_size=model_cfg.tcr_reformer_bucket_size,
                num_hashes=model_cfg.tcr_reformer_num_hashes,
            )
        else:
            stack = TorchTransformerSelfAttentionStack(
                num_layers=model_cfg.tcr_encoder_layers,
                d_model=model_cfg.d_model,
                nhead=model_cfg.tcr_encoder_nhead,
                dim_feedforward=model_cfg.tcr_encoder_ff,
                dropout=model_cfg.tcr_encoder_dropout,
            )
        bert = cls(embedder, stack)
        if model_cfg.tcr_finetune_last_only:
            bert.set_fine_tuning_mode(True)
        logger.info(
            "[tcr-tower] built TcrBert (stack=%s, d_model=%d, layers=%d, nhead=%d, ff=%d, fine_tune_last_only=%s)",
            model_cfg.tcr_stack_type,
            model_cfg.d_model,
            model_cfg.tcr_encoder_layers,
            model_cfg.tcr_encoder_nhead,
            model_cfg.tcr_encoder_ff,
            model_cfg.tcr_finetune_last_only,
        )
        return bert
