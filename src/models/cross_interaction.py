"""TCR ↔ pMHC cross-attention interaction layer (step 12).

This is the *core fusion* stage: it consumes the two tower outputs
(``[B, L_tcr, D]`` from :class:`TcrBert.get_token_embeddings_of` and
``[B, L_pmhc, D]`` from the Stage-3 top model, where ``L_pmhc`` is the
concatenation of antigen and MHC residues along the sequence axis) and
produces contextualised versions of both side-by-side so the downstream
pooler + head can see both compartments under a shared attention budget.

Two public classes live here:

- :class:`CrossAttentionBlock`
    Single cross-attention + FFN block with pre-LN and residual. Deliberately
    a clone of the private ``_CrossAttention`` inside :mod:`.pmhc_tower` — the
    two are kept separate so future experiments can swap one without
    touching the other.

- :class:`CrossInteractionStack`
    N-layer wrapper driving three modes:

    ``tcr_to_pmhc``
        per layer: ``tcr ← CA(q=tcr, kv=pmhc, kv_pad=pmhc_pad)``
    ``pmhc_to_tcr``
        per layer: ``pmhc ← CA(q=pmhc, kv=tcr, kv_pad=tcr_pad)``
    ``bidirectional``
        per layer: first update ``tcr`` against the current ``pmhc``; then
        update ``pmhc`` against the **refreshed** ``tcr``. Two independent
        blocks per layer (separate weights) — keeps the TCR representation
        from being a blind spot of the pMHC update.

Chain-bias hook (``use_chain_bias=True``)
    Adds a learnable table ``Embedding(n_pmhc_chains * n_tcr_chains,
    n_heads)`` to the ``Q K^T`` logits of every cross-attention block.
    Indexed by the (pmhc_compartment, tcr_compartment) pair at each
    (query, key) position; the compartment ids must be passed via
    ``forward``. Initialised to zero so the initial behaviour with the
    flag on is bit-identical to the non-biased variant.

    pMHC compartment ids follow :class:`PMHCChainIndex`
    (``NULL / ANT / MHC``). TCR compartment ids follow
    :class:`~src.data.tokenizers.TcrCompartmentIndex`.

    Defaults to off per the step-12 spec; enable explicitly in the YAML
    config (``use_chain_bias: true``) to try it.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "CrossAttentionBlock",
    "CrossInteractionStack",
    "PMHCChainIndex",
    "ALLOWED_CROSS_MODES",
    "DEFAULT_N_TCR_CHAINS",
    "DEFAULT_N_PMHC_CHAINS",
]


ALLOWED_CROSS_MODES: Tuple[str, ...] = ("tcr_to_pmhc", "pmhc_to_tcr", "bidirectional")

DEFAULT_N_TCR_CHAINS: int = 8
"""NULL + CLS + CDR{1,2,3}{a,b}; matches :class:`TcrCompartmentIndex`."""

DEFAULT_N_PMHC_CHAINS: int = 3
"""NULL / ANT / MHC — see :class:`PMHCChainIndex`."""


class PMHCChainIndex(IntEnum):
    """Compartment ids for the pMHC side of the cross-attention.

    The top-level Stage-3 model concatenates antigen and MHC tokens along
    the sequence axis (see step 13). For each resulting position we label
    which chain it belongs to; the chain-bias table uses these labels to
    pick a learnable per-head scalar added to the attention logits.

    ``NULL`` is reserved for pad positions; the same positions are already
    ignored by ``pmhc_pad_mask`` so the bias on them is irrelevant, but
    having an explicit slot keeps the lookup out-of-bound-safe.
    """

    NULL = 0
    ANT = 1
    MHC = 2


# ---------------------------------------------------------------------------
# Cross-attention block.
# ---------------------------------------------------------------------------


class CrossAttentionBlock(nn.Module):
    """Single pre-LN cross-attention + FFN block with residual connections.

    Structure::

        q' = q  + Dropout(MHA(LN(q), LN(kv), LN(kv), key_padding_mask=kv_pad))
        y  = q' + FFN(LN(q'))

    ``attn_bias`` (optional) is forwarded to :class:`nn.MultiheadAttention`
    as ``attn_mask`` (float additive) — callers use this to inject the
    chain-bias lookup from :class:`CrossInteractionStack`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0; got {d_model}")
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {dropout}")
        if ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be > 0; got {ffn_mult}")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.dropout_p = float(dropout)
        self.ffn_mult = int(ffn_mult)

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        *,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one cross-attention + FFN pass.

        Parameters
        ----------
        q
            ``[B, L_q, D]`` query token representations.
        kv
            ``[B, L_kv, D]`` key/value token representations.
        kv_key_padding_mask
            Optional ``[B, L_kv]`` BoolTensor; ``True`` = ignore this key.
        attn_bias
            Optional ``[B * n_heads, L_q, L_kv]`` FloatTensor added to the
            attention logits before softmax. Intended for the chain-bias
            table; leave ``None`` for vanilla cross-attention.
        return_weights
            If ``True``, also return the (head-averaged) attention weights
            ``[B, L_q, L_kv]``. Uses ``average_attn_weights=True`` on MHA
            which is the canonical visualisation path.

        Returns
        -------
        Tensor
            ``[B, L_q, D]`` when ``return_weights=False`` (default).
        Tuple[Tensor, Tensor]
            ``(out, weights)`` with ``weights`` of shape ``[B, L_q, L_kv]``
            when ``return_weights=True``.
        """
        if q.dim() != 3 or q.size(-1) != self.d_model:
            raise ValueError(
                f"q must be [B, L, {self.d_model}]; got {tuple(q.shape)}"
            )
        if kv.dim() != 3 or kv.size(-1) != self.d_model:
            raise ValueError(
                f"kv must be [B, L, {self.d_model}]; got {tuple(kv.shape)}"
            )

        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)

        # PyTorch ≥ 2.x warns when ``attn_mask`` is float and
        # ``key_padding_mask`` is bool and both are supplied. Fold the pad
        # mask into the float bias so we only pass one mask to MHA.
        effective_attn_mask = attn_bias
        effective_kpm = kv_key_padding_mask
        if attn_bias is not None and kv_key_padding_mask is not None:
            B, Lk = kv.size(0), kv.size(1)
            Lq = q.size(1)
            pad_additive = torch.zeros(
                B, Lk, dtype=attn_bias.dtype, device=attn_bias.device,
            )
            pad_additive = pad_additive.masked_fill(kv_key_padding_mask, float("-inf"))
            # Broadcast to [B, n_heads, Lq, Lk] then fold into [B*n_heads, Lq, Lk].
            pad_additive = (
                pad_additive
                .unsqueeze(1)  # [B, 1, Lk]
                .unsqueeze(1)  # [B, 1, 1, Lk]
                .expand(B, self.n_heads, Lq, Lk)
                .reshape(B * self.n_heads, Lq, Lk)
            )
            effective_attn_mask = attn_bias + pad_additive
            effective_kpm = None

        attn_out, attn_w = self.attn(
            qn, kvn, kvn,
            key_padding_mask=effective_kpm,
            attn_mask=effective_attn_mask,
            need_weights=return_weights,
            average_attn_weights=True,
        )
        x = q + self.drop_attn(attn_out)
        x = x + self.ffn(self.ln_ffn(x))
        if return_weights:
            return x, attn_w
        return x


# ---------------------------------------------------------------------------
# Cross-interaction stack.
# ---------------------------------------------------------------------------


class CrossInteractionStack(nn.Module):
    """N-layer TCR↔pMHC cross-attention stack.

    Parameters
    ----------
    n_layers
        Number of stacked layers (``>= 0``; zero is allowed for ablations
        and results in an identity forward).
    mode
        One of ``"tcr_to_pmhc" / "pmhc_to_tcr" / "bidirectional"``.
        In ``bidirectional`` mode the TCR update happens first and feeds the
        pMHC update of the same layer, so TCR context always flows forward.
    d_model, n_heads, dropout, ffn_mult
        Forwarded to every :class:`CrossAttentionBlock`.
    use_chain_bias
        If ``True``, adds a learnable
        ``Embedding(n_pmhc_chains * n_tcr_chains, n_heads)`` bias to every
        block's QK^T. Required inputs at forward time become
        ``tcr_chain_ids`` + ``pmhc_chain_ids``.
    n_pmhc_chains, n_tcr_chains
        Sizes of the chain-bias table along each axis. The defaults match
        :class:`PMHCChainIndex` (3) and
        :class:`~src.data.tokenizers.TcrCompartmentIndex` (8)
        respectively.
    """

    def __init__(
        self,
        n_layers: int,
        mode: str,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.1,
        ffn_mult: int = 4,
        use_chain_bias: bool = False,
        n_pmhc_chains: int = DEFAULT_N_PMHC_CHAINS,
        n_tcr_chains: int = DEFAULT_N_TCR_CHAINS,
    ) -> None:
        super().__init__()
        if n_layers < 0:
            raise ValueError(f"n_layers must be >= 0; got {n_layers}")
        if mode not in ALLOWED_CROSS_MODES:
            raise ValueError(f"mode must be one of {ALLOWED_CROSS_MODES}; got {mode!r}")
        if n_pmhc_chains < 1 or n_tcr_chains < 1:
            raise ValueError(
                f"chain-bias table must have positive sides; got "
                f"n_pmhc={n_pmhc_chains}, n_tcr={n_tcr_chains}"
            )

        self.n_layers = int(n_layers)
        self.mode = mode
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.ffn_mult = int(ffn_mult)
        self.use_chain_bias = bool(use_chain_bias)
        self.n_pmhc_chains = int(n_pmhc_chains)
        self.n_tcr_chains = int(n_tcr_chains)

        need_tcr_blocks = mode in ("tcr_to_pmhc", "bidirectional")
        need_pmhc_blocks = mode in ("pmhc_to_tcr", "bidirectional")

        self.tcr_blocks: nn.ModuleList = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, n_heads, dropout, ffn_mult)
                for _ in range(n_layers)
            ]
            if need_tcr_blocks
            else []
        )
        self.pmhc_blocks: nn.ModuleList = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, n_heads, dropout, ffn_mult)
                for _ in range(n_layers)
            ]
            if need_pmhc_blocks
            else []
        )

        if self.use_chain_bias:
            self.chain_bias = nn.Embedding(
                num_embeddings=self.n_pmhc_chains * self.n_tcr_chains,
                embedding_dim=n_heads,
            )
            # Init at zero so enabling the flag starts as a no-op.
            nn.init.zeros_(self.chain_bias.weight)
        else:
            self.chain_bias = None

    # ----- Construction helpers -----

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        *,
        n_pmhc_chains: int = DEFAULT_N_PMHC_CHAINS,
        n_tcr_chains: Optional[int] = None,
    ) -> "CrossInteractionStack":
        """Build directly from a :class:`ModelConfig`."""
        if n_tcr_chains is None:
            n_tcr_chains = int(model_cfg.tcr_n_compartments)
        return cls(
            n_layers=model_cfg.cross_layers,
            mode=model_cfg.cross_mode,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            dropout=model_cfg.dropout,
            ffn_mult=model_cfg.cross_ffn_mult,
            use_chain_bias=model_cfg.use_chain_bias,
            n_pmhc_chains=n_pmhc_chains,
            n_tcr_chains=n_tcr_chains,
        )

    # ----- Forward -----

    def forward(
        self,
        tcr: torch.Tensor,
        pmhc: torch.Tensor,
        *,
        tcr_pad_mask: Optional[torch.Tensor] = None,
        pmhc_pad_mask: Optional[torch.Tensor] = None,
        tcr_chain_ids: Optional[torch.Tensor] = None,
        pmhc_chain_ids: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, List[torch.Tensor]]],
    ]:
        """Run the stack.

        Parameters
        ----------
        tcr, pmhc
            Token-level reps ``[B, L_tcr, D]`` / ``[B, L_pmhc, D]`` from the
            TCR tower and the (antigen ⊕ MHC)-concatenated pMHC tower.
        tcr_pad_mask, pmhc_pad_mask
            Optional ``[B, L]`` BoolTensors (``True`` = ignore). Passed as
            the ``kv_key_padding_mask`` of whichever block uses that side
            as K/V.
        tcr_chain_ids, pmhc_chain_ids
            Required when ``use_chain_bias=True``. ``[B, L]`` LongTensors
            with entries in ``[0, n_tcr_chains)`` / ``[0, n_pmhc_chains)``.
        return_weights
            If ``True``, additionally return a dict of per-layer attention
            weights (see below). Head-averaged via
            ``average_attn_weights=True`` on MHA.

        Returns
        -------
        Tuple[Tensor, Tensor]
            ``(tcr_out, pmhc_out)`` when ``return_weights=False`` (default).
            Branches that the mode does not update are returned
            **unchanged** so callers downstream (e.g. the fusion head in
            step 13) get the freshest view of both sides regardless of mode.
        Tuple[Tensor, Tensor, Dict[str, List[Tensor]]]
            When ``return_weights=True``. The dict has keys
            ``"tcr_from_pmhc"`` (list of ``[B, L_tcr, L_pmhc]`` per TCR
            update) and ``"pmhc_from_tcr"`` (list of ``[B, L_pmhc, L_tcr]``
            per pMHC update). Lists may be empty when the mode does not
            exercise that direction. ``n_layers=0`` → both lists empty.
        """
        self._validate_inputs(
            tcr, pmhc, tcr_pad_mask, pmhc_pad_mask, tcr_chain_ids, pmhc_chain_ids,
        )
        aux_weights: Dict[str, List[torch.Tensor]] = {
            "tcr_from_pmhc": [],
            "pmhc_from_tcr": [],
        }

        if self.n_layers == 0:
            if return_weights:
                return tcr, pmhc, aux_weights
            return tcr, pmhc

        for layer_idx in range(self.n_layers):
            if self.mode == "tcr_to_pmhc":
                tcr, w = self._run_tcr_update(
                    layer_idx, tcr, pmhc, pmhc_pad_mask, tcr_chain_ids, pmhc_chain_ids,
                    return_weights=return_weights,
                )
                if return_weights and w is not None:
                    aux_weights["tcr_from_pmhc"].append(w)
            elif self.mode == "pmhc_to_tcr":
                pmhc, w = self._run_pmhc_update(
                    layer_idx, tcr, pmhc, tcr_pad_mask, tcr_chain_ids, pmhc_chain_ids,
                    return_weights=return_weights,
                )
                if return_weights and w is not None:
                    aux_weights["pmhc_from_tcr"].append(w)
            else:  # bidirectional
                tcr, w_tcr = self._run_tcr_update(
                    layer_idx, tcr, pmhc, pmhc_pad_mask, tcr_chain_ids, pmhc_chain_ids,
                    return_weights=return_weights,
                )
                pmhc, w_pmhc = self._run_pmhc_update(
                    layer_idx, tcr, pmhc, tcr_pad_mask, tcr_chain_ids, pmhc_chain_ids,
                    return_weights=return_weights,
                )
                if return_weights:
                    if w_tcr is not None:
                        aux_weights["tcr_from_pmhc"].append(w_tcr)
                    if w_pmhc is not None:
                        aux_weights["pmhc_from_tcr"].append(w_pmhc)

        if return_weights:
            return tcr, pmhc, aux_weights
        return tcr, pmhc

    # ----- Per-direction dispatch -----

    def _run_tcr_update(
        self,
        layer_idx: int,
        tcr: torch.Tensor,
        pmhc: torch.Tensor,
        pmhc_pad_mask: Optional[torch.Tensor],
        tcr_chain_ids: Optional[torch.Tensor],
        pmhc_chain_ids: Optional[torch.Tensor],
        *,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias = self._chain_bias_for(
            direction="tcr_as_q",
            n_q=tcr.size(1),
            n_k=pmhc.size(1),
            dtype=tcr.dtype,
            device=tcr.device,
            tcr_chain_ids=tcr_chain_ids,
            pmhc_chain_ids=pmhc_chain_ids,
        )
        result = self.tcr_blocks[layer_idx](
            q=tcr, kv=pmhc,
            kv_key_padding_mask=pmhc_pad_mask,
            attn_bias=bias,
            return_weights=return_weights,
        )
        if return_weights:
            out, w = result  # type: ignore[misc]
            return out, w
        return result, None  # type: ignore[return-value]

    def _run_pmhc_update(
        self,
        layer_idx: int,
        tcr: torch.Tensor,
        pmhc: torch.Tensor,
        tcr_pad_mask: Optional[torch.Tensor],
        tcr_chain_ids: Optional[torch.Tensor],
        pmhc_chain_ids: Optional[torch.Tensor],
        *,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bias = self._chain_bias_for(
            direction="pmhc_as_q",
            n_q=pmhc.size(1),
            n_k=tcr.size(1),
            dtype=pmhc.dtype,
            device=pmhc.device,
            tcr_chain_ids=tcr_chain_ids,
            pmhc_chain_ids=pmhc_chain_ids,
        )
        result = self.pmhc_blocks[layer_idx](
            q=pmhc, kv=tcr,
            kv_key_padding_mask=tcr_pad_mask,
            attn_bias=bias,
            return_weights=return_weights,
        )
        if return_weights:
            out, w = result  # type: ignore[misc]
            return out, w
        return result, None  # type: ignore[return-value]

    # ----- Chain-bias lookup -----

    def _chain_bias_for(
        self,
        *,
        direction: str,
        n_q: int,
        n_k: int,
        dtype: torch.dtype,
        device: torch.device,
        tcr_chain_ids: Optional[torch.Tensor],
        pmhc_chain_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_chain_bias:
            return None
        assert tcr_chain_ids is not None and pmhc_chain_ids is not None

        if direction == "tcr_as_q":
            # Q = tcr [B, L_q], K = pmhc [B, L_k]
            # idx[b, lq, lk] = pmhc_cid[b, lk] * n_tcr + tcr_cid[b, lq]
            idx = (
                pmhc_chain_ids.unsqueeze(1) * self.n_tcr_chains
                + tcr_chain_ids.unsqueeze(2)
            )
        elif direction == "pmhc_as_q":
            # Q = pmhc [B, L_q], K = tcr [B, L_k]
            # idx[b, lq, lk] = pmhc_cid[b, lq] * n_tcr + tcr_cid[b, lk]
            idx = (
                pmhc_chain_ids.unsqueeze(2) * self.n_tcr_chains
                + tcr_chain_ids.unsqueeze(1)
            )
        else:
            raise ValueError(f"unknown direction {direction!r}")

        # idx: [B, L_q, L_k] -> embedding returns [B, L_q, L_k, n_heads].
        bias = self.chain_bias(idx)
        B = bias.size(0)
        # MHA accepts attn_mask of shape [B * n_heads, L_q, L_k] when batch-wise.
        bias = bias.permute(0, 3, 1, 2).contiguous()
        bias = bias.reshape(B * self.n_heads, n_q, n_k)
        return bias.to(dtype=dtype, device=device)

    # ----- Validation -----

    def _validate_inputs(
        self,
        tcr: torch.Tensor,
        pmhc: torch.Tensor,
        tcr_pad_mask: Optional[torch.Tensor],
        pmhc_pad_mask: Optional[torch.Tensor],
        tcr_chain_ids: Optional[torch.Tensor],
        pmhc_chain_ids: Optional[torch.Tensor],
    ) -> None:
        if tcr.dim() != 3 or tcr.size(-1) != self.d_model:
            raise ValueError(f"tcr must be [B, L, {self.d_model}]; got {tuple(tcr.shape)}")
        if pmhc.dim() != 3 or pmhc.size(-1) != self.d_model:
            raise ValueError(f"pmhc must be [B, L, {self.d_model}]; got {tuple(pmhc.shape)}")
        if tcr.size(0) != pmhc.size(0):
            raise ValueError(
                f"tcr / pmhc batch size mismatch: {tcr.size(0)} vs {pmhc.size(0)}"
            )
        if tcr_pad_mask is not None and tcr_pad_mask.shape != tcr.shape[:2]:
            raise ValueError(
                f"tcr_pad_mask shape {tuple(tcr_pad_mask.shape)} must match tcr prefix "
                f"{tuple(tcr.shape[:2])}"
            )
        if pmhc_pad_mask is not None and pmhc_pad_mask.shape != pmhc.shape[:2]:
            raise ValueError(
                f"pmhc_pad_mask shape {tuple(pmhc_pad_mask.shape)} must match pmhc prefix "
                f"{tuple(pmhc.shape[:2])}"
            )
        if self.use_chain_bias:
            if tcr_chain_ids is None or pmhc_chain_ids is None:
                raise ValueError(
                    "use_chain_bias=True requires both tcr_chain_ids and pmhc_chain_ids at forward()"
                )
            if tcr_chain_ids.shape != tcr.shape[:2]:
                raise ValueError(
                    f"tcr_chain_ids shape {tuple(tcr_chain_ids.shape)} must match tcr prefix"
                )
            if pmhc_chain_ids.shape != pmhc.shape[:2]:
                raise ValueError(
                    f"pmhc_chain_ids shape {tuple(pmhc_chain_ids.shape)} must match pmhc prefix"
                )
            # Out-of-range id would silently wrap via embedding lookup, so we assert.
            if tcr_chain_ids.numel() > 0 and (
                int(tcr_chain_ids.max()) >= self.n_tcr_chains
                or int(tcr_chain_ids.min()) < 0
            ):
                raise ValueError(
                    f"tcr_chain_ids must live in [0, {self.n_tcr_chains}); "
                    f"got range [{int(tcr_chain_ids.min())}, {int(tcr_chain_ids.max())}]"
                )
            if pmhc_chain_ids.numel() > 0 and (
                int(pmhc_chain_ids.max()) >= self.n_pmhc_chains
                or int(pmhc_chain_ids.min()) < 0
            ):
                raise ValueError(
                    f"pmhc_chain_ids must live in [0, {self.n_pmhc_chains}); "
                    f"got range [{int(pmhc_chain_ids.min())}, {int(pmhc_chain_ids.max())}]"
                )

    # ----- Diagnostics -----

    @property
    def trainable_param_counts(self) -> dict:
        """Count trainable parameters split by block ownership.

        Useful for logging ``pmhc_cross / tcr_cross / chain_bias`` sizes so
        the operator can sanity-check the stack at construction time.
        """
        def _count(ms: List[nn.Module]) -> int:
            return sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in ms)

        out = {
            "tcr_cross_blocks": _count(list(self.tcr_blocks)),
            "pmhc_cross_blocks": _count(list(self.pmhc_blocks)),
            "chain_bias": sum(
                p.numel() for p in (self.chain_bias.parameters() if self.chain_bias is not None else [])
                if p.requires_grad
            ),
        }
        out["total"] = sum(out.values())
        return out
