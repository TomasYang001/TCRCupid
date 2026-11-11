"""``TCREpitopeSpecificityModel`` — 顶层拼装 (pMHC 塔 + TCR 塔 + CrossInteraction + Head).

Step 13 的交付物。forward 签名严格按照 prompt 锁定（见下文 :meth:`TCREpitopeSpecificityModel.forward`）。

Pipeline
--------
1. :class:`PMHCTower` 产出 ``ant_repr [B, La, D]`` 与 ``mhc_repr [B, Lm, D]``。
2. 方案 A：沿序列轴 concat →
   ``pmhc_tokens_repr [B, La + Lm, D]``、``pmhc_pad_mask``、``pmhc_valid_mask``。
3. :meth:`TcrBert.get_token_embeddings_of` → ``tcr_tokens_repr [B, Lt, D]``。
4. :class:`CrossInteractionStack` → ``(tcr_joint, pmhc_joint)``
   （``use_chain_bias=True`` 时自动构造 compartment ids：pMHC 用
   :class:`PMHCChainIndex`，TCR 用 tokenizer 第 3 维 compartment id）。
5. 分别 masked mean pool（**不做** L2 归一）→ ``pmhc_vec [B, D]`` / ``tcr_vec [B, D]``。
6. :meth:`_fuse` 按 ``fuse_mode`` 组合：

   - ``concat_diff_mul`` (默认) → ``[p, t, |p-t|, p*t] ∈ [B, 4D]``
   - ``concat``                → ``[p, t] ∈ [B, 2D]``
   - ``bilinear``              → ``Bilinear(D, D, D)(p, t) ∈ [B, D]``
7. :class:`MLPClassifier`（``hidden_dims = head_hidden_dims``, ``out_dim=1``） →
   ``logits [B]``。

Aux 输出 (``return_aux=True``)::

    {
        "pmhc_vec":       Tensor [B, D],
        "tcr_vec":        Tensor [B, D],
        "attn_weights":   {"tcr_from_pmhc": [w, ...], "pmhc_from_tcr": [w, ...]}
    }

``attn_weights`` 直接来自 :meth:`CrossInteractionStack.forward(return_weights=True)`
并做了 head-average（shape ``[B, Lq, Lk]``）。列表长度 = ``cross_layers``，
若某方向在当前 ``cross_mode`` 下不被触发则为空列表。

冻结约定
--------
以下 flag 在 ``__init__`` 末尾统一应用（pMHC backbone 冻结由 :class:`PMHCTower`
自行负责，这里 **不重复**）：

- ``freeze_tcr_embedder``         → CdrEmbedder 全部 ``requires_grad=False``
- ``tcr_finetune_last_only``      → 冻 TCR stack 的 ``0..N-2`` 层，保留最后一层可训
- ``freeze_tcr_encoder_layers=K`` → 冻 TCR stack 的前 ``K`` 层（与 last_only 二选一，
  若都设置则 ``last_only`` 优先——警告之）

:meth:`warmup_unfreeze(epoch)` 按 ``ModelConfig.warmup_unfreeze_schedule`` 渐进放开：
``{epoch: [action, ...]}``，action ∈ ``{tcr_encoder, tcr_embedder, antigen_backbone,
mhc_backbone, all}``。每个 (epoch, action) 只生效一次。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data.tokenizers import TcrCompartmentIndex
from .cross_interaction import CrossInteractionStack, PMHCChainIndex
from .heads import GatedAttentionPool, MLPClassifier, masked_mean_pool
from .pmhc_tower import PMHCTower
from .tcr_tower import TcrBert

logger = logging.getLogger(__name__)

__all__ = [
    "TCREpitopeSpecificityModel",
    "VALID_UNFREEZE_ACTIONS",
]


VALID_UNFREEZE_ACTIONS: frozenset = frozenset(
    {"tcr_encoder", "tcr_embedder", "antigen_backbone", "mhc_backbone", "all"}
)


class TCREpitopeSpecificityModel(nn.Module):
    """Stage-3 顶层模型：pMHC 塔 + TCR 塔 + CrossInteraction + Fusion + Head.

    Parameters
    ----------
    model_cfg
        :class:`ModelConfig` —— 所有维度、冻结、warmup schedule 的唯一来源。
    pmhc_tower
        预先构造的 :class:`PMHCTower` 实例（注入式，方便测试中传 stub backbone）。
    tcr_bert
        预先构造的 :class:`TcrBert` 实例。

    Notes
    -----
    - ``pmhc_tower.d_model`` / ``tcr_bert.d_model`` 必须与 ``model_cfg.d_model``
      相等；不一致直接报错，避免后面 cross-attn 里 shape 不匹配。
    - 顶层 **不** 强制 ESM-C 依赖；调用 :meth:`from_config` 才触发 ``esm`` 加载。
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        *,
        pmhc_tower: PMHCTower,
        tcr_bert: TcrBert,
    ) -> None:
        super().__init__()
        if not isinstance(pmhc_tower, PMHCTower):
            raise TypeError(
                f"pmhc_tower must be PMHCTower; got {type(pmhc_tower).__name__}"
            )
        if not isinstance(tcr_bert, TcrBert):
            raise TypeError(
                f"tcr_bert must be TcrBert; got {type(tcr_bert).__name__}"
            )
        D = int(model_cfg.d_model)
        if pmhc_tower.d_model != D:
            raise ValueError(
                f"pmhc_tower.d_model ({pmhc_tower.d_model}) must match "
                f"ModelConfig.d_model ({D})"
            )
        if tcr_bert.d_model != D:
            raise ValueError(
                f"tcr_bert.d_model ({tcr_bert.d_model}) must match "
                f"ModelConfig.d_model ({D})"
            )

        self.cfg = model_cfg
        self.d_model = D

        self.pmhc_tower = pmhc_tower
        self.tcr_bert = tcr_bert
        self.cross_interaction = CrossInteractionStack.from_config(model_cfg)
        self.pmhc_chain_embedding: Optional[nn.Embedding]
        if bool(getattr(model_cfg, "use_pmhc_chain_embedding", False)):
            self.pmhc_chain_embedding = nn.Embedding(
                num_embeddings=len(PMHCChainIndex),
                embedding_dim=D,
                padding_idx=int(PMHCChainIndex.NULL),
            )
            nn.init.normal_(self.pmhc_chain_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.pmhc_chain_embedding.weight[int(PMHCChainIndex.NULL)].zero_()
        else:
            self.pmhc_chain_embedding = None

        # ---- Pooling -----------------------------------------------------
        pool_mode = model_cfg.pool_mode
        if pool_mode in ("attention", "mean_attention"):
            self.pmhc_attn_pool: Optional[GatedAttentionPool] = GatedAttentionPool(
                D, hidden_dim=D, dropout=model_cfg.dropout,
            )
            self.tcr_attn_pool: Optional[GatedAttentionPool] = GatedAttentionPool(
                D, hidden_dim=D, dropout=model_cfg.dropout,
            )
        else:
            self.pmhc_attn_pool = None
            self.tcr_attn_pool = None
        self.side_dim = 2 * D if pool_mode == "mean_attention" else D

        # ---- Fusion ------------------------------------------------------
        fuse_mode = model_cfg.fuse_mode
        if fuse_mode == "concat_diff_mul":
            self.fuse_in_dim = 4 * self.side_dim
            self._bilinear: Optional[nn.Bilinear] = None
            self._lowrank_p: Optional[nn.Module] = None
            self._lowrank_t: Optional[nn.Module] = None
            self._gated_fuse: Optional[nn.Module] = None
            self._pair_p: Optional[nn.Module] = None
            self._pair_t: Optional[nn.Module] = None
            self._pair_cnn: Optional[nn.Module] = None
        elif fuse_mode == "concat":
            self.fuse_in_dim = 2 * self.side_dim
            self._bilinear = None
            self._lowrank_p = None
            self._lowrank_t = None
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "bilinear":
            # 按 prompt 要求 head_hidden_dims 默认 ``[2D, D]`` 起步，
            # bilinear 输出 ``D`` 可以直接喂进 head 的 pre-norm。
            self.fuse_in_dim = self.side_dim
            self._bilinear = nn.Bilinear(self.side_dim, self.side_dim, self.side_dim, bias=True)
            self._lowrank_p = None
            self._lowrank_t = None
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "concat_diff_mul_lowrank":
            rank = int(model_cfg.fusion_lowrank_dim)
            self.fuse_in_dim = 4 * self.side_dim + rank
            self._bilinear = None
            self._lowrank_p = nn.Sequential(
                nn.LayerNorm(self.side_dim),
                nn.Linear(self.side_dim, rank, bias=False),
            )
            self._lowrank_t = nn.Sequential(
                nn.LayerNorm(self.side_dim),
                nn.Linear(self.side_dim, rank, bias=False),
            )
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "gated_concat_diff_mul":
            self.fuse_in_dim = 5 * self.side_dim
            self._bilinear = None
            self._lowrank_p = None
            self._lowrank_t = None
            self._gated_fuse = nn.Sequential(
                nn.LayerNorm(4 * self.side_dim),
                nn.Linear(4 * self.side_dim, self.side_dim),
                nn.Sigmoid(),
            )
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "pair_map_cnn":
            rank = int(model_cfg.fusion_lowrank_dim)
            self.fuse_in_dim = 5 * self.side_dim
            self._bilinear = None
            self._lowrank_p = None
            self._lowrank_t = None
            self._gated_fuse = None
            self._pair_p = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, rank, bias=False),
            )
            self._pair_t = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, rank, bias=False),
            )
            self._pair_cnn = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveMaxPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, self.side_dim),
                nn.GELU(),
                nn.LayerNorm(self.side_dim),
            )
        elif fuse_mode == "focused_concat_diff_mul":
            self.fuse_in_dim = 8 * self.side_dim
            self._bilinear = None
            self._lowrank_p = None
            self._lowrank_t = None
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "focused_concat_diff_mul_lowrank":
            rank = int(model_cfg.fusion_lowrank_dim)
            self.fuse_in_dim = 8 * self.side_dim + rank
            self._bilinear = None
            self._lowrank_p = nn.Sequential(
                nn.LayerNorm(self.side_dim),
                nn.Linear(self.side_dim, rank, bias=False),
            )
            self._lowrank_t = nn.Sequential(
                nn.LayerNorm(self.side_dim),
                nn.Linear(self.side_dim, rank, bias=False),
            )
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "contact_concat_diff_mul":
            self.fuse_in_dim = 4 * self.side_dim
            self._bilinear = None
            self._lowrank_p = None
            self._lowrank_t = None
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        elif fuse_mode == "contact_concat_diff_mul_lowrank":
            rank = int(model_cfg.fusion_lowrank_dim)
            self.fuse_in_dim = 4 * self.side_dim + rank
            self._bilinear = None
            self._lowrank_p = nn.Sequential(
                nn.LayerNorm(self.side_dim),
                nn.Linear(self.side_dim, rank, bias=False),
            )
            self._lowrank_t = nn.Sequential(
                nn.LayerNorm(self.side_dim),
                nn.Linear(self.side_dim, rank, bias=False),
            )
            self._gated_fuse = None
            self._pair_p = None
            self._pair_t = None
            self._pair_cnn = None
        else:  # pragma: no cover - guarded by ModelConfig
            raise ValueError(f"unsupported fuse_mode: {fuse_mode!r}")

        # head_hidden_dims 已在 config 里校验；必要时可以传空 tuple 关掉隐藏层。
        head_hidden_dims = tuple(model_cfg.head_hidden_dims)
        self.head = MLPClassifier(
            in_dim=self.fuse_in_dim,
            hidden_dims=head_hidden_dims,
            out_dim=1,
            dropout=model_cfg.head_dropout,
            activation="gelu",
            pre_norm=True,
        )

        # 已执行过的 (epoch, action) 集合 —— 保证 warmup_unfreeze 幂等。
        self._applied_unfreeze: set[Tuple[int, str]] = set()

        self._apply_initial_freeze()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        *,
        ant_backbone: nn.Module,
        mhc_backbone: nn.Module,
        hidden_dim_override: Optional[Tuple[int, int]] = None,
    ) -> "TCREpitopeSpecificityModel":
        """Build with injected pMHC backbones (skips ``import esm`` in tests).

        Actual ESM-C loading happens via :meth:`PMHCTower.from_config`; this
        factory stays dependency-free so stubs can be substituted.
        """
        pmhc_tower = PMHCTower(
            model_cfg,
            ant_backbone=ant_backbone,
            mhc_backbone=mhc_backbone,
            hidden_dim_override=hidden_dim_override,
        )
        tcr_bert = TcrBert.from_config(model_cfg)
        return cls(model_cfg, pmhc_tower=pmhc_tower, tcr_bert=tcr_bert)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        *,
        antigen_tokens: torch.Tensor,
        mhc_tokens: torch.Tensor,
        tcr_tokens: torch.Tensor,
        antigen_pad_mask: torch.Tensor,
        mhc_pad_mask: torch.Tensor,
        tcr_pad_mask: torch.Tensor,
        antigen_valid_mask: torch.Tensor,
        mhc_valid_mask: torch.Tensor,
        tcr_valid_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Run the full Stage-3 specificity pipeline.

        Parameters (all keyword-only)
        -----------------------------
        antigen_tokens, mhc_tokens
            ESM-C token ids from :class:`PMHCESMTokenizer`.
        tcr_tokens
            ``[B, Lt, 4]`` tensor from :class:`CdrTcrTokenizer`.
        *_pad_mask
            ``True = ignore``. Used as ``kv_key_padding_mask`` inside MHA.
        *_valid_mask
            ``True = valid (non-special)``. Used for masked mean pool only.
        return_aux
            If ``True`` additionally return a dict with ``pmhc_vec`` / ``tcr_vec``
            and per-layer attention weights from the cross-interaction stack.

        Returns
        -------
        Tensor
            ``logits [B]`` (``out_dim=1`` squeezed).
        Tuple[Tensor, Dict]
            ``(logits, aux)`` when ``return_aux=True``.
        """
        # 1. pMHC tower ---------------------------------------------------
        ant_repr, mhc_repr = self.pmhc_tower(
            antigen_tokens=antigen_tokens,
            mhc_tokens=mhc_tokens,
            antigen_pad_mask=antigen_pad_mask,
            mhc_pad_mask=mhc_pad_mask,
        )
        self._check_pair("antigen", ant_repr, antigen_pad_mask, antigen_valid_mask)
        self._check_pair("mhc", mhc_repr, mhc_pad_mask, mhc_valid_mask)

        # 2. Concat along seq axis (方案 A) -------------------------------
        pmhc_repr = torch.cat([ant_repr, mhc_repr], dim=1)                  # [B, La+Lm, D]
        pmhc_pad_mask = torch.cat([antigen_pad_mask, mhc_pad_mask], dim=1)  # [B, La+Lm]
        pmhc_valid_mask = torch.cat(
            [antigen_valid_mask, mhc_valid_mask], dim=1
        )                                                                   # [B, La+Lm]

        # 3. TCR tower ----------------------------------------------------
        tcr_repr = self.tcr_bert.get_token_embeddings_of(tcr_tokens)        # [B, Lt, D]
        self._check_pair("tcr", tcr_repr, tcr_pad_mask, tcr_valid_mask)

        # 4. Chain bias ids (仅当 use_chain_bias=True) ---------------------
        tcr_chain_ids, pmhc_chain_ids = self._build_chain_ids(
            ant_repr=ant_repr,
            mhc_repr=mhc_repr,
            antigen_pad_mask=antigen_pad_mask,
            mhc_pad_mask=mhc_pad_mask,
            tcr_tokens=tcr_tokens,
        )
        if self.pmhc_chain_embedding is not None:
            if pmhc_chain_ids is None:
                raise RuntimeError("pMHC chain embedding requested but pmhc_chain_ids were not built")
            pmhc_repr = pmhc_repr + self.pmhc_chain_embedding(pmhc_chain_ids)

        # 5. Cross interaction -------------------------------------------
        attn_weights: Optional[Dict[str, List[torch.Tensor]]] = None
        if return_aux:
            tcr_joint, pmhc_joint, attn_weights = self.cross_interaction(
                tcr=tcr_repr,
                pmhc=pmhc_repr,
                tcr_pad_mask=tcr_pad_mask,
                pmhc_pad_mask=pmhc_pad_mask,
                tcr_chain_ids=tcr_chain_ids,
                pmhc_chain_ids=pmhc_chain_ids,
                return_weights=True,
            )
        else:
            tcr_joint, pmhc_joint = self.cross_interaction(
                tcr=tcr_repr,
                pmhc=pmhc_repr,
                tcr_pad_mask=tcr_pad_mask,
                pmhc_pad_mask=pmhc_pad_mask,
                tcr_chain_ids=tcr_chain_ids,
                pmhc_chain_ids=pmhc_chain_ids,
            )

        # 6. Masked mean pool (no L2) ------------------------------------
        pmhc_vec = self._pool_side("pmhc", pmhc_joint, pmhc_valid_mask)     # [B, D or 2D]
        tcr_vec = self._pool_side("tcr", tcr_joint, tcr_valid_mask)         # [B, D or 2D]
        peptide_vec: Optional[torch.Tensor] = None
        cdr3_vec: Optional[torch.Tensor] = None
        if self._uses_focused_fusion():
            peptide_focus_mask = torch.cat(
                [antigen_valid_mask, torch.zeros_like(mhc_valid_mask)],
                dim=1,
            )
            compartment_ids = tcr_tokens[:, :, 3].long()
            cdr3_focus_mask = tcr_valid_mask & (
                compartment_ids.eq(int(TcrCompartmentIndex.CDR3a))
                | compartment_ids.eq(int(TcrCompartmentIndex.CDR3b))
            )
            peptide_vec = self._pool_side("pmhc", pmhc_joint, peptide_focus_mask)
            cdr3_vec = self._pool_side("tcr", tcr_joint, cdr3_focus_mask)

        # 7. Fuse --------------------------------------------------------
        fused = self._fuse(
            pmhc_vec,
            tcr_vec,
            peptide_vec=peptide_vec,
            cdr3_vec=cdr3_vec,
            pmhc_tokens=pmhc_joint,
            tcr_tokens=tcr_joint,
            pmhc_valid_mask=pmhc_valid_mask,
            tcr_valid_mask=tcr_valid_mask,
        )                                                                    # [B, fuse_in_dim]

        # 8. Head --------------------------------------------------------
        logits = self.head(fused).squeeze(-1)                               # [B]

        if return_aux:
            aux: Dict[str, Any] = {
                "pmhc_vec": pmhc_vec,
                "tcr_vec": tcr_vec,
                "attn_weights": attn_weights,
            }
            if peptide_vec is not None and cdr3_vec is not None:
                aux["peptide_vec"] = peptide_vec
                aux["cdr3_vec"] = cdr3_vec
            return logits, aux
        return logits

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _pool_side(
        self,
        side: str,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.cfg.pool_mode
        mean_vec = masked_mean_pool(x, valid_mask)
        if mode == "mean":
            return mean_vec

        attn_pool = self.pmhc_attn_pool if side == "pmhc" else self.tcr_attn_pool
        if attn_pool is None:  # pragma: no cover - guarded by ModelConfig
            raise RuntimeError(f"pool_mode={mode!r} requires an attention pool")
        attn_vec = attn_pool(x, valid_mask)

        if mode == "attention":
            return attn_vec
        if mode == "mean_attention":
            return torch.cat([mean_vec, attn_vec], dim=-1)
        raise ValueError(f"unknown pool_mode: {mode!r}")  # pragma: no cover

    def _uses_focused_fusion(self) -> bool:
        return self.cfg.fuse_mode in (
            "focused_concat_diff_mul",
            "focused_concat_diff_mul_lowrank",
            "contact_concat_diff_mul",
            "contact_concat_diff_mul_lowrank",
        )

    def _fuse(
        self,
        pmhc_vec: torch.Tensor,
        tcr_vec: torch.Tensor,
        *,
        peptide_vec: Optional[torch.Tensor] = None,
        cdr3_vec: Optional[torch.Tensor] = None,
        pmhc_tokens: Optional[torch.Tensor] = None,
        tcr_tokens: Optional[torch.Tensor] = None,
        pmhc_valid_mask: Optional[torch.Tensor] = None,
        tcr_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = self.cfg.fuse_mode
        if mode == "concat_diff_mul":
            return torch.cat(
                [pmhc_vec, tcr_vec, (pmhc_vec - tcr_vec).abs(), pmhc_vec * tcr_vec],
                dim=-1,
            )
        if mode == "concat":
            return torch.cat([pmhc_vec, tcr_vec], dim=-1)
        if mode == "bilinear":
            assert self._bilinear is not None
            return self._bilinear(pmhc_vec, tcr_vec)
        if mode == "concat_diff_mul_lowrank":
            assert self._lowrank_p is not None and self._lowrank_t is not None
            lowrank = self._lowrank_p(pmhc_vec) * self._lowrank_t(tcr_vec)
            return torch.cat(
                [
                    pmhc_vec,
                    tcr_vec,
                    (pmhc_vec - tcr_vec).abs(),
                    pmhc_vec * tcr_vec,
                    lowrank,
                ],
                dim=-1,
            )
        if mode == "gated_concat_diff_mul":
            if self._gated_fuse is None:
                raise RuntimeError("gated fusion module is not initialized")
            base = torch.cat(
                [pmhc_vec, tcr_vec, (pmhc_vec - tcr_vec).abs(), pmhc_vec * tcr_vec],
                dim=-1,
            )
            gate = self._gated_fuse(base)
            gated = gate * pmhc_vec + (1.0 - gate) * tcr_vec
            return torch.cat([base, gated], dim=-1)
        if mode == "pair_map_cnn":
            if (
                pmhc_tokens is None
                or tcr_tokens is None
                or pmhc_valid_mask is None
                or tcr_valid_mask is None
            ):
                raise RuntimeError("pair_map_cnn requires token tensors and valid masks")
            pair_vec = self._pair_map_vector(
                pmhc_tokens=pmhc_tokens,
                tcr_tokens=tcr_tokens,
                pmhc_valid_mask=pmhc_valid_mask,
                tcr_valid_mask=tcr_valid_mask,
            )
            return torch.cat(
                [
                    pmhc_vec,
                    tcr_vec,
                    (pmhc_vec - tcr_vec).abs(),
                    pmhc_vec * tcr_vec,
                    pair_vec,
                ],
                dim=-1,
            )
        if mode in ("focused_concat_diff_mul", "focused_concat_diff_mul_lowrank"):
            if peptide_vec is None or cdr3_vec is None:
                raise RuntimeError(f"{mode} requires peptide_vec and cdr3_vec")
            parts = [
                pmhc_vec,
                tcr_vec,
                (pmhc_vec - tcr_vec).abs(),
                pmhc_vec * tcr_vec,
                peptide_vec,
                cdr3_vec,
                (peptide_vec - cdr3_vec).abs(),
                peptide_vec * cdr3_vec,
            ]
            if mode == "focused_concat_diff_mul_lowrank":
                assert self._lowrank_p is not None and self._lowrank_t is not None
                parts.append(self._lowrank_p(peptide_vec) * self._lowrank_t(cdr3_vec))
            return torch.cat(parts, dim=-1)
        if mode in ("contact_concat_diff_mul", "contact_concat_diff_mul_lowrank"):
            if peptide_vec is None or cdr3_vec is None:
                raise RuntimeError(f"{mode} requires peptide_vec and cdr3_vec")
            parts = [
                peptide_vec,
                cdr3_vec,
                (peptide_vec - cdr3_vec).abs(),
                peptide_vec * cdr3_vec,
            ]
            if mode == "contact_concat_diff_mul_lowrank":
                assert self._lowrank_p is not None and self._lowrank_t is not None
                parts.append(self._lowrank_p(peptide_vec) * self._lowrank_t(cdr3_vec))
            return torch.cat(parts, dim=-1)
        raise ValueError(f"unknown fuse_mode: {mode!r}")  # pragma: no cover

    def _pair_map_vector(
        self,
        *,
        pmhc_tokens: torch.Tensor,
        tcr_tokens: torch.Tensor,
        pmhc_valid_mask: torch.Tensor,
        tcr_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._pair_p is None or self._pair_t is None or self._pair_cnn is None:
            raise RuntimeError("pair-map modules are not initialized")
        p = self._pair_p(pmhc_tokens)
        t = self._pair_t(tcr_tokens)
        pair = torch.einsum("bpr,btr->bpt", p, t) / math.sqrt(max(1, p.size(-1)))
        valid = pmhc_valid_mask.unsqueeze(2) & tcr_valid_mask.unsqueeze(1)
        pair = pair.masked_fill(~valid, 0.0)
        return self._pair_cnn(pair.unsqueeze(1))

    # ------------------------------------------------------------------
    # Chain-id construction
    # ------------------------------------------------------------------

    def _build_chain_ids(
        self,
        *,
        ant_repr: torch.Tensor,
        mhc_repr: torch.Tensor,
        antigen_pad_mask: torch.Tensor,
        mhc_pad_mask: torch.Tensor,
        tcr_tokens: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cross_interaction.use_chain_bias and self.pmhc_chain_embedding is None:
            return None, None

        B, La, _ = ant_repr.shape
        Lm = mhc_repr.size(1)
        device = ant_repr.device

        ant_chain = torch.full(
            (B, La), int(PMHCChainIndex.ANT), dtype=torch.long, device=device
        )
        mhc_chain = torch.full(
            (B, Lm), int(PMHCChainIndex.MHC), dtype=torch.long, device=device
        )
        # Pad positions → NULL chain id (stays inside the chain_bias table
        # and is irrelevant since pad keys are masked to −inf anyway).
        ant_chain = ant_chain.masked_fill(antigen_pad_mask, int(PMHCChainIndex.NULL))
        mhc_chain = mhc_chain.masked_fill(mhc_pad_mask, int(PMHCChainIndex.NULL))
        pmhc_chain_ids = torch.cat([ant_chain, mhc_chain], dim=1)

        # The 4-tuple tcr_token is [aa_id, within_pos, total_len, compartment_id].
        # The compartment id slot matches TcrCompartmentIndex 1:1.
        tcr_chain_ids = (
            tcr_tokens[:, :, 3].long()
            if self.cross_interaction.use_chain_bias
            else None
        )

        return tcr_chain_ids, pmhc_chain_ids

    # ------------------------------------------------------------------
    # Input sanity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_pair(
        tag: str,
        repr_: torch.Tensor,
        pad_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if repr_.dim() != 3:
            raise ValueError(
                f"{tag} representation must be [B, L, D]; got {tuple(repr_.shape)}"
            )
        if pad_mask.shape != repr_.shape[:2]:
            raise ValueError(
                f"{tag}_pad_mask shape {tuple(pad_mask.shape)} must match "
                f"{tag}_repr prefix {tuple(repr_.shape[:2])}"
            )
        if valid_mask.shape != repr_.shape[:2]:
            raise ValueError(
                f"{tag}_valid_mask shape {tuple(valid_mask.shape)} must match "
                f"{tag}_repr prefix {tuple(repr_.shape[:2])}"
            )
        if pad_mask.dtype != torch.bool:
            raise TypeError(
                f"{tag}_pad_mask must be BoolTensor; got dtype={pad_mask.dtype}"
            )
        if valid_mask.dtype != torch.bool:
            raise TypeError(
                f"{tag}_valid_mask must be BoolTensor; got dtype={valid_mask.dtype}"
            )

    # ------------------------------------------------------------------
    # Freeze / warmup-unfreeze
    # ------------------------------------------------------------------

    def _apply_initial_freeze(self) -> None:
        """Apply the freeze flags on :class:`ModelConfig`.

        :class:`PMHCTower` already handled ``freeze_antigen_*`` /
        ``freeze_mhc_*`` inside its own ``__init__``; we do **not** re-run
        those here.
        """
        cfg = self.cfg

        if cfg.freeze_tcr_embedder:
            for p in self.tcr_bert.token_embedder.parameters():
                p.requires_grad = False
            logger.info(
                "[TCREpitopeSpecificityModel] froze tcr_bert.token_embedder (%d params)",
                sum(p.numel() for p in self.tcr_bert.token_embedder.parameters()),
            )

        if cfg.tcr_finetune_last_only:
            if (
                cfg.freeze_tcr_encoder_layers is not None
                and cfg.freeze_tcr_encoder_layers > 0
            ):
                logger.warning(
                    "[TCREpitopeSpecificityModel] tcr_finetune_last_only=True takes "
                    "precedence over freeze_tcr_encoder_layers=%d",
                    cfg.freeze_tcr_encoder_layers,
                )
            self.tcr_bert.set_fine_tuning_mode(True)
            logger.info(
                "[TCREpitopeSpecificityModel] TCR stack in finetune_last_only mode"
            )
        elif (
            cfg.freeze_tcr_encoder_layers is not None
            and cfg.freeze_tcr_encoder_layers > 0
        ):
            self._freeze_tcr_bottom_k(int(cfg.freeze_tcr_encoder_layers))

    def _freeze_tcr_bottom_k(self, k: int) -> None:
        stack = self.tcr_bert.self_attention_stack
        torch_stack = getattr(stack, "stack", None)
        layers = getattr(torch_stack, "layers", None) if torch_stack is not None else None
        if layers is None:
            logger.warning(
                "[TCREpitopeSpecificityModel] freeze_tcr_encoder_layers=%d requested but "
                "tcr_bert.self_attention_stack has no .stack.layers; skipping",
                k,
            )
            return
        n_freeze = min(k, len(layers))
        for i in range(n_freeze):
            for p in layers[i].parameters():
                p.requires_grad = False
        logger.info(
            "[TCREpitopeSpecificityModel] froze bottom %d/%d TCR encoder layers",
            n_freeze,
            len(layers),
        )

    def warmup_unfreeze(self, epoch: int) -> List[str]:
        """Unfreeze modules per :attr:`ModelConfig.warmup_unfreeze_schedule`.

        At each call, apply every ``(sched_epoch, action)`` pair whose
        ``sched_epoch <= epoch`` that has not been applied yet, in
        ascending order of ``sched_epoch``.

        Returns the list of action names that fired **this call** — empty
        list = nothing new at this epoch (or no schedule configured).
        """
        schedule = self.cfg.warmup_unfreeze_schedule
        if not schedule:
            return []
        fired_this_call: List[str] = []
        for sched_epoch in sorted(schedule):
            if sched_epoch > int(epoch):
                break
            for action in schedule[sched_epoch]:
                key = (sched_epoch, action)
                if key in self._applied_unfreeze:
                    continue
                self._apply_unfreeze_action(action)
                self._applied_unfreeze.add(key)
                fired_this_call.append(action)
        return fired_this_call

    def _apply_unfreeze_action(self, action: str) -> None:
        if action not in VALID_UNFREEZE_ACTIONS:
            raise ValueError(
                f"unknown warmup_unfreeze action {action!r}; expected one of "
                f"{sorted(VALID_UNFREEZE_ACTIONS)}"
            )
        if action == "tcr_encoder":
            for p in self.tcr_bert.self_attention_stack.parameters():
                p.requires_grad = True
        elif action == "tcr_embedder":
            for p in self.tcr_bert.token_embedder.parameters():
                p.requires_grad = True
        elif action == "antigen_backbone":
            for p in self.pmhc_tower.ant_backbone.parameters():
                p.requires_grad = True
        elif action == "mhc_backbone":
            for p in self.pmhc_tower.mhc_backbone.parameters():
                p.requires_grad = True
        elif action == "all":
            for p in self.parameters():
                p.requires_grad = True
        logger.info("[TCREpitopeSpecificityModel] warmup_unfreeze fired: %s", action)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def trainable_param_counts(self) -> Dict[str, int]:
        """Grouped ``param.numel()`` sum over ``requires_grad=True`` tensors.

        Keys: ``pmhc_tower / tcr_bert / cross_interaction / head / bilinear /
        total`` — the last one is the authoritative total (matches
        ``sum(p.numel() for p in self.parameters() if p.requires_grad)``).
        """

        def _count(m: Optional[nn.Module]) -> int:
            if m is None:
                return 0
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        out: Dict[str, int] = {
            "pmhc_tower": _count(self.pmhc_tower),
            "tcr_bert": _count(self.tcr_bert),
            "cross_interaction": _count(self.cross_interaction),
            "head": _count(self.head),
            "bilinear": _count(self._bilinear),
            "pool": _count(self.pmhc_attn_pool) + _count(self.tcr_attn_pool),
            "lowrank": _count(self._lowrank_p) + _count(self._lowrank_t),
            "gated_fuse": _count(self._gated_fuse),
            "pair_map": _count(self._pair_p) + _count(self._pair_t) + _count(self._pair_cnn),
        }
        if self.pmhc_chain_embedding is not None:
            out["pmhc_chain_embedding"] = _count(self.pmhc_chain_embedding)
        out["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return out
