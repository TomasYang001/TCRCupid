"""``PMHCTower`` — 基于 PYfile/pMHC_Model.py 精简重写（不含分类头）.

实现契约（step 8 prompt）::

    ant_repr, mhc_repr = tower(
        antigen_tokens=..., mhc_tokens=...,
        antigen_pad_mask=..., mhc_pad_mask=...,
    )
    # ant_repr : [B, La, D]
    # mhc_repr : [B, Lm, D]

组件：

1. 两个独立 ``ESMC.from_pretrained(...)``（懒加载；允许外部注入已加载的 backbone
   或 stub 用于测试）。
2. 两路各接 ``Linear(d_backbone, d_model, bias=False)`` —— 若 backbone 维度已等于
   ``d_model``，则替换为 :class:`nn.Identity`.
3. 两路各接一个 :class:`TokenResidualFCN`：LayerNorm → Linear→GELU→Linear→GELU→Linear
   + 残差；入口先 ``_cast_like(x, ln.weight)`` 对齐 dtype。
4. 塔内可选 ``N_pmhc = model_cfg.pmhc_inner_cross_layers`` 层
   :class:`_CrossAttention`（默认 Q=ant, K/V=mhc），``N_pmhc=0`` 时直接跳过。
   当 ``model_cfg.pmhc_inner_cross_bidirectional=True`` 时，每层额外执行
   ``mhc <- CrossAttention(q=mhc, kv=ant)``，让 MHC pseudo tokens 也携带 peptide context。
5. 冻结策略：
   - ``freeze_antigen_backbone=True`` → 整个 antigen backbone 冻；同理 mhc；
   - 否则按 ``freeze_antigen_bottom_k`` / ``freeze_mhc_bottom_k`` 冻前 K 层
     transformer block（通过 :func:`_find_transformer_blocks` 查找）。
6. **无分类头**（pMHC binding head 在 step 9 `models/heads.py` 里）。

与 ``PYfile/pMHC_Model.py`` 相比：
- 迁移了 ``_forward_backbone / _extract_token_embeddings / _infer_hidden_dim /
  _infer_hidden_dim_from_dummy`` 四件套以兼容不同 ``esm`` 版本；
- 去掉了原文件的 ``MLPClassifier / ESMC_pMHC_BindingModel``（职责分到 step 9/13）；
- 所有 ``esm`` 依赖 **lazy import**，缺失 esm 时 stub backbone 依旧可测试。
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "PMHCTower",
    "PMHCBindingModel",
    "TokenResidualFCN",
]


# ---------------------------------------------------------------------------
# ESM backbone compatibility utilities (ESM-C plus HF ESM2).
# ---------------------------------------------------------------------------


_ESM2_ALIASES = {
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    "esm2-650m": "facebook/esm2_t33_650M_UR50D",
    "esm2_650": "facebook/esm2_t33_650M_UR50D",
    "esm2-650": "facebook/esm2_t33_650M_UR50D",
    "esm2_t33_650m_ur50d": "facebook/esm2_t33_650M_UR50D",
}
_ESM3_ALIASES = {
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


class HFESM2Backbone(nn.Module):
    """Thin wrapper that makes HuggingFace ESM2 look like the ESM-C backbones.

    The rest of the pMHC tower expects a module whose forward accepts token ids
    and returns an object with ``last_hidden_state`` or ``embeddings``. HF ESM2
    already does that; the wrapper only derives an attention mask from the
    tokenizer pad id and exposes common attributes for hidden-size and
    bottom-layer freezing.
    """

    def __init__(
        self,
        name: str,
        *,
        torch_dtype: torch.dtype = torch.float32,
        init_from_config_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer  # lazy import

        self.name = _canonical_backbone_name(name)
        if init_from_config_only:
            logger.warning(
                "[pmhc-tower] initializing HF ESM2 architecture from local config only: %s",
                self.name,
            )
            config = AutoConfig.from_pretrained(self.name, local_files_only=True)
            self.model = AutoModel.from_config(config).to(dtype=torch_dtype)
        else:
            self.model = AutoModel.from_pretrained(
                self.name,
                torch_dtype=torch_dtype,
                use_safetensors=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            local_files_only=bool(init_from_config_only),
        )

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def encoder(self) -> Any:
        return getattr(self.model, "encoder", None)

    @property
    def hidden_size(self) -> Optional[int]:
        value = getattr(self.model.config, "hidden_size", None)
        return int(value) if isinstance(value, int) else None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        *,
        tokens: Optional[torch.Tensor] = None,
        sequence_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None:
            input_ids = tokens if tokens is not None else sequence_tokens
        if input_ids is None:
            raise ValueError("HFESM2Backbone.forward requires token ids")
        if attention_mask is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                attention_mask = input_ids.ne(int(pad_id)).to(dtype=torch.long)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_pmhc_backbone(name: str, *, torch_dtype: torch.dtype) -> nn.Module:
    resolved_name = _canonical_backbone_name(name)
    if _is_hf_esm2_name(resolved_name):
        return HFESM2Backbone(
            resolved_name,
            torch_dtype=torch_dtype,
            init_from_config_only=_env_flag("TCR_EPI_CROSSATTN_HF_ESM2_CONFIG_ONLY"),
        ).to(dtype=torch_dtype)
    if _is_esm3_name(resolved_name):
        from esm.models.esm3 import ESM3  # lazy import

        return ESM3.from_pretrained(resolved_name).to(dtype=torch_dtype)

    from esm.models.esmc import ESMC  # lazy import

    return ESMC.from_pretrained(resolved_name).to(dtype=torch_dtype)


def _cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Cast ``x`` to the device + dtype of ``ref``.

    Used at every ``LayerNorm`` entry point so fp16/bf16 backbones mixed with
    fp32 LN weights don't raise a dtype mismatch.
    """
    if x.device != ref.device or x.dtype != ref.dtype:
        x = x.to(device=ref.device, dtype=ref.dtype)
    return x


def _forward_backbone(
    model: nn.Module,
    tokens: torch.Tensor,
    return_embeddings: bool = True,
) -> Any:
    """Call a backbone's forward regardless of its argument naming.

    Handles ESM-C / ESM3 / HF ESM2 variants which use ``sequence_tokens``,
    ``tokens``, or ``input_ids``. Falls back to positional as a last resort.
    """
    sig = inspect.signature(model.forward)
    params = sig.parameters
    kwargs: dict = {}

    if "sequence_tokens" in params:
        kwargs["sequence_tokens"] = tokens
    elif "tokens" in params:
        kwargs["tokens"] = tokens
    elif "input_ids" in params:
        kwargs["input_ids"] = tokens
    else:
        return model(tokens)

    # Some esm builds accept a LogitsConfig to toggle which tracks come back.
    if "logits_config" in params:
        try:
            from esm.sdk.api import LogitsConfig  # local import

            kwargs["logits_config"] = LogitsConfig(
                sequence=False, return_embeddings=return_embeddings
            )
        except Exception:  # pragma: no cover - defensive
            pass

    return model(**kwargs)


def _extract_token_embeddings(out: Any) -> torch.Tensor:
    """Extract ``[B, L, D]`` token embeddings from a variety of ESM output shapes."""
    if hasattr(out, "embeddings"):
        emb = out.embeddings
        if torch.is_tensor(emb):
            return emb
        if hasattr(emb, "sequence") and torch.is_tensor(emb.sequence):
            return emb.sequence
        if isinstance(emb, dict) and "sequence" in emb and torch.is_tensor(emb["sequence"]):
            return emb["sequence"]

    if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
        return out.last_hidden_state

    if hasattr(out, "hidden_states"):
        hs = out.hidden_states
        if isinstance(hs, (list, tuple)) and len(hs) > 0 and torch.is_tensor(hs[-1]):
            return hs[-1]

    if isinstance(out, dict):
        for k in ("embeddings", "last_hidden_state", "hidden_states", "sequence_embeddings"):
            if k in out:
                v = out[k]
                if torch.is_tensor(v):
                    return v
                if isinstance(v, (list, tuple)) and len(v) > 0 and torch.is_tensor(v[-1]):
                    return v[-1]
                if isinstance(v, dict) and "sequence" in v and torch.is_tensor(v["sequence"]):
                    return v["sequence"]

    raise TypeError(
        f"Cannot extract token embeddings from backbone output type={type(out)}"
    )


def _infer_hidden_dim(backbone: nn.Module) -> Optional[int]:
    """Best-effort hidden-size readout from common backbone attributes."""
    for attr in ("embed_dim", "d_model", "hidden_size", "dim", "model_dim"):
        v = getattr(backbone, attr, None)
        if isinstance(v, int):
            return v
    for attr in ("args", "cfg", "config"):
        cfg = getattr(backbone, attr, None)
        if cfg is None:
            continue
        for key in ("embed_dim", "hidden_size", "model_dim", "d_model"):
            v = getattr(cfg, key, None)
            if isinstance(v, int):
                return v
            if isinstance(cfg, dict) and key in cfg and isinstance(cfg[key], int):
                return cfg[key]
    return None


def _infer_hidden_dim_from_dummy(backbone: nn.Module) -> int:
    """Run a 1-residue forward to recover ``D`` from ``embeddings.shape[-1]``.

    Requires the backbone to expose ``encode`` (ESM-C / ESM3 SDK);
    stubs that don't expose it should report ``embed_dim`` via
    :func:`_infer_hidden_dim` instead.
    """
    if not hasattr(backbone, "encode"):
        raise RuntimeError(
            "Backbone has no .encode(); cannot infer hidden_dim from a dummy forward. "
            "Either expose one of {embed_dim/d_model/hidden_size/dim/model_dim} "
            "or provide an ESM-C-like encode(...) method."
        )

    try:
        dev = next(backbone.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")

    from esm.sdk.api import ESMProtein  # local import

    with torch.no_grad():
        pt = backbone.encode(ESMProtein(sequence="A"))
        toks = pt.sequence
        if toks.dim() == 1:
            toks = toks.unsqueeze(0)
        toks = toks.to(dev)

        out = _forward_backbone(backbone, toks, return_embeddings=True)
        emb = _extract_token_embeddings(out)
        if emb.dim() == 4 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        if emb.dim() != 3:
            raise RuntimeError(
                f"Unexpected embedding shape from dummy forward: {tuple(emb.shape)}"
            )
        return int(emb.size(-1))


# ---------------------------------------------------------------------------
# Transformer-block locator (for "freeze bottom-K layers").
# ---------------------------------------------------------------------------


def _find_transformer_blocks(backbone: nn.Module) -> Optional[nn.ModuleList]:
    """Return the backbone's ``ModuleList`` of transformer blocks if discoverable.

    Known-good paths in order of preference:
      - ``backbone.transformer.blocks``  (ESM-C / ESM3 SDK)
      - ``backbone.encoder.layer``        (HF ESM2)
      - ``backbone.encoder.layers``
      - ``backbone.layers``
    """
    for path in (
        ("transformer", "blocks"),
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("model", "encoder", "layer"),
        ("model", "encoder", "layers"),
        ("layers",),
    ):
        obj: Any = backbone
        ok = True
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    return None


def _freeze_bottom_k_layers(backbone: nn.Module, k: int, tag: str) -> int:
    """Freeze the first ``k`` transformer blocks of a backbone.

    Returns the number of actually frozen blocks (0 if no block container was found).
    """
    if k is None or k <= 0:
        return 0
    blocks = _find_transformer_blocks(backbone)
    if blocks is None:
        logger.warning(
            "[pmhc-tower] %s: cannot locate transformer blocks; bottom-K freeze skipped",
            tag,
        )
        return 0
    n = min(k, len(blocks))
    for i in range(n):
        for p in blocks[i].parameters():
            p.requires_grad = False
    logger.info("[pmhc-tower] %s: froze bottom %d / %d transformer blocks", tag, n, len(blocks))
    return n


def _freeze_all(backbone: nn.Module, tag: str) -> None:
    n = 0
    for p in backbone.parameters():
        p.requires_grad = False
        n += 1
    logger.info("[pmhc-tower] %s: froze all %d parameters", tag, n)


# ---------------------------------------------------------------------------
# TokenResidualFCN — per-token 3-layer MLP with residual (PYfile style).
# ---------------------------------------------------------------------------


class TokenResidualFCN(nn.Module):
    """Per-token residual MLP. Output dim == input dim."""

    def __init__(self, d: int, hidden_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        h = d * hidden_mult
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = _cast_like(x, self.ln.weight)
        x = self.ln(x)
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(F.gelu(self.fc2(x)))
        x = self.drop(self.fc3(x))
        return r + x


# ---------------------------------------------------------------------------
# _CrossAttention — pre-LN + MHA + residual + FFN + residual.
# ---------------------------------------------------------------------------
#
# 注：step 12 会把一份同构的 ``CrossAttentionBlock`` 放到 ``models/cross_interaction.py``，
# 届时考虑把本文件里这个私有实现替换为对外公共模块的 import。v0.1 先保持塔内私有
# 以便 step 8 单点完成。
#
class _CrossAttention(nn.Module):
    def __init__(
        self,
        d: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_mult: int = 4,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

        self.ln_out = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ffn_mult, d),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rq = q
        qn = self.ln_q(_cast_like(q, self.ln_q.weight))
        kvn = self.ln_kv(_cast_like(kv, self.ln_kv.weight))
        attn_out, _ = self.attn(
            qn, kvn, kvn,
            key_padding_mask=kv_key_padding_mask,
            need_weights=False,
        )
        x = rq + attn_out
        rx = x
        xn = self.ln_out(_cast_like(x, self.ln_out.weight))
        x = rx + self.ffn(xn)
        return x


# ---------------------------------------------------------------------------
# PMHCTower.
# ---------------------------------------------------------------------------


class PMHCTower(nn.Module):
    """Two ESM-C towers + per-token FCN + optional inner cross-attention.

    Parameters
    ----------
    model_cfg
        Frozen :class:`ModelConfig` — the single source of truth for d_model,
        dropout, freeze flags, inner-cross-attn depth, etc.
    ant_backbone / mhc_backbone
        Pre-loaded ``nn.Module`` backbones. If ``None``, the tower refuses to
        build — use :meth:`from_config` to actually load ESM-C. This split
        keeps ``__init__`` free of ``import esm`` so test environments don't
        require the dependency.
    hidden_dim_override
        Optional ``(d_ant, d_mhc)`` tuple to bypass :func:`_infer_hidden_dim`
        for stubs that don't expose any of the standard attributes.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        *,
        ant_backbone: Optional[nn.Module] = None,
        mhc_backbone: Optional[nn.Module] = None,
        hidden_dim_override: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.cfg = model_cfg

        if ant_backbone is None or mhc_backbone is None:
            raise ValueError(
                "PMHCTower(...) requires both ant_backbone and mhc_backbone. "
                "Use PMHCTower.from_config(cfg) to load real ESM-C backbones, "
                "or inject pre-built (possibly stub) modules in tests."
            )

        self.ant_backbone = ant_backbone
        self.mhc_backbone = mhc_backbone

        # ---- Hidden dims ---------------------------------------------------
        d_ant, d_mhc = self._resolve_hidden_dims(hidden_dim_override)

        # ---- Projection to d_model ----------------------------------------
        D = int(model_cfg.d_model)
        self.d_model = D
        self.ant_proj: nn.Module = (
            nn.Identity() if d_ant == D else nn.Linear(d_ant, D, bias=False)
        )
        self.mhc_proj: nn.Module = (
            nn.Identity() if d_mhc == D else nn.Linear(d_mhc, D, bias=False)
        )

        # ---- Per-token residual FCN ---------------------------------------
        self.ant_fcn = TokenResidualFCN(
            D,
            hidden_mult=model_cfg.pmhc_token_fcn_hidden_mult,
            dropout=model_cfg.dropout,
        )
        self.mhc_fcn = TokenResidualFCN(
            D,
            hidden_mult=model_cfg.pmhc_token_fcn_hidden_mult,
            dropout=model_cfg.dropout,
        )

        # ---- Optional inner cross-attention (Q=ant, K/V=mhc) --------------
        self.inner_cross_layers = nn.ModuleList(
            [
                _CrossAttention(
                    D,
                    n_heads=model_cfg.n_heads,
                    dropout=model_cfg.dropout,
                    ffn_mult=model_cfg.pmhc_token_fcn_hidden_mult,
                )
                for _ in range(int(model_cfg.pmhc_inner_cross_layers))
            ]
        )
        self.inner_cross_mhc_layers = nn.ModuleList(
            [
                _CrossAttention(
                    D,
                    n_heads=model_cfg.n_heads,
                    dropout=model_cfg.dropout,
                    ffn_mult=model_cfg.pmhc_token_fcn_hidden_mult,
                )
                for _ in range(int(model_cfg.pmhc_inner_cross_layers))
            ]
            if bool(getattr(model_cfg, "pmhc_inner_cross_bidirectional", False))
            else []
        )

        # ---- Freeze strategy ----------------------------------------------
        self._apply_freeze()

    # ----- Construction helpers -----

    @classmethod
    def from_config(cls, model_cfg: ModelConfig) -> "PMHCTower":  # pragma: no cover - needs esm
        """Load real protein-language-model backbones and return a ready-to-train tower."""

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[model_cfg.pmhc_backbone_dtype]

        ant = _load_pmhc_backbone(
            model_cfg.antigen_backbone_name,
            torch_dtype=torch_dtype,
        )
        # Always load a fresh instance — the two pMHC streams are independently
        # fine-tunable even when they use the same backbone family/name.
        mhc = _load_pmhc_backbone(
            model_cfg.mhc_backbone_name,
            torch_dtype=torch_dtype,
        )

        return cls(model_cfg, ant_backbone=ant, mhc_backbone=mhc)

    # ----- Freeze management -----

    def _apply_freeze(self) -> None:
        cfg = self.cfg
        if cfg.freeze_antigen_backbone:
            _freeze_all(self.ant_backbone, tag="antigen")
        elif cfg.freeze_antigen_bottom_k:
            _freeze_bottom_k_layers(
                self.ant_backbone, cfg.freeze_antigen_bottom_k, tag="antigen"
            )

        if cfg.freeze_mhc_backbone:
            _freeze_all(self.mhc_backbone, tag="mhc")
        elif cfg.freeze_mhc_bottom_k:
            _freeze_bottom_k_layers(
                self.mhc_backbone, cfg.freeze_mhc_bottom_k, tag="mhc"
            )

    # ----- Hidden-dim resolution -----

    def _resolve_hidden_dims(
        self, override: Optional[Tuple[int, int]]
    ) -> Tuple[int, int]:
        if override is not None:
            d_ant, d_mhc = override
            if d_ant <= 0 or d_mhc <= 0:
                raise ValueError(f"invalid hidden_dim_override: {override!r}")
            return int(d_ant), int(d_mhc)

        d_ant = _infer_hidden_dim(self.ant_backbone)
        if d_ant is None:
            d_ant = _infer_hidden_dim_from_dummy(self.ant_backbone)

        d_mhc = _infer_hidden_dim(self.mhc_backbone)
        if d_mhc is None:
            d_mhc = _infer_hidden_dim_from_dummy(self.mhc_backbone)

        return int(d_ant), int(d_mhc)

    # ----- Backbone encode -----

    def _encode(self, backbone: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        out = _forward_backbone(backbone, tokens, return_embeddings=True)
        emb = _extract_token_embeddings(out)
        if emb.dim() == 4 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        if emb.dim() != 3:
            raise RuntimeError(
                f"backbone returned embeddings of shape {tuple(emb.shape)}; "
                "expected [B, L, D]"
            )
        return emb

    # ----- Forward -----

    def forward(
        self,
        *,
        antigen_tokens: torch.Tensor,
        mhc_tokens: torch.Tensor,
        antigen_pad_mask: Optional[torch.Tensor] = None,
        mhc_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(ant_repr [B, La, D], mhc_repr [B, Lm, D])``.

        ``antigen_pad_mask`` is used only when the optional bidirectional
        pMHC inner cross-attention is enabled.
        """

        ant_emb = self._encode(self.ant_backbone, antigen_tokens)  # [B, La, D_ant]
        mhc_emb = self._encode(self.mhc_backbone, mhc_tokens)      # [B, Lm, D_mhc]

        if isinstance(self.ant_proj, nn.Linear):
            ant_emb = _cast_like(ant_emb, self.ant_proj.weight)
        if isinstance(self.mhc_proj, nn.Linear):
            mhc_emb = _cast_like(mhc_emb, self.mhc_proj.weight)
        ant_emb = self.ant_proj(ant_emb)
        mhc_emb = self.mhc_proj(mhc_emb)

        ant_repr = self.ant_fcn(ant_emb)
        mhc_repr = self.mhc_fcn(mhc_emb)

        for i, layer in enumerate(self.inner_cross_layers):
            ant_repr = layer(
                q=ant_repr,
                kv=mhc_repr,
                kv_key_padding_mask=mhc_pad_mask,
            )
            if i < len(self.inner_cross_mhc_layers):
                mhc_repr = self.inner_cross_mhc_layers[i](
                    q=mhc_repr,
                    kv=ant_repr,
                    kv_key_padding_mask=antigen_pad_mask,
                )

        return ant_repr, mhc_repr

    # ----- Diagnostics -----

    @property
    def trainable_param_counts(self) -> dict:
        """Convenient introspection for freeze QA / logging."""
        out = {"antigen_backbone": 0, "mhc_backbone": 0, "heads": 0, "total": 0}
        for mod, key in (
            (self.ant_backbone, "antigen_backbone"),
            (self.mhc_backbone, "mhc_backbone"),
        ):
            out[key] = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        head_modules: List[nn.Module] = [
            self.ant_proj,
            self.mhc_proj,
            self.ant_fcn,
            self.mhc_fcn,
            self.inner_cross_layers,
            self.inner_cross_mhc_layers,
        ]
        out["heads"] = sum(
            sum(p.numel() for p in m.parameters() if p.requires_grad)
            for m in head_modules
        )
        out["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return out


# ---------------------------------------------------------------------------
# PMHCBindingModel — Stage-1 pMHC binding composite (tower + binding head).
# ---------------------------------------------------------------------------


class PMHCBindingModel(nn.Module):
    """Stage-1 composite: :class:`PMHCTower` + :class:`PMHCBindingHead`.

    Only :mod:`scripts.train_pmhc` is supposed to instantiate this class.
    The specificity stage (step 13) loads **only** ``self.pmhc_tower`` via
    :func:`ckpt_adapter.load_pmhc_ckpt.load_pmhc_tower_into`; the binding
    head is intentionally absent from the persisted checkpoint to keep
    Stage-3 loads free of key conflicts.

    Parameters
    ----------
    model_cfg
        Frozen :class:`ModelConfig`.
    ant_backbone / mhc_backbone
        Pre-built backbones. Pass stubs in tests; pass real ``ESMC`` instances
        via :meth:`from_config`.
    hidden_dim_override
        Forwarded to :class:`PMHCTower` for stub-friendly dim inference.
    head_dropout / head_hidden_mult
        Stage-1 head-specific knobs. Kept **separate** from the specificity
        head's ``ModelConfig.head_dropout`` / ``head_hidden_dims`` so we can
        tune pMHC pre-training without retuning the downstream head.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        *,
        ant_backbone: Optional[nn.Module] = None,
        mhc_backbone: Optional[nn.Module] = None,
        hidden_dim_override: Optional[Tuple[int, int]] = None,
        head_dropout: float = 0.2,
        head_hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = model_cfg

        self.pmhc_tower = PMHCTower(
            model_cfg,
            ant_backbone=ant_backbone,
            mhc_backbone=mhc_backbone,
            hidden_dim_override=hidden_dim_override,
        )

        # Local import: heads.py must be importable without the tower & vice
        # versa (break a circular risk if heads.py ever grows).
        from .heads import PMHCBindingHead

        self.pmhc_head = PMHCBindingHead(
            d_model=model_cfg.d_model,
            hidden_mult=head_hidden_mult,
            dropout=head_dropout,
        )

    # ----- Construction helper -----

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        *,
        head_dropout: float = 0.2,
        head_hidden_mult: int = 2,
    ) -> "PMHCBindingModel":  # pragma: no cover - needs esm
        """Build the binding model with freshly-loaded pMHC backbones."""

        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[model_cfg.pmhc_backbone_dtype]

        ant = _load_pmhc_backbone(
            model_cfg.antigen_backbone_name,
            torch_dtype=torch_dtype,
        )
        mhc = _load_pmhc_backbone(
            model_cfg.mhc_backbone_name,
            torch_dtype=torch_dtype,
        )

        return cls(
            model_cfg,
            ant_backbone=ant,
            mhc_backbone=mhc,
            head_dropout=head_dropout,
            head_hidden_mult=head_hidden_mult,
        )

    # ----- Forward -----

    def forward(
        self,
        *,
        antigen_tokens: torch.Tensor,
        mhc_tokens: torch.Tensor,
        antigen_pad_mask: Optional[torch.Tensor] = None,
        mhc_pad_mask: Optional[torch.Tensor] = None,
        antigen_valid_mask: torch.Tensor,
        **unused_batch: Any,
    ) -> torch.Tensor:
        """Return ``[B]`` binding logits.

        ``**unused_batch`` absorbs the rest of the Batch Contract (tcr_*,
        label, _meta, ...) so the same dict produced by :func:`tcr_epi_collate_fn`
        can be handed to Stage-1 training without key-stripping.
        """
        if antigen_valid_mask is None:
            raise ValueError(
                "PMHCBindingModel.forward requires 'antigen_valid_mask' "
                "(True=valid, excludes CLS/EOS/PAD). Check your collate."
            )
        del unused_batch

        ant_repr, _mhc_repr = self.pmhc_tower(
            antigen_tokens=antigen_tokens,
            mhc_tokens=mhc_tokens,
            antigen_pad_mask=antigen_pad_mask,
            mhc_pad_mask=mhc_pad_mask,
        )
        return self.pmhc_head(ant_repr, antigen_valid_mask)  # [B]

    # ----- Checkpoint I/O -----

    def save_checkpoint(
        self,
        path: Any,
        *,
        meta: Any,
    ) -> None:
        """Persist only the tower under the frozen step-9 schema.

        See :mod:`ckpt_adapter.load_pmhc_ckpt` for the exact layout and the
        matching loader. The binding head is deliberately dropped here.
        """
        from ..ckpt_adapter.load_pmhc_ckpt import save_pmhc_tower_ckpt

        save_pmhc_tower_ckpt(self.pmhc_tower, path, meta=meta)
