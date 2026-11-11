"""Single source of truth for model / training / length hyper-parameters.

See ``DATA_CONTRACT.md`` (§3 / §9) for the constraints encoded here.

主要类型:

- :class:`LengthConfig`     — peptide / mhc_pseudo / cdr / chain_cdr_concat / tcr_tokens 上限
- :class:`ModelConfig`      — 全局 d_model / 两路 backbone / TCR 塔 / 交互层 / 融合与 head
- :class:`TrainConfig`      — stage / lr / batch / epochs / amp / early_stopping 等
- :class:`ExperimentConfig` — 顶层聚合 + ``from_yaml / from_json / to_yaml`` I/O

所有业务代码（``data/`` / ``models/`` / ``training/`` / ``scripts/``）必须**只**从本模块读取
上述超参；禁止自行 hard-code d_model / 序列长度 / 阈值等常量。
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, Union

import yaml

logger = logging.getLogger(__name__)

__all__ = [
    "LengthConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "ALLOWED_STAGES",
    "ALLOWED_CROSS_MODES",
    "ALLOWED_FUSE_MODES",
    "ALLOWED_PMHC_DTYPES",
    "ALLOWED_ES_METRICS",
    "ALLOWED_TCR_PRETRAIN_MODES",
    "ALLOWED_TCR_STACK_TYPES",
    "ALLOWED_TCR_REPR_POOL",
    "ALLOWED_POOL_MODES",
    "ALLOWED_LOSS_TYPES",
    "ALLOWED_GENERATION_DECODER_ARCHES",
    "ALLOWED_GENERATION_DECODER_CONDITIONS",
    "ALLOWED_GENERATION_DECODE_STRATEGIES",
]


# ---------------------------------------------------------------------------
# Enum-like value sets (kept as frozensets for fast membership checks).
# ---------------------------------------------------------------------------

ALLOWED_STAGES: frozenset[str] = frozenset(
    {"pmhc_pretrain", "tcr_pretrain", "specificity_finetune", "cdr3b_generation"}
)
ALLOWED_CROSS_MODES: frozenset[str] = frozenset(
    {"tcr_to_pmhc", "pmhc_to_tcr", "bidirectional"}
)
ALLOWED_FUSE_MODES: frozenset[str] = frozenset(
    {
        "concat_diff_mul",
        "concat",
        "bilinear",
        "concat_diff_mul_lowrank",
        "focused_concat_diff_mul",
        "focused_concat_diff_mul_lowrank",
        "contact_concat_diff_mul",
        "contact_concat_diff_mul_lowrank",
        "gated_concat_diff_mul",
        "pair_map_cnn",
    }
)
ALLOWED_POOL_MODES: frozenset[str] = frozenset(
    {"mean", "attention", "mean_attention"}
)
ALLOWED_PMHC_DTYPES: frozenset[str] = frozenset(
    {"float32", "float16", "bfloat16"}
)
# 已知的 early-stopping 指标名（DATA_CONTRACT §9 + 后续 metrics.py 约定）。
# 允许使用未列出的名字，但会打印 warning 提示可能是笔误。
ALLOWED_ES_METRICS: frozenset[str] = frozenset(
    {
        "val_loss",
        "val_total_loss",
        "val_auroc",
        "val_auprc",
        "val_mean_per_epitope_auprc",
        "val_median_per_epitope_auprc",
        "val_mean_per_phla_auprc",
        "val_median_per_phla_auprc",
        "val_same_hla_auprc",
        "val_token_nll",
        "val_perplexity",
        "val_exact_match",
    }
)

ALLOWED_TCR_PRETRAIN_MODES: frozenset[str] = frozenset(
    {"mlm_only", "ac_only", "joint_autocontrastive_mlm"}
)
ALLOWED_TCR_STACK_TYPES: frozenset[str] = frozenset({"transformer", "roformer", "reformer", "sceptr_hybrid"})
ALLOWED_TCR_REPR_POOL: frozenset[str] = frozenset({"cls", "average_pool"})
ALLOWED_LOSS_TYPES: frozenset[str] = frozenset({"bce", "focal_bce"})
ALLOWED_GENERATION_DECODER_ARCHES: frozenset[str] = frozenset(
    {"transformer", "prenorm_swiglu"}
)
ALLOWED_GENERATION_DECODER_CONDITIONS: frozenset[str] = frozenset(
    {"cross_attn", "pooled_add", "pooled_gate"}
)
ALLOWED_GENERATION_DECODE_STRATEGIES: frozenset[str] = frozenset(
    {"greedy", "beam"}
)


def _require(name: str, value: Any, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(
            f"{name!r} must be one of {sorted(allowed)!r}; got {value!r}"
        )


def _filter_known(cls: Type, d: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only keys that match a dataclass field; warn about the rest."""
    known = {f.name for f in fields(cls)}
    extras = {k: v for k, v in d.items() if k not in known}
    if extras:
        logger.warning(
            "[config] ignoring unknown keys for %s: %s",
            cls.__name__,
            sorted(extras),
        )
    return {k: v for k, v in d.items() if k in known}


# ---------------------------------------------------------------------------
# LengthConfig — locked by DATA_CONTRACT §3 / §7.2.
# ---------------------------------------------------------------------------


@dataclass
class LengthConfig:
    """Immutable length upper bounds + paired-αβ TCR per-chain budget."""

    # Upper bounds (DATA_CONTRACT §3)
    peptide: int = 15
    mhc_pseudo: int = 40
    cdr: int = 20
    chain_cdr_concat: int = 48
    tcr_tokens: int = 96

    # Per-chain CDR budget (DATA_CONTRACT §7.2)
    cdr1_budget: int = 7
    cdr2_budget: int = 8
    cdr3_budget: int = 20

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not (isinstance(v, int) and v > 0):
                raise ValueError(f"LengthConfig.{f.name} must be positive int; got {v!r}")

        chain_sum = self.cdr1_budget + self.cdr2_budget + self.cdr3_budget
        if chain_sum > self.chain_cdr_concat:
            raise ValueError(
                f"cdr1+cdr2+cdr3 budget ({chain_sum}) exceeds chain_cdr_concat "
                f"({self.chain_cdr_concat})"
            )
        # 布局 = [CLS] + αCDR1/2/3 + βCDR1/2/3 总长 ≤ tcr_tokens
        layout_upper = 1 + 2 * self.chain_cdr_concat
        if layout_upper < self.tcr_tokens:
            raise ValueError(
                f"tcr_tokens={self.tcr_tokens} exceeds max layout "
                f"(1 + 2 * chain_cdr_concat = {layout_upper})"
            )

    @property
    def per_chain_budget(self) -> Tuple[int, int, int]:
        return (self.cdr1_budget, self.cdr2_budget, self.cdr3_budget)


# ---------------------------------------------------------------------------
# ModelConfig.
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Architectural hyper-parameters for the dual-tower + cross-attn model."""

    # ----- Global -----
    d_model: int = 512
    n_heads: int = 8
    dropout: float = 0.1

    # ----- pMHC tower -----
    antigen_backbone_name: str = "esmc_600m"
    mhc_backbone_name: str = "esmc_600m"
    freeze_antigen_backbone: bool = True
    freeze_mhc_backbone: bool = True
    # 细粒度冻结：仅当 ``freeze_*_backbone=False`` 时生效；
    # ``None`` 或 0 表示"全放开"，正整数 K 表示冻结 backbone transformer 前 K 层。
    freeze_antigen_bottom_k: Optional[int] = None
    freeze_mhc_bottom_k: Optional[int] = None
    pmhc_inner_cross_layers: int = 1
    pmhc_inner_cross_bidirectional: bool = False
    pmhc_token_fcn_hidden_mult: int = 4
    pmhc_backbone_dtype: str = "bfloat16"  # validated against ALLOWED_PMHC_DTYPES
    pmhc_pretrained_ckpt: Optional[str] = None  # Stage-3 加载 Stage-1 塔权重
    use_pmhc_chain_embedding: bool = False

    # ----- TCR tower -----
    tcr_vocab_size: int = 23                 # NULL/MASK/CLS + 20 AA
    tcr_max_compartment_pos: int = 100       # sin-pos 最大支持长度（DATA_CONTRACT §7.2）
    tcr_n_compartments: int = 8              # NULL + CLS + αCDR1/2/3 + βCDR1/2/3
    tcr_encoder_layers: int = 6
    tcr_encoder_nhead: int = 8               # 独立于全局 n_heads；默认相同
    tcr_encoder_ff: int = 2048
    tcr_encoder_dropout: float = 0.1
    tcr_stack_type: str = "transformer"      # transformer | roformer | reformer | sceptr_hybrid
    tcr_rope_base: float = 10000.0
    tcr_reformer_bucket_size: int = 16
    tcr_reformer_num_hashes: int = 2
    tcr_pretrained_ckpt: Optional[str] = None  # tcrlm 官方 ckpt 或 Stage-2 自训产物
    tcr_finetune_last_only: bool = False
    # 细粒度冻结（与 tcr_finetune_last_only 并存；前者相当于 K=num_layers-1 的特殊写法）：
    freeze_tcr_encoder_layers: Optional[int] = None  # 正整数 K 则冻 tcr stack 前 K 层
    freeze_tcr_embedder: bool = False                # True 则冻 CdrEmbedder 全部参数

    # ----- Cross interaction -----
    cross_layers: int = 2
    cross_mode: str = "bidirectional"  # validated against ALLOWED_CROSS_MODES
    cross_ffn_mult: int = 4
    use_chain_bias: bool = False       # add learnable (pmhc_chain × tcr_chain) bias to QK^T
    #                                    v0.1 default off; enable in configs/*.yaml as needed.

    # ----- Fusion & head -----
    fuse_mode: str = "concat_diff_mul"  # validated against ALLOWED_FUSE_MODES
    pool_mode: str = "mean"             # mean | attention | mean_attention
    fusion_lowrank_dim: int = 256       # only used by concat_diff_mul_lowrank
    head_hidden_dims: Tuple[int, ...] = (1024, 512)
    head_dropout: float = 0.2

    # ----- Warmup-unfreeze schedule (Stage-3) -----
    # ``{epoch: [action_name, ...]}``;  epoch 为 0-based inclusive 触发阈值。
    # 合法 action：``tcr_encoder / tcr_embedder / antigen_backbone / mhc_backbone / all``。
    # 训练循环每个 epoch 调用 ``model.warmup_unfreeze(epoch)`` 自动渐进放开。
    warmup_unfreeze_schedule: Optional[Dict[int, Tuple[str, ...]]] = None

    # ----- CDR3beta generation decoder -----
    generation_decoder_arch: str = "transformer"
    generation_decoder_condition: str = "cross_attn"
    generation_attention_output_gate: bool = False
    generation_tie_embeddings: bool = False

    def __post_init__(self) -> None:
        # head_hidden_dims: YAML 反序列化后是 list，dataclass 需要 tuple。
        if isinstance(self.head_hidden_dims, list):
            self.head_hidden_dims = tuple(self.head_hidden_dims)

        # Enum-like validation
        _require("cross_mode", self.cross_mode, ALLOWED_CROSS_MODES)
        _require("fuse_mode", self.fuse_mode, ALLOWED_FUSE_MODES)
        _require("pool_mode", self.pool_mode, ALLOWED_POOL_MODES)
        _require("tcr_stack_type", self.tcr_stack_type, ALLOWED_TCR_STACK_TYPES)
        _require("pmhc_backbone_dtype", self.pmhc_backbone_dtype, ALLOWED_PMHC_DTYPES)
        _require(
            "generation_decoder_arch",
            self.generation_decoder_arch,
            ALLOWED_GENERATION_DECODER_ARCHES,
        )
        _require(
            "generation_decoder_condition",
            self.generation_decoder_condition,
            ALLOWED_GENERATION_DECODER_CONDITIONS,
        )

        # Numeric / divisibility sanity
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive; got {self.d_model}")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.tcr_encoder_nhead <= 0 or self.d_model % self.tcr_encoder_nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by tcr_encoder_nhead "
                f"({self.tcr_encoder_nhead})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")
        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError(f"head_dropout must be in [0, 1); got {self.head_dropout}")
        if not (0.0 <= self.tcr_encoder_dropout < 1.0):
            raise ValueError(
                f"tcr_encoder_dropout must be in [0, 1); got {self.tcr_encoder_dropout}"
            )
        if self.pmhc_inner_cross_layers < 0:
            raise ValueError("pmhc_inner_cross_layers must be >= 0")
        if self.pmhc_token_fcn_hidden_mult <= 0:
            raise ValueError("pmhc_token_fcn_hidden_mult must be > 0")
        for name, val in (
            ("freeze_antigen_bottom_k", self.freeze_antigen_bottom_k),
            ("freeze_mhc_bottom_k", self.freeze_mhc_bottom_k),
            ("freeze_tcr_encoder_layers", self.freeze_tcr_encoder_layers),
        ):
            if val is not None and val < 0:
                raise ValueError(f"{name} must be None or a non-negative int; got {val}")
        if self.freeze_antigen_backbone and self.freeze_antigen_bottom_k:
            logger.warning(
                "[config] freeze_antigen_backbone=True makes freeze_antigen_bottom_k=%s redundant.",
                self.freeze_antigen_bottom_k,
            )
        if self.freeze_mhc_backbone and self.freeze_mhc_bottom_k:
            logger.warning(
                "[config] freeze_mhc_backbone=True makes freeze_mhc_bottom_k=%s redundant.",
                self.freeze_mhc_bottom_k,
            )
        if self.cross_layers < 0:
            raise ValueError("cross_layers must be >= 0")
        if self.cross_ffn_mult <= 0:
            raise ValueError("cross_ffn_mult must be > 0")
        if self.fusion_lowrank_dim <= 0:
            raise ValueError("fusion_lowrank_dim must be > 0")
        if self.tcr_encoder_layers < 1:
            raise ValueError("tcr_encoder_layers must be >= 1")
        if self.tcr_rope_base <= 1.0:
            raise ValueError("tcr_rope_base must be > 1")
        if self.tcr_reformer_bucket_size <= 0:
            raise ValueError("tcr_reformer_bucket_size must be > 0")
        if self.tcr_reformer_num_hashes <= 0:
            raise ValueError("tcr_reformer_num_hashes must be > 0")
        if self.tcr_vocab_size < 4:
            raise ValueError("tcr_vocab_size must cover NULL/MASK/CLS + at least 1 AA")
        if self.tcr_n_compartments < 2:
            raise ValueError("tcr_n_compartments must include at least NULL + one real segment")
        if self.tcr_max_compartment_pos <= 0:
            raise ValueError("tcr_max_compartment_pos must be > 0")
        if len(self.head_hidden_dims) < 1 or any(h <= 0 for h in self.head_hidden_dims):
            raise ValueError(
                f"head_hidden_dims must be a non-empty tuple of positive ints; "
                f"got {self.head_hidden_dims!r}"
            )

        # warmup_unfreeze_schedule: 只做结构检查，具体 action 合法性交给
        # ``TCREpitopeSpecificityModel.warmup_unfreeze`` 运行时再兜底（允许新增 action）。
        if self.warmup_unfreeze_schedule is not None:
            new_schedule: Dict[int, Tuple[str, ...]] = {}
            for k, v in self.warmup_unfreeze_schedule.items():
                try:
                    epoch = int(k)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"warmup_unfreeze_schedule keys must be int-like; got {k!r}"
                    ) from exc
                if epoch < 0:
                    raise ValueError(
                        f"warmup_unfreeze_schedule epoch must be >= 0; got {epoch}"
                    )
                if isinstance(v, str):
                    raise ValueError(
                        f"warmup_unfreeze_schedule[{epoch}] must be a list/tuple of "
                        f"action names, not a single string; got {v!r}"
                    )
                actions = tuple(str(a) for a in v)
                if not actions:
                    raise ValueError(
                        f"warmup_unfreeze_schedule[{epoch}] must contain at least one action"
                    )
                new_schedule[epoch] = actions
            self.warmup_unfreeze_schedule = new_schedule


# ---------------------------------------------------------------------------
# TrainConfig.
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training-loop hyper-parameters (stage-agnostic base)."""

    stage: str = "specificity_finetune"          # validated against ALLOWED_STAGES
    batch_size: int = 64
    epochs: int = 20
    lr_backbone: float = 1.0e-5
    lr_head: float = 2.0e-4
    weight_decay: float = 1.0e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: bool = True
    pos_weight_auto: bool = True
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    early_stopping_metric: str = "val_mean_per_epitope_auprc"
    early_stopping_patience: int = 5

    # Reproducibility / DataLoader extras (DATA_CONTRACT §9)
    seed: int = 42
    num_workers: int = 4

    # Stage-2 TCR MLM (``train_tcr_mlm``): token-mask probability per eligible AA position.
    mlm_mask_prob: float = 0.15

    # Stage-2 joint SCEPTR-style (``train_tcr_mlm`` when ``training_mode=joint_autocontrastive_mlm``).
    training_mode: str = "mlm_only"
    lambda_cl: float = 1.0
    contrastive_temperature: float = 0.05
    frac_dropped_tokens: float = 0.2
    prob_drop_chain: float = 0.5
    supervised_positives: bool = False
    tcr_repr_pool: str = "cls"
    mlm_label_smoothing: float = 0.0

    # Stage generation: pMHC-conditioned CDR3beta autoregressive decoding.
    generation_max_len: int = 20
    generation_num_candidates: int = 16
    generation_temperature: float = 1.0
    generation_top_k: int = 20
    generation_top_p: float = 0.95
    generation_eval_nearest_train_distance: bool = False
    generation_decode_strategy: str = "greedy"
    generation_beam_width: int = 1
    generation_beam_length_penalty: float = 0.0

    def __post_init__(self) -> None:
        _require("stage", self.stage, ALLOWED_STAGES)
        _require("training_mode", self.training_mode, ALLOWED_TCR_PRETRAIN_MODES)
        _require("tcr_repr_pool", self.tcr_repr_pool, ALLOWED_TCR_REPR_POOL)
        _require("loss_type", self.loss_type, ALLOWED_LOSS_TYPES)
        _require(
            "generation_decode_strategy",
            self.generation_decode_strategy,
            ALLOWED_GENERATION_DECODE_STRATEGIES,
        )
        if self.early_stopping_metric not in ALLOWED_ES_METRICS:
            logger.warning(
                "[config] early_stopping_metric=%r is not in the known set %s; "
                "allowed but double-check for typos.",
                self.early_stopping_metric,
                sorted(ALLOWED_ES_METRICS),
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.lr_backbone <= 0.0 or self.lr_head <= 0.0:
            raise ValueError("learning rates must be > 0")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.focal_gamma < 0.0:
            raise ValueError(f"focal_gamma must be >= 0; got {self.focal_gamma}")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing must be in [0, 1); got {self.label_smoothing}"
            )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be > 0")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be >= 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if not (0.0 < self.mlm_mask_prob <= 1.0):
            raise ValueError(
                f"mlm_mask_prob must be in (0, 1]; got {self.mlm_mask_prob}"
            )
        if self.lambda_cl < 0.0:
            raise ValueError(f"lambda_cl must be >= 0; got {self.lambda_cl}")
        if self.contrastive_temperature <= 0.0:
            raise ValueError(
                f"contrastive_temperature must be > 0; got {self.contrastive_temperature}"
            )
        if not (0.0 <= self.frac_dropped_tokens <= 1.0):
            raise ValueError(
                f"frac_dropped_tokens must be in [0, 1]; got {self.frac_dropped_tokens}"
            )
        if not (0.0 <= self.prob_drop_chain <= 1.0):
            raise ValueError(
                f"prob_drop_chain must be in [0, 1]; got {self.prob_drop_chain}"
            )
        if not (0.0 <= self.mlm_label_smoothing < 1.0):
            raise ValueError(
                f"mlm_label_smoothing must be in [0, 1); got {self.mlm_label_smoothing}"
            )
        if self.generation_max_len <= 0 or self.generation_max_len > 20:
            raise ValueError(
                "generation_max_len must be in [1, 20] for phase-1 CDR3b "
                f"generation; got {self.generation_max_len}"
            )
        if self.generation_num_candidates <= 0:
            raise ValueError("generation_num_candidates must be > 0")
        if self.generation_temperature <= 0.0:
            raise ValueError("generation_temperature must be > 0")
        if self.generation_top_k < 0:
            raise ValueError("generation_top_k must be >= 0")
        if not (0.0 < self.generation_top_p <= 1.0):
            raise ValueError("generation_top_p must be in (0, 1]")
        if self.generation_beam_width <= 0:
            raise ValueError("generation_beam_width must be > 0")
        if self.generation_beam_length_penalty < 0.0:
            raise ValueError("generation_beam_length_penalty must be >= 0")


# ---------------------------------------------------------------------------
# ExperimentConfig — YAML 顶层容器。
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Top-level container bundling model / train / lengths configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    lengths: LengthConfig = field(default_factory=LengthConfig)

    # ----- Constructors -----

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentConfig":
        if not isinstance(d, Mapping):
            raise TypeError(
                f"ExperimentConfig.from_dict expects a mapping; got {type(d).__name__}"
            )
        known_top = {"model", "train", "lengths"}
        extras = {k: v for k, v in d.items() if k not in known_top}
        if extras:
            logger.warning(
                "[config] ignoring unknown top-level keys: %s", sorted(extras)
            )
        model_d = d.get("model") or {}
        train_d = d.get("train") or {}
        lengths_d = d.get("lengths") or {}
        if not isinstance(model_d, Mapping):
            raise TypeError("'model' section must be a mapping")
        if not isinstance(train_d, Mapping):
            raise TypeError("'train' section must be a mapping")
        if not isinstance(lengths_d, Mapping):
            raise TypeError("'lengths' section must be a mapping")

        return cls(
            model=ModelConfig(**_filter_known(ModelConfig, model_d)),
            train=TrainConfig(**_filter_known(TrainConfig, train_d)),
            lengths=LengthConfig(**_filter_known(LengthConfig, lengths_d)),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load experiment config from JSON (same schema as YAML)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config json not found: {path}")
        import json

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise TypeError(
                f"top-level of {path} must be a mapping; got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config yaml not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise TypeError(
                f"top-level of {path} must be a mapping; got {type(raw).__name__}"
            )
        logger.info("[config] loaded from %s", path)
        return cls.from_dict(raw)

    # ----- Serializers -----

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": asdict(self.model),
            "train": asdict(self.train),
            "lengths": asdict(self.lengths),
        }
        # tuple → list，便于 YAML 序列化。
        d["model"]["head_hidden_dims"] = list(self.model.head_hidden_dims)
        return d

    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(),
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        logger.info("[config] wrote %s", path)
