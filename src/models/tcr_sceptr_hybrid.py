"""TCR encoder with the requested hybrid input embedding.

Token rows are [AA ID, one-based position within CDR, CDR length, CDR ID].
IDs: NULL=0, MASK=1, CLS=2, amino acids=3..22; CDR IDs=1..6, CLS/PAD=0.
This is a new checkpoint format, not a drop-in state dict for the old TcrBert.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NULL, MASK, CLS = 0, 1, 2
FORMAT = "tcr_sceptr_hybrid_v1"


@dataclass(frozen=True)
class HybridConfig:
    d_model: int = 512
    layers: int = 4
    heads: int = 8
    aa_embedding_dim: int = 512
    dropout: float = 0.1
    activation: str = "gelu"  # Retain the existing encoder; only input/recipe change.
    max_cdr_length: int = 100
    sin_scale: float = 30.0  # libtcrlm's SinPositionEmbedding convention.

    def __post_init__(self):
        if self.d_model < 1 or self.heads < 1 or self.d_model % self.heads:
            raise ValueError("d_model must be positive and divisible by heads")
        if self.layers < 1 or self.aa_embedding_dim < 2 or self.aa_embedding_dim % 2:
            raise ValueError("layers must be positive; aa_embedding_dim must be positive/even")
        if self.max_cdr_length < 1 or self.sin_scale <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid position embedding or dropout settings")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")


class HybridCdrEmbedder(nn.Module):
    """Linear([learned-AA + sine-position; relative-position; CDR-one-hot])."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        width = config.aa_embedding_dim
        self.aa_embedding = nn.Embedding(23, width, padding_idx=NULL)
        positions = torch.arange(config.max_cdr_length, dtype=torch.float32)[:, None]
        wavelengths = config.sin_scale ** (torch.arange(0, width, 2).float() / width)
        table = torch.zeros(config.max_cdr_length + 1, width)
        table[1:, 0::2] = torch.sin(positions / wavelengths)
        table[1:, 1::2] = torch.cos(positions / wavelengths)
        self.register_buffer("sin_positions", table)
        # Six CDR flags, no extra CLS flag; CLS has its own learned AA embedding.
        self.projection = nn.Linear(width + 1 + 6, config.d_model, bias=False)

    def input_features(self, tokens: torch.Tensor) -> torch.Tensor:
        aa, position, length, cdr = tokens.unbind(dim=-1)
        sequence_features = self.aa_embedding(aa) + self.sin_positions[position]
        relative = (position.float() - 1) / (length - 1).clamp_min(1).float()
        relative = torch.where(length == 1, 0.5, relative)
        relative = torch.where(position == 0, 0.0, relative)
        cdr_flags = F.one_hot(cdr, num_classes=7)[..., 1:].float()
        return torch.cat((sequence_features, relative.unsqueeze(-1), cdr_flags), dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # No additional sqrt(d) scaling: the concatenation is linearly projected.
        return self.projection(self.input_features(tokens))


class SceptrHybridTcr(nn.Module):
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.embedder = HybridCdrEmbedder(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.heads,
            dim_feedforward=4 * config.d_model, dropout=config.dropout,
            activation=config.activation, batch_first=True, norm_first=False,
        )
        # Keep masks with internal NULL positions in the ordinary dense path.
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.mlm_head = nn.Linear(config.d_model, 20)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embedder(tokens), src_key_padding_mask=tokens[..., 0] == NULL)

    def get_cls_vector_representations_of(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode_tokens(tokens)[:, 0], p=2, dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_cls_vector_representations_of(tokens)

    def predict_masked_amino_acids(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(self.encode_tokens(tokens))


def sceptr_learning_rate(step: int, d_model: int, warmup: int = 10000) -> float:
    """Original AdamWithScheduling rule; call BEFORE update, starting at step 1."""
    if min(step, d_model, warmup) < 1:
        raise ValueError("step, d_model and warmup must be positive")
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def autocontrastive_loss(z: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Original DotProductLoss, for [all view-1; all view-2] and one positive."""
    n = z.shape[0]
    if n < 2 or n % 2 or temperature <= 0:
        raise ValueError("AC needs two views per TCR and positive temperature")
    logits = z @ z.T / temperature
    diagonal = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diagonal, logits.detach().min())
    logits = logits - logits.max(dim=1, keepdim=True).values
    positives = (torch.arange(n, device=z.device) + n // 2) % n
    denominator = (logits.exp() * (~diagonal)).sum(dim=1)
    return (denominator.log() - logits[torch.arange(n, device=z.device), positives]).mean()


def model_checkpoint(model: SceptrHybridTcr, state: dict, epoch: int, loss: float) -> dict:
    return {"format": FORMAT, "model_config": asdict(model.config),
            "amino_acids": AMINO_ACIDS, "epoch": epoch, "val_total_loss": loss,
            "model": state}


def load_model_checkpoint(path: str | Path, device: str = "cpu") -> SceptrHybridTcr:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT or payload.get("amino_acids") != AMINO_ACIDS:
        raise ValueError("Expected a tcr_sceptr_hybrid_v1 model checkpoint")
    model = SceptrHybridTcr(HybridConfig(**payload["model_config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device=device, dtype=torch.float32).eval()
