"""SCEPTR hybrid encoder hand-off to the existing Stage-3 TcrBert interface."""
from dataclasses import asdict
import logging
from pathlib import Path

import torch
from torch import nn

from .tcr_sceptr_hybrid import AMINO_ACIDS, FORMAT, HybridConfig, SceptrHybridTcr
from .tcr_tower import TcrBert, TorchTransformerSelfAttentionStack

LOG = logging.getLogger(__name__)


class Stage3HybridEmbedder(nn.Module):
    """Keep Stage-3 compartment IDs; convert them only at the hybrid boundary.

    External tokens: [AA, one-based position, full CDR length, compartment].
    Compartments are PAD=0, CLS=1, CDR1a..CDR3b=2..7 so collate, masks and
    optional chain/focus logic retain their existing contract. Native SCEPTR
    compartments are CLS/PAD=0 and CDR1a..CDR3b=1..6.
    """
    def __init__(self, embedder, d_model):
        super().__init__()
        self.hybrid = embedder
        self.d_model = d_model

    def forward(self, tokens):
        native = tokens.clone()
        native[..., 3] = (native[..., 3] - 1).clamp_min(0)
        return self.hybrid(native)


class HybridAttentionStack(TorchTransformerSelfAttentionStack):
    def __init__(self, encoder, config):
        # Reuse the exact pretrained encoder, including disabled nested tensors.
        nn.Module.__init__(self)
        self._d_model = config.d_model
        self._num_layers = config.layers
        self.stack = encoder


class SceptrSpecificityTcr(TcrBert):
    def __init__(self, model_cfg):
        if model_cfg.tcr_encoder_ff != 4 * model_cfg.d_model:
            raise ValueError("SCEPTR hybrid requires tcr_encoder_ff == 4 * d_model")
        if model_cfg.tcr_vocab_size != 23 or model_cfg.tcr_n_compartments != 8:
            raise ValueError("SCEPTR Stage-3 requires 23 AA IDs and 8 external compartments")
        config = HybridConfig(d_model=model_cfg.d_model, layers=model_cfg.tcr_encoder_layers,
            heads=model_cfg.tcr_encoder_nhead, aa_embedding_dim=model_cfg.d_model,
            dropout=model_cfg.tcr_encoder_dropout, max_cdr_length=model_cfg.tcr_max_compartment_pos)
        native = SceptrHybridTcr(config)
        super().__init__(Stage3HybridEmbedder(native.embedder, config.d_model),
                         HybridAttentionStack(native.encoder, config))
        self.hybrid_config = config
        if model_cfg.tcr_finetune_last_only:
            self.set_fine_tuning_mode(True)

    def load_pretrained_checkpoint(self, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT or payload.get("amino_acids") != AMINO_ACIDS:
            raise ValueError("Expected a tcr_sceptr_hybrid_v1 checkpoint with matching AA IDs")
        source_config = HybridConfig(**payload["model_config"])
        if source_config != self.hybrid_config:
            raise ValueError(f"SCEPTR checkpoint architecture mismatch: {asdict(source_config)}")
        # Validate the whole source, including its pretraining-only MLM head.
        native = SceptrHybridTcr(source_config)
        native.load_state_dict(payload["model"], strict=True)
        self.token_embedder.hybrid.load_state_dict(native.embedder.state_dict(), strict=True)
        self.self_attention_stack.stack.load_state_dict(native.encoder.state_dict(), strict=True)
        LOG.info("Loaded all SCEPTR embedding/encoder tensors from %s (epoch=%s); "
                 "MLM prediction head is not part of specificity fine-tuning", path, payload.get("epoch"))
        return {"format": FORMAT, "epoch": payload.get("epoch"),
                "model_config": asdict(source_config), "encoder_tensors": len(self.state_dict())}
