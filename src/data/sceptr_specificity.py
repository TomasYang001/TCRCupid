"""Compact, untruncated SCEPTR CDR tokens with Stage-3 compartment semantics."""
import torch
from torch.nn.utils.rnn import pad_sequence

from .sceptr_hybrid import tokenise_cdrs
from .tokenizers import CdrTcrTokenizer
from ..models.tcr_sceptr_hybrid import AMINO_ACIDS


class SceptrSpecificityTokenizer:
    def __init__(self, max_cdr_length=100):
        self.max_cdr_length = max_cdr_length

    def encode_batch(self, samples):
        if not samples:
            raise ValueError("Cannot tokenize an empty batch")
        rows = []
        for sample in samples:
            cdrs = [str((sample.get(chain) or {}).get(cdr) or "").strip().upper()
                    for chain in ("cdr_alpha", "cdr_beta") for cdr in ("cdr1", "cdr2", "cdr3")]
            if not any(cdrs) or any(set(seq) - set(AMINO_ACIDS) for seq in cdrs):
                raise ValueError("SCEPTR requires at least one CDR and canonical amino acids")
            native = tokenise_cdrs(cdrs, self.max_cdr_length)
            native[:, 3] += 1  # CLS 0->1, CDRs 1..6->2..7; padding is added below.
            rows.append(native)
        tokens = pad_sequence(rows, batch_first=True, padding_value=0)
        return tokens, tokens[..., 0].eq(0)


def get_specificity_tcr_tokenizer(model_cfg, length_cfg):
    if model_cfg.tcr_stack_type == "sceptr_hybrid":
        return SceptrSpecificityTokenizer(model_cfg.tcr_max_compartment_pos)
    return CdrTcrTokenizer(length_cfg=length_cfg)
