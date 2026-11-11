"""Compact six-CDR tokenisation and original SCEPTR MLM/AC augmentation rules."""
from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ..models.tcr_sceptr_hybrid import AMINO_ACIDS, NULL, MASK, CLS
from .dataset import _normalize_columns

CDR_COLUMNS = [f"CDR{i}{chain}" for chain in "ab" for i in (1, 2, 3)]
AA_IDS = {aa: i + 3 for i, aa in enumerate(AMINO_ACIDS)}


class CdrCsvDataset(Dataset):
    """Accept the existing prepared train.csv/val.csv; never truncate or filter rows."""

    def __init__(self, path: Path, max_cdr_length: int = 100):
        frame = _normalize_columns(pd.read_csv(path, dtype=str, keep_default_na=False))
        frame = frame[CDR_COLUMNS].copy()
        for name in CDR_COLUMNS:
            frame[name] = frame[name].str.strip().str.upper()
            valid = frame[name].str.fullmatch(f"[{AMINO_ACIDS}]+")
            valid &= frame[name].str.len() <= max_cdr_length
            if not valid.all():
                bad_row = int((~valid).to_numpy().nonzero()[0][0]) + 2
                raise ValueError(f"{path}:{bad_row}: invalid/empty/over-length {name}; no rows dropped")
        if len(frame) < 2:
            raise ValueError(f"{path}: AC requires at least two TCRs")
        self.frame = frame
        self.rows = frame.to_numpy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return tuple(self.rows[index])


def tokenise_cdrs(cdrs, max_cdr_length: int = 100) -> torch.Tensor:
    if len(cdrs) != 6:
        raise ValueError("Expected six CDR strings in alpha1/2/3, beta1/2/3 order")
    rows = [[CLS, 0, 0, 0]]
    for compartment, sequence in enumerate(cdrs, start=1):
        if len(sequence) > max_cdr_length:
            raise ValueError("CDR exceeds max_cdr_length; refusing to truncate")
        rows.extend([AA_IDS[aa], pos, len(sequence), compartment]
                    for pos, aa in enumerate(sequence, start=1))
    return torch.tensor(rows, dtype=torch.long)


def mask_for_mlm(tokens: torch.Tensor, rng=random) -> tuple[torch.Tensor, torch.Tensor]:
    # Original SCEPTR chooses ceil(15% * L), without replacement, excluding CLS.
    count = tokens.shape[0] - 1
    if count == 0:
        raise ValueError("MLM requires at least one amino acid")
    selected = rng.sample(range(1, count + 1), math.ceil(count * 0.15))
    masked = tokens.clone()
    targets = torch.full((len(tokens),), -100, dtype=torch.long)
    for index in selected:
        original = int(tokens[index, 0])
        targets[index] = original - 3  # 20-class amino-acid target, excluding specials.
        draw = rng.random()
        if draw < 0.1:
            masked[index, 0] = rng.choice([i for i in range(3, 23) if i != original])
        elif 1.0 - draw < 0.1:
            pass
        else:
            masked[index, 0] = MASK
    return masked, targets


class SceptrCollator:
    def __init__(self, max_cdr_length: int = 100, *, include_ac=True, include_mlm=True,
                 chain_drop=0.5):
        if not (include_ac or include_mlm) or not 0 <= chain_drop <= 1:
            raise ValueError("Need at least one objective and a valid chain-drop probability")
        self.max_cdr_length = max_cdr_length
        self.include_ac, self.include_mlm = include_ac, include_mlm
        self.chain_drop = chain_drop

    def augment(self, cdrs, rng=random):
        parts = list(cdrs)
        if any(parts[:3]) and any(parts[3:]) and rng.random() < self.chain_drop:
            start = 0 if rng.random() < 0.5 else 3
            parts[start:start + 3] = ["", "", ""]
        tokens = tokenise_cdrs(parts, self.max_cdr_length)
        # Drop only AA IDs. Position/length/CDR fields of retained tokens stay fixed.
        count = tokens.shape[0] - 1
        selected = rng.sample(range(1, count + 1), math.ceil(count * 0.2))
        tokens[selected, 0] = NULL
        return tokens

    def __call__(self, batch):
        # Consume the same two seeds for every objective. Removing a branch or chain
        # dropout must not change MLM masks in the other controlled experiments.
        ac_rng = random.Random(random.getrandbits(64))
        mlm_rng = random.Random(random.getrandbits(64))
        result = {}
        if self.include_ac:
            views = [self.augment(cdrs, ac_rng) for _ in range(2) for cdrs in batch]
            result["views"] = pad_sequence(views, batch_first=True, padding_value=NULL)
        if self.include_mlm:
            mlm = [mask_for_mlm(tokenise_cdrs(cdrs, self.max_cdr_length), mlm_rng) for cdrs in batch]
            result.update(mlm_tokens=pad_sequence([x[0] for x in mlm], batch_first=True, padding_value=NULL),
                          mlm_targets=pad_sequence([x[1] for x in mlm], batch_first=True, padding_value=-100))
        return result
