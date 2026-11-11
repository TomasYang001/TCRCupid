"""Evaluate a saved TCRCupid pMHC presentation model on the EL test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import LengthConfig, ModelConfig
from .data.collate import PMHCBindingCollator
from .data.dataset import PMHCBindingDataset
from .data.tokenizers import get_pmhc_tokenizer
from .models.pmhc_tower import HFESM2Backbone, PMHCBindingModel
from .training.metrics import compute_binary_metrics


ROOT = Path(__file__).resolve().parents[1]
SIZES = ("8M", "35M", "150M", "650M", "3B")


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    assets = args.asset_root.resolve()
    data = args.data or assets / "datasets" / "presentation" / "test.csv"
    weights = args.weights or assets / "weights" / "presentation" / args.size
    tower = weights / "pmhc_tower.pt"
    head = weights / "pmhc_binding_head.pt"
    for path in (data, tower, head):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation asset is missing: {path}")

    device = torch.device(args.device)
    dataset = PMHCBindingDataset(data, length_cfg=LengthConfig(peptide=15, mhc_pseudo=40))
    if not len(dataset):
        raise ValueError("No valid pMHC rows in the supplied test set")

    backbone = (ROOT / "configs" / f"esm2_{args.size}").as_posix()
    config = ModelConfig(
        antigen_backbone_name=backbone,
        mhc_backbone_name=backbone,
        d_model=512,
        n_heads=8,
        dropout=0.1,
        pmhc_inner_cross_layers=1,
        pmhc_inner_cross_bidirectional=False,
        pmhc_token_fcn_hidden_mult=4,
        pmhc_backbone_dtype="bfloat16" if args.precision == "bf16" else "float32",
        use_pmhc_chain_embedding=False,
    )
    dtype = getattr(torch, config.pmhc_backbone_dtype)
    model = PMHCBindingModel(
        config,
        ant_backbone=HFESM2Backbone(backbone, torch_dtype=dtype, init_from_config_only=True),
        mhc_backbone=HFESM2Backbone(backbone, torch_dtype=dtype, init_from_config_only=True),
        head_dropout=0.2,
        head_hidden_mult=2,
    )
    checkpoint = torch.load(tower, map_location="cpu", weights_only=True, mmap=True)
    model.pmhc_tower.load_state_dict(checkpoint["pmhc_tower"], strict=True)
    del checkpoint
    model.pmhc_head.load_state_dict(torch.load(head, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()

    tokenizer = get_pmhc_tokenizer(backbone)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=PMHCBindingCollator(tokenizer))
    labels, scores = [], []
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            logits = model(**batch)
        scores.extend(logits.float().sigmoid().cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())

    return {"size": args.size, "n": len(labels), "n_positive": int(sum(labels)), **compute_binary_metrics(labels, scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=SIZES, default="150M")
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data", type=Path, help="Override the EL test CSV")
    parser.add_argument("--weights", type=Path, help="Directory containing the matching tower and head")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
