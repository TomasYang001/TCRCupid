"""Evaluate trained TCRCupid checkpoints on labelled specificity benchmarks."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data.collate import TcrEpiCollator
from .data.dataset import TCRPMHCDataset
from .data.full_cohort import FullCohortDataset
from .data.sceptr_specificity import get_specificity_tcr_tokenizer
from .data.tokenizers import get_pmhc_tokenizer
from .models.pmhc_tower import HFESM2Backbone
from .models.specificity_model import TCREpitopeSpecificityModel
from .scripts.normalize_specificity_scores import normalize_scores, score_metrics, select_alpha
from .specificity_metrics import evaluate_predictions, summarize_folds


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "antigen_tokens", "mhc_tokens", "tcr_tokens",
    "antigen_pad_mask", "mhc_pad_mask", "tcr_pad_mask",
    "antigen_valid_mask", "mhc_valid_mask", "tcr_valid_mask",
)


def build_model(config, esm_config):
    """Construct the original architecture using the local ESM2 configuration."""
    dtype = getattr(torch, config.pmhc_backbone_dtype)
    esm_path = esm_config.resolve().as_posix()
    antigen = HFESM2Backbone(esm_path, torch_dtype=dtype, init_from_config_only=True)
    mhc = HFESM2Backbone(esm_path, torch_dtype=dtype, init_from_config_only=True)
    antigen.config.use_cache = False
    mhc.config.use_cache = False
    return TCREpitopeSpecificityModel.from_config(
        config, ant_backbone=antigen, mhc_backbone=mhc
    )


def load_checkpoint(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(payload.get("state_dict", payload), strict=True)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    dtype = torch.bfloat16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
    predictions = []
    for batch in loader:
        model_input = {key: batch[key].to(device) for key in INPUTS}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            logits = model(**model_input)
        probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        for label, probability, meta in zip(batch["label"].tolist(), probabilities, batch["meta"]):
            predictions.append({
                "y_true": int(label),
                "y_prob": probability,
                "Peptide": meta.get("Peptide", ""),
                "HLA": meta.get("HLA", ""),
                "negative_type": meta.get("negative_type", ""),
                "hla_match_level": meta.get("hla_match_level", ""),
            })
    return pd.DataFrame.from_records(predictions)


def run_split(model, cfg, collate, data, device, batch_size, *, full_cohort=False):
    dataset_class = FullCohortDataset if full_cohort else TCRPMHCDataset
    dataset = dataset_class(data, length_cfg=cfg.lengths)
    if not len(dataset):
        raise ValueError(f"No benchmark rows passed input validation in {data}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    frame = predict(model, loader, device, cfg.train.amp)
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame


def evaluate_hla_normalization(test, validation, center_alpha):
    fit = None
    if center_alpha is None:
        fit = select_alpha(
            validation, group_by="HLA", input_col="y_prob", alpha_values=np.arange(101) * 0.02
        )
        center_alpha = fit["alpha"]
    results = {"center_alpha": center_alpha, "validation_fit": fit}
    for mode in ("group_center", "group_zscore", "group_rank"):
        scores = normalize_scores(test, mode=mode, group_by="HLA", alpha=center_alpha)
        results[mode] = {
            key: value if np.isfinite(value) else None
            for key, value in score_metrics(test, scores).items()
        }
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("strict", "strict_with_10x", "cv"), default="strict")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/specificity.yaml")
    parser.add_argument("--esm-config", type=Path, default=ROOT / "configs/esm2_150M")
    parser.add_argument("--asset-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Override the checkpoint for a single test set.")
    parser.add_argument("--data", type=Path, help="Override the labelled CSV for a single test set.")
    parser.add_argument("--hla-normalize", action="store_true", help="Evaluate strict-split HLA score normalization.")
    parser.add_argument("--validation-data", type=Path, help="Override the labelled validation CSV for HLA centering.")
    parser.add_argument(
        "--hla-center-alpha", type=float, help="Use a fixed centering factor instead of refitting on validation."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", type=Path, help="Save predictions and metrics to this directory.")
    args = parser.parse_args()
    if args.split == "cv" and (args.checkpoint or args.data):
        parser.error("--split cv reads fold_00 through fold_04 from --weights-dir and --data-dir.")
    if args.hla_normalize and args.split != "strict":
        parser.error("--hla-normalize uses the strict split and its validation set.")
    if args.hla_center_alpha is not None and not np.isfinite(args.hla_center_alpha):
        parser.error("--hla-center-alpha must be finite.")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive.")

    names = [f"fold_{index:02d}" for index in range(5)] if args.split == "cv" else [args.split]
    data_dir = args.data_dir or args.asset_root / "datasets/specificity"
    weights_dir = args.weights_dir or args.asset_root / "weights/specificity"
    datasets = [args.data or data_dir / f"{name}.csv" for name in names]
    checkpoints = [args.checkpoint or weights_dir / f"{name}.pt" for name in names]
    validation_data = args.validation_data or data_dir / "strict_validation.csv"
    for path in [*datasets, *checkpoints]:
        if not path.is_file():
            parser.error(f"Required file not found: {path}")
    fit_hla = args.hla_normalize and args.hla_center_alpha is None
    if fit_hla and not validation_data.is_file():
        parser.error(f"Required validation file not found: {validation_data}")

    cfg = ExperimentConfig.from_yaml(args.config)
    torch.manual_seed(cfg.train.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg.model, args.esm_config).to(device)
    collate = TcrEpiCollator(
        get_pmhc_tokenizer(args.esm_config.resolve().as_posix()),
        get_specificity_tcr_tokenizer(cfg.model, cfg.lengths),
    )
    batch_size = args.batch_size or cfg.train.batch_size

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    folds = []
    for name, data, checkpoint in zip(names, datasets, checkpoints):
        load_checkpoint(model, checkpoint)
        frame = run_split(
            model, cfg, collate, data, device, batch_size, full_cohort=name == "strict_with_10x"
        )
        results[name] = evaluate_predictions(frame)
        if args.split == "cv":
            folds.append(frame)
        if args.hla_normalize:
            validation = (
                run_split(model, cfg, collate, validation_data, device, batch_size) if fit_hla else None
            )
            results["hla_normalization"] = evaluate_hla_normalization(
                frame, validation, args.hla_center_alpha
            )
        if args.output_dir:
            frame.to_csv(args.output_dir / f"{name}_predictions.csv", index=False)
    if args.split == "cv":
        results["cv_mean_sd"] = summarize_folds(results.values())
        results["cv_pooled_oof"] = evaluate_predictions(pd.concat(folds, ignore_index=True))

    serialized = json.dumps(results, indent=2, allow_nan=False)
    if args.output_dir:
        (args.output_dir / "metrics.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
