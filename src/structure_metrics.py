"""Recompute cross-target statistics from saved AlphaFold 3 results."""

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = ("native", "cupid", "translate", "diff", "gratcr")
METRICS = (
    "cdr3b_ca_plddt",
    "beta_peptide_iptm",
    "pae_cdr3b_peptide_mean",
    "iptm",
    "cdr3b_ca_rmsd",
)
LOWER_IS_BETTER = {"pae_cdr3b_peptide_mean", "cdr3b_ca_rmsd"}
SUMMARIES = ("median", "best5_beta_peptide_iptm")
DATA = Path(__file__).resolve().parent / "datasets" / "structure"


def summarize(rows):
    """Use each target once per model and omit unavailable metric values."""
    targets = sorted({row["pdb"] for row in rows})
    indexed = {(row["pdb"], row["model"], row["summary"]): row for row in rows}
    expected = {(pdb, model, summary) for pdb in targets for model in MODELS for summary in SUMMARIES}
    if not targets or len(indexed) != len(rows) or indexed.keys() != expected:
        raise ValueError("Expected one median and one best-by-ipTM row for each target and model")

    across_targets = []
    for summary in SUMMARIES:
        for model in MODELS:
            for metric in METRICS:
                values = [indexed[pdb, model, summary][metric] for pdb in targets]
                available = [value for value in values if value is not None]
                across_targets.append({
                    "summary": summary,
                    "model": model,
                    "metric": metric,
                    "n_targets": len(available),
                    "median": float(np.median(available)) if available else None,
                    "mean": float(np.mean(available)) if available else None,
                })

    paired_differences = []
    for baseline in MODELS[2:]:
        for metric in METRICS:
            differences = []
            for pdb in targets:
                cupid = indexed[pdb, "cupid", "median"][metric]
                other = indexed[pdb, baseline, "median"][metric]
                if cupid is not None and other is not None:
                    differences.append(cupid - other)
            sign = -1 if metric in LOWER_IS_BETTER else 1
            benefit = sign * np.asarray(differences, dtype=float)
            count = len(benefit)
            row = {
                "baseline": baseline,
                "metric": metric,
                "paired_targets": count,
                "median_cupid_minus_baseline": float(np.median(differences)) if count else None,
                "mean_benefit": float(benefit.mean()) if count else None,
                "wins": int((benefit > 1e-9).sum()),
                "ties": int((np.abs(benefit) <= 1e-9).sum()),
                "losses": int((benefit < -1e-9).sum()),
                "benefit_ci95": None,
            }
            if count:
                rng = np.random.default_rng(42)
                boot = benefit[rng.integers(0, count, size=(10000, count))].mean(axis=1)
                row["benefit_ci95"] = np.quantile(boot, (0.025, 0.975)).tolist()
            paired_differences.append(row)

    return {
        "targets": targets,
        "n_targets": len(targets),
        "n_af3_structures_underlying_summaries": sum(
            int(row["n_candidates"]) for row in rows if row["summary"] == "median"
        ),
        "across_targets": across_targets,
        "paired_differences": paired_differences,
        "target_summaries": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("unseen16", "all20", "tumor_cases"), default="unseen16")
    parser.add_argument("--input", type=Path, help="Override the AF3 per-target summary JSON")
    args = parser.parse_args()
    default_name = "tumor_cases.json" if args.cohort == "tumor_cases" else "benchmark_metrics.json"
    with (args.input or DATA / default_name).open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data["summaries"]
    if args.cohort == "unseen16":
        targets = set(data["unseen_targets"])
        rows = [row for row in rows if row["pdb"] in targets]
    print(json.dumps(summarize(rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
