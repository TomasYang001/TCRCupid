"""Reaggregate fixed per-pMHC metrics from the four-generator comparison."""

import argparse
import json
from math import exp
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parent / "datasets" / "generation" / "test_group_metrics.json"
METRICS = ("srr", "novelty", "hit1", "hit10", "hit20", "hit50", "hit100")


def evaluate(data):
    groups = data["pMHC_groups"]
    models = data["models"]
    result = {}
    srr = {}
    for model, values in models.items():
        per_group = values["group_metrics"]
        if len(per_group) != groups:
            raise ValueError(f"{model} requires {groups} aligned pMHC groups")
        matrix = np.asarray([[row[key] for key in METRICS] for row in per_group], dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Nonfinite metrics for {model}")
        result[model] = dict(zip(METRICS, matrix.mean(axis=0).tolist()))
        nll = values["native_token_nll"]
        result[model]["native_token_ppl"] = exp(nll) if nll is not None else None
        srr[model] = matrix[:, 0]

    rng = np.random.default_rng(42)
    samples = rng.integers(0, groups, size=(10000, groups))
    paired = {}
    for baseline in models:
        if baseline == "TCRCupid":
            continue
        differences = srr["TCRCupid"] - srr[baseline]
        paired[baseline] = {
            "srr_difference": float(differences.mean()),
            "srr_ci95": np.quantile(differences[samples].mean(axis=1), (0.025, 0.975)).tolist(),
        }
    return {
        "test_records": data["test_records"],
        "contexts": data["contexts"],
        "pMHC_groups": groups,
        "metrics": result,
        "paired_srr": paired,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as handle:
        result = evaluate(json.load(handle))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
