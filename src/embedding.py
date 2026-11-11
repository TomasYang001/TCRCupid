"""Frozen paired-TCR embeddings and reference-based benchmark evaluation."""

import argparse
from pathlib import Path
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn.utils.rnn import pad_sequence

from .data.sceptr_hybrid import CDR_COLUMNS, tokenise_cdrs
from .models.tcr_sceptr_hybrid import load_model_checkpoint


def read_table(path, columns, **kwargs):
    return pd.read_csv(path, usecols=columns, keep_default_na=False, **kwargs)


def validate_ids(column, path):
    ids = column.to_numpy(dtype=str)
    if len(ids) == 0 or "" in ids or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Expected unique, nonempty TCR identifiers in {path}")
    return ids


def receptor_ids(path):
    return validate_ids(read_table(path, ["tcr_id"], dtype=str)["tcr_id"], path)


def encode(args):
    frame = read_table(args.receptors, ["tcr_id", *CDR_COLUMNS], dtype=str)
    ids = validate_ids(frame["tcr_id"], args.receptors)
    model = load_model_checkpoint(args.checkpoint, args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            chunk = frame.iloc[start : start + args.batch_size]
            rows = chunk[CDR_COLUMNS].itertuples(index=False, name=None)
            tokens = [tokenise_cdrs(row, model.config.max_cdr_length) for row in rows]
            batch = pad_sequence(tokens, batch_first=True)
            vectors.append(model(batch.to(args.device)).float().cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, tcr_id=ids, embedding=np.concatenate(vectors))


def load_vectors(path):
    with np.load(path, allow_pickle=False) as data:
        ids = data["tcr_id"]
        vectors = data["embedding"]
    if (
        ids.ndim != 1
        or ids.dtype.kind not in "US"
        or len(ids) == 0
        or len(np.unique(ids)) != len(ids)
        or vectors.ndim != 2
        or len(vectors) != len(ids)
        or not np.isfinite(vectors).all()
    ):
        raise ValueError(f"Invalid embedding archive: {path}")
    return ids.astype(str), vectors


def precision_at(labels, scores, k):
    """Average precision at the cutoff over tied candidates."""
    k = min(k, len(labels))
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > cutoff
    ties = scores == cutoff
    return (labels[above].sum() + (k - above.sum()) * labels[ties].mean()) / k


def retrieval_metrics(labels, scores):
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("Each query set must contain positives and negatives")
    metrics = {
        "AUROC": roc_auc_score(labels, scores),
        "AUPRC": average_precision_score(labels, scores),
        "PPVn": precision_at(labels, scores, positives),
    }
    for k in (10, 100):
        precision = precision_at(labels, scores, k)
        metrics[f"precision_at_{k}"] = precision
        metrics[f"recall_at_{k}"] = precision * min(k, len(labels)) / positives
    return metrics


def target_retrieval(target, membership, binders, id_to_row, vectors, batch_size, scratch):
    references = sorted(membership["tcr_id"].unique())
    if not references:
        raise ValueError(f"No reference splits for {target.target_id}")
    reference_columns = {key: i for i, key in enumerate(references)}
    if not set(references).issubset(binders):
        raise ValueError(f"Reference receptors must be target binders for {target.target_id}")
    reference_vectors = vectors[[id_to_row[key] for key in references]]
    labels = np.zeros(len(vectors), dtype=bool)
    labels[[id_to_row[key] for key in binders]] = True
    with TemporaryFile(dir=scratch) as cache:
        distances = np.memmap(
            cache, mode="w+", dtype=np.float32, shape=(len(vectors), len(references)), order="F"
        )
        try:
            for start in range(0, len(references), batch_size):
                end = min(start + batch_size, len(references))
                block = torch.cdist(vectors, reference_vectors[start:end]).float().cpu().numpy()
                distances[:, start:end] = block

            for (k, split_id), split in membership.groupby(["k", "split_id"], sort=True):
                keys = split["tcr_id"].tolist()
                if len(keys) != k or len(set(keys)) != k:
                    raise ValueError(f"Unexpected reference count for {target.target_id}, k={k}, {split_id}")
                query = np.ones(len(vectors), dtype=bool)
                query[[id_to_row[key] for key in keys]] = False
                scores = -distances[:, [reference_columns[key] for key in keys]].min(axis=1)
                yield {
                    "model": "hybrid",
                    "target_id": target.target_id,
                    "Peptide": target.Peptide,
                    "HLA": target.HLA,
                    "k": k,
                    **retrieval_metrics(labels[query], scores[query]),
                }
        finally:
            del distances


def retrieve(args):
    ids, embeddings = load_vectors(args.embeddings)
    expected = receptor_ids(args.benchmark / "01_tcrs.csv")
    if not np.array_equal(ids, expected):
        raise ValueError("The embedding IDs must match the complete benchmark and its row order")
    targets = read_table(args.benchmark / "03_targets_gt300.csv", ["target_id", "Peptide", "HLA"], dtype=str)
    membership = read_table(
        args.benchmark / "05_reference_membership.csv",
        ["target_id", "k", "split_id", "tcr_id"],
        dtype={"target_id": str, "k": int, "split_id": str, "tcr_id": str},
    )
    binders = read_table(args.benchmark / "04_target_binders.csv", ["target_id", "tcr_id"], dtype=str)
    if args.target:
        missing = set(args.target) - set(targets["target_id"])
        if missing:
            raise ValueError(f"Unknown target IDs: {sorted(missing)}")
        targets = targets.loc[targets["target_id"].isin(args.target)]
    id_to_row = {key: index for index, key in enumerate(ids)}
    vectors = torch.as_tensor(embeddings, dtype=torch.float64, device=args.device)
    rows = []
    for target in targets.itertuples(index=False):
        rows.extend(
            target_retrieval(
                target,
                membership.loc[membership["target_id"] == target.target_id],
                set(binders.loc[binders["target_id"] == target.target_id, "tcr_id"]),
                id_to_row,
                vectors,
                args.batch_size,
                args.scratch,
            )
        )
    if not rows:
        raise ValueError("No retrieval results were computed")
    scores = pd.DataFrame(rows)
    grouping = ["model", "target_id", "Peptide", "HLA", "k"]
    per_target = scores.groupby(grouping).agg(["mean", "std"])
    macro = scores.groupby(grouping).mean().groupby(["model", "k"]).mean()
    per_target.columns = ["_".join(name) for name in per_target.columns]
    args.output.mkdir(parents=True, exist_ok=True)
    per_target.to_csv(args.output / "retrieval_per_target_k.csv")
    macro.to_csv(args.output / "retrieval_macro_k.csv")


def geometry(args):
    ids, vectors = load_vectors(args.embeddings)
    cohort = receptor_ids(args.cohort)
    index = pd.Index(ids).get_indexer(cohort)
    if (index < 0).any() or len(cohort) < 2:
        raise ValueError("Expected at least two cohort receptors with embeddings")
    distances = {"hybrid": squareform(pdist(vectors[index])).astype(np.float32)}
    for name, path in args.baseline:
        path = Path(path)
        if name in distances or not np.array_equal(receptor_ids(path / "ids.csv"), cohort):
            raise ValueError(f"Duplicate model name or mismatched cohort for {name}")
        matrix = np.load(path / "distance.npy", allow_pickle=False)
        if matrix.shape != (len(cohort), len(cohort)) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid distance matrix for {name}")
        if not np.allclose(matrix, matrix.T, atol=1e-4) or not np.allclose(np.diag(matrix), 0):
            raise ValueError(f"Expected a symmetric distance matrix with a zero diagonal for {name}")
        distances[name] = matrix.astype(np.float32, copy=False)

    pairs = np.triu_indices(len(cohort), 1)
    pair_table = pd.DataFrame({name: matrix[pairs] for name, matrix in distances.items()})
    k = min(26, len(cohort) - 1)
    neighbors = {}
    for name, matrix in distances.items():
        ranking = matrix.copy()
        np.fill_diagonal(ranking, np.inf)
        neighbors[name] = np.argsort(ranking, axis=1, kind="stable")[:, :k]
    overlap = pd.DataFrame(index=distances.keys(), columns=distances.keys(), dtype=float)
    for left, x in neighbors.items():
        for right, y in neighbors.items():
            overlap.loc[left, right] = np.mean([len(set(a) & set(b)) / k for a, b in zip(x, y)])

    distance_correlations = pair_table.corr(method="spearman")
    if args.pgen:
        columns = ["log10_alpha", "log10_beta", "log10_product"]
        generation = read_table(args.pgen, ["tcr_id", *columns], dtype={"tcr_id": str})
        if not np.array_equal(generation["tcr_id"], cohort):
            raise ValueError("The Pgen table must match the geometry cohort and its row order")
        correlations = []
        for column in columns:
            values = pd.to_numeric(generation[column], errors="coerce").to_numpy()
            lower = np.minimum(values[pairs[0]], values[pairs[1]])
            valid = np.isfinite(lower)
            for name in distances:
                correlations.append(
                    {
                        "model": name,
                        "pgen": column,
                        "n_pairs": int(valid.sum()),
                        "spearman": spearmanr(pair_table[name][valid], lower[valid]).statistic,
                    }
                )
            pair_table[f"min_{column}"] = lower
    args.output.mkdir(parents=True, exist_ok=True)
    distance_correlations.to_csv(args.output / "geometry_distance_spearman.csv")
    overlap.to_csv(args.output / f"geometry_neighbor_overlap{k}.csv")
    if args.pgen:
        pd.DataFrame(correlations).to_csv(args.output / "geometry_pgen_correlations.csv", index=False)
    if args.write_pairs:
        pair_table.insert(0, "tcr_id_2", cohort[pairs[1]])
        pair_table.insert(0, "tcr_id_1", cohort[pairs[0]])
        pair_table.to_csv(args.output / "geometry_pairs.csv", index=False)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    embedding = commands.add_parser("encode", help="Encode six CDRs with the pretrained TCR tower")
    embedding.add_argument("--receptors", required=True, type=Path)
    embedding.add_argument("--checkpoint", required=True, type=Path)
    embedding.add_argument("--output", required=True, type=Path, help="NPZ embedding archive")
    embedding.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    embedding.add_argument("--batch-size", default=512, type=positive_int)
    embedding.set_defaults(run=encode)
    retrieval = commands.add_parser("retrieve", help="Evaluate the original nearest-reference splits")
    retrieval.add_argument("--embeddings", required=True, type=Path)
    retrieval.add_argument("--benchmark", required=True, type=Path)
    retrieval.add_argument("--target", action="append", help="Optional target ID, repeatable")
    retrieval.add_argument("--output", required=True, type=Path, help="Output directory")
    retrieval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    retrieval.add_argument("--batch-size", default=256, type=positive_int, help="Distance block references")
    retrieval.add_argument("--scratch", type=Path, help="Temporary distance storage (up to 6 GB)")
    retrieval.set_defaults(run=retrieve)
    repertoire = commands.add_parser("geometry", help="Compare receptor distances on an ordered cohort")
    repertoire.add_argument("--embeddings", required=True, type=Path)
    repertoire.add_argument("--cohort", required=True, type=Path, help="Ordered geometry cohort CSV")
    repertoire.add_argument("--baseline", action="append", nargs=2, default=[], metavar=("NAME", "DIR"),
                            help="Archived comparator with ids.csv and distance.npy, repeatable")
    repertoire.add_argument("--pgen", type=Path, help="Archived OLGA probabilities for the cohort")
    repertoire.add_argument("--output", required=True, type=Path, help="Output directory")
    repertoire.add_argument("--write-pairs", action="store_true", help="Write pair-level plot data")
    repertoire.set_defaults(run=geometry)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
