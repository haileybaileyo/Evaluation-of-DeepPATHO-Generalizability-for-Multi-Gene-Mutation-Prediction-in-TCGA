"""Slide-level evaluation: per-gene AUC with bootstrap 95% CIs, reproducing
Coudray et al.'s Table 1 format so the new model can be compared cell-by-cell
against the published Inception v3 numbers.

Usage:
    python -m deeppatho_gen.evaluate \
        --patch-scores runs/phase1_gene_mutation/test_patch_scores.csv \
        --labels data/raw/mutation_labels.csv \
        --genes-config configs/genes.yaml \
        --out runs/phase1_gene_mutation/slide_auc_table.csv

`patch-scores` is expected to already contain one row per test-set patch
with columns [file_id, case_id, <gene>_prob for each gene] -- produced by
running the trained model over the held-out test split (see
generate_patch_scores() below, called from a checkpoint at eval time).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from .mil.aggregation import aggregate_slide_scores


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_iter: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate + 95% CI via the bootstrap (Coudray et al.'s method:
    1000 resamples of the test slides)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = roc_auc_score(y_true, y_score)

    boot_aucs = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        y_t, y_s = y_true[idx], y_score[idx]
        if len(np.unique(y_t)) < 2:
            continue  # skip resamples with no positive or no negative slides
        boot_aucs.append(roc_auc_score(y_t, y_s))

    lo, hi = np.percentile(boot_aucs, [2.5, 97.5]) if boot_aucs else (np.nan, np.nan)
    return point, lo, hi


def build_auc_table(patch_scores: pd.DataFrame, labels: pd.DataFrame, gene_cols: list[str]) -> pd.DataFrame:
    slide_scores = aggregate_slide_scores(patch_scores, slide_col="file_id", prob_cols=gene_cols)

    case_by_file = patch_scores[["file_id", "case_id"]].drop_duplicates()
    slide_scores = slide_scores.merge(case_by_file, on="file_id")

    labels_by_case = labels.groupby("case_id")[gene_cols].max()

    rows = []
    for method, sub in slide_scores.groupby("method"):
        sub = sub.merge(labels_by_case, on="case_id", suffixes=("_pred", "_true"))
        for gene in gene_cols:
            y_true = sub[f"{gene}_true"].values
            y_score = sub[f"{gene}_pred"].values
            if len(np.unique(y_true)) < 2:
                rows.append({"gene": gene, "method": method, "n_slides": len(sub),
                             "n_positive": int(y_true.sum()), "auc": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                continue
            point, lo, hi = bootstrap_auc_ci(y_true, y_score)
            rows.append({
                "gene": gene, "method": method, "n_slides": len(sub),
                "n_positive": int(y_true.sum()), "auc": round(point, 3),
                "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            })
    return pd.DataFrame(rows).sort_values(["method", "auc"], ascending=[True, False])


# Published Coudray et al. 2018 Table 1 numbers (per-slide AUC, average-probability
# aggregation), kept here so evaluate.py can print a side-by-side diff for free.
COUDRAY_2018_TABLE1_MEAN_PROB_AUC = {
    "STK11": 0.856, "EGFR": 0.826, "SETBP1": 0.775, "TP53": 0.760,
    "FAT1": 0.750, "KRAS": 0.733, "KEAP1": 0.675, "LRP1B": 0.656,
    "FAT4": 0.642, "NF1": 0.640,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-scores", required=True)
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--genes-config", default="configs/genes.yaml")
    ap.add_argument("--out", default="runs/slide_auc_table.csv")
    args = ap.parse_args()

    genes_cfg = yaml.safe_load(Path(args.genes_config).read_text())
    gene_cols = list(genes_cfg["phase1_genes"])

    patch_scores = pd.read_csv(args.patch_scores)
    labels = pd.read_csv(args.labels)

    table = build_auc_table(patch_scores, labels, gene_cols)

    table["coudray_2018_auc"] = table.apply(
        lambda r: COUDRAY_2018_TABLE1_MEAN_PROB_AUC.get(r["gene"]) if r["method"] == "mean_probability" else np.nan,
        axis=1,
    )
    table["delta_vs_coudray"] = table["auc"] - table["coudray_2018_auc"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
