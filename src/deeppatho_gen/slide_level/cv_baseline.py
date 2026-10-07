"""Patient-level repeated stratified cross-validation on pooled slide features.

Answers three questions with seconds-long runs (no GPU, no .svs reading):
  R0  Is the measurement trustworthy?  Repeated stratified K-fold over ALL
      patients (not one 9-positive test set) + bootstrap CI over patients,
      and a label-permutation negative control (should give AUC ~ 0.5).
  R1  Is there EGFR signal in frozen UNI-v2 slide features at all?
      (L2 logistic regression on mean-pooled features.)
  R2  Does the pipeline reproduce signals that are known to be learnable?
      The same procedure is run for the other genes in mutation_labels.csv
      (STK11, TP53, KRAS, FAT1, SETBP1, ALK); Coudray et al. 2018 report
      per-slide AUCs for the first five (reference column only -- their
      split/slide type/test size differ, so values are not directly comparable).

Unit of analysis is the PATIENT: slide vectors of the same case are averaged
(weighted by patch count for mean-pooling, element-wise max for max-pooling),
labels are per case (any sample mutated -> positive, as in the main pipeline),
and every fold splits by patient, so no patient is ever in train and test.

Usage:
    python -m deeppatho_gen.slide_level.cv_baseline \
        --pooled data/processed/uni_v2_luad_slide_pooled.npz \
        --labels data/raw/mutation_labels.csv --out-dir runs/slide_level/R0_R2_mean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Coudray et al. 2018, Table 1, per-slide AUC (average predicted probability).
COUDRAY_SLIDE_AUC = {"STK11": 0.856, "EGFR": 0.826, "SETBP1": 0.775, "TP53": 0.760, "FAT1": 0.750, "KRAS": 0.733}


def build_patient_matrix(pooled_path: str, pool: str):
    z = np.load(pooled_path)
    cases = z["case_ids"]
    n_patches = z["n_patches"].astype(np.float64)
    patients, rows = [], []
    for c in sorted(set(cases)):
        idx = np.where(cases == c)[0]
        parts = []
        if pool in ("mean", "meanmax"):
            parts.append((z["mean"][idx] * n_patches[idx, None]).sum(0) / n_patches[idx].sum())
        if pool in ("max", "meanmax"):
            parts.append(z["max"][idx].max(0))
        patients.append(c)
        rows.append(np.concatenate(parts))
    return np.array(patients), np.stack(rows).astype(np.float32)


def make_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-5, -1, 9), cv=StratifiedKFold(3, shuffle=True, random_state=seed),
            scoring="roc_auc", class_weight="balanced", max_iter=3000, n_jobs=1,
        ),
    )


def oof_scores(X, y, n_splits: int, seed: int) -> np.ndarray:
    """Out-of-fold probability for every patient from one K-fold repeat."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = make_model(seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def bootstrap_auc_ci(y, score, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], score[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_gene(X, y, n_splits, n_repeats, seed, n_jobs):
    oofs = np.stack(Parallel(n_jobs=n_jobs)(delayed(oof_scores)(X, y, n_splits, seed + r) for r in range(n_repeats)))
    aucs = [roc_auc_score(y, o) for o in oofs]
    aps = [average_precision_score(y, o) for o in oofs]
    lo, hi = bootstrap_auc_ci(y, oofs.mean(0))
    return {
        "n_patients": int(len(y)), "n_positive": int(y.sum()),
        "auc_mean": float(np.mean(aucs)), "auc_std_over_repeats": float(np.std(aucs)),
        "auc_bootstrap_ci95": [lo, hi], "pr_auc_mean": float(np.mean(aps)),
        "pr_auc_baseline": float(y.mean()), "auc_per_repeat": [float(a) for a in aucs],
    }


def permutation_control(X, y, n_splits, n_perm, seed, n_jobs):
    def one(p):
        yp = np.random.default_rng(seed + 1000 + p).permutation(y)
        return roc_auc_score(yp, oof_scores(X, yp, n_splits, seed + p))
    aucs = Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in range(n_perm))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "aucs": [float(a) for a in aucs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", default="data/processed/uni_v2_luad_slide_pooled.npz")
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--genes", nargs="+", default=["EGFR", "STK11", "TP53", "KRAS", "FAT1", "SETBP1", "ALK"])
    ap.add_argument("--pool", choices=["mean", "max", "meanmax"], default="mean")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=10, help="label-permutation repeats for the negative control")
    ap.add_argument("--perm-genes", nargs="+", default=["EGFR", "STK11"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out-dir", default="runs/slide_level/R0_R2_mean")
    args = ap.parse_args()

    patients, X = build_patient_matrix(args.pooled, args.pool)
    labels = pd.read_csv(args.labels).groupby("case_id")[args.genes].max()
    keep = [i for i, p in enumerate(patients) if p in labels.index]
    patients, X = patients[keep], X[keep]
    Y = labels.loc[patients]
    print(f"{len(patients)} patients with features+labels, feature dim {X.shape[1]} (pool={args.pool})", flush=True)

    results = {"config": vars(args), "n_patients": len(patients), "genes": {}, "permutation_control": {}}
    for g in args.genes:
        y = Y[g].values.astype(int)
        if y.sum() < 10:
            print(f"skip {g}: only {y.sum()} positives")
            continue
        r = evaluate_gene(X, y, args.n_splits, args.n_repeats, args.seed, args.n_jobs)
        r["coudray_slide_auc_reference"] = COUDRAY_SLIDE_AUC.get(g)
        results["genes"][g] = r
        print(f"{g:7s} pos={r['n_positive']:3d}  AUC={r['auc_mean']:.3f}  CI95={r['auc_bootstrap_ci95'][0]:.3f}-"
              f"{r['auc_bootstrap_ci95'][1]:.3f}  PR-AUC={r['pr_auc_mean']:.3f} (chance {r['pr_auc_baseline']:.3f})  "
              f"[Coudray {r['coudray_slide_auc_reference']}]", flush=True)
    for g in args.perm_genes:
        if g in args.genes:
            p = permutation_control(X, Y[g].values.astype(int), args.n_splits, args.n_perm, args.seed, args.n_jobs)
            results["permutation_control"][g] = p
            print(f"permuted-label control {g}: AUC={p['auc_mean']:.3f} +/- {p['auc_std']:.3f} (expect ~0.5)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([
        {"gene": g, "n_positive": r["n_positive"], "auc_mean": r["auc_mean"], "ci_lo": r["auc_bootstrap_ci95"][0],
         "ci_hi": r["auc_bootstrap_ci95"][1], "pr_auc": r["pr_auc_mean"], "pr_auc_chance": r["pr_auc_baseline"],
         "coudray_slide_auc": r["coudray_slide_auc_reference"]}
        for g, r in results["genes"].items()
    ]).to_csv(out / "results.csv", index=False)
    print(f"\nWrote {out / 'results.json'} and results.csv")


if __name__ == "__main__":
    main()
