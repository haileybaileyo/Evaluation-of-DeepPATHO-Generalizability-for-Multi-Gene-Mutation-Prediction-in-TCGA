"""R0b: the same pooled-feature logistic regression as cv_baseline.py, but
evaluated on the EXACT fixed patient split the patch-level experiments used
(stratified 70/15/15, split_seed 42 -> train 329 / val 71 / test 71 patients).

Why: R1's 0.687 comes from repeated cross-validation over all patients, while
the patch-level results (Inception 0.504, UNI 0.615, ...) come from one fixed
test split with only 11 positive slides. This script removes that difference
so the two can be compared on the same test patients, with the same
protocol as the patch pipeline: train on train, early/hyper-parameter choice
inside train only, decision threshold fit on VAL only (F1-maximizing, via
metrics.find_best_threshold), report test at that threshold and at 0.5.

Two views of the same fitted model:
  patient-level : patient-averaged features -> one score per patient
  slide-level   : the same model applied to each slide's mean feature, so the
                  test set is directly comparable to the 86-slide patch test.

Usage:
    python -m deeppatho_gen.slide_level.fixed_split_logreg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..dataset import stratified_case_level_split
from ..metrics import binary_metrics, find_best_threshold
from .cv_baseline import build_patient_matrix, make_model


def get_fixed_split(coords_csv: str, labels_csv: str, gene: str, train_frac=0.70, val_frac=0.15, seed=42):
    """Same case set and call as build_index() in the training scripts
    (coords inner-joined with labels -> stratified_case_level_split)."""
    coords = pd.read_csv(coords_csv, usecols=["case_id"])
    labels = pd.read_csv(labels_csv).groupby("case_id")[gene].max()
    case_ids = sorted(set(coords.case_id) & set(labels.index))
    case_labels = {c: int(labels[c]) for c in case_ids}
    split = stratified_case_level_split(case_ids, case_labels, train_frac=train_frac, val_frac=val_frac, seed=seed)
    return split, case_labels


def _fmt(m):
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    n = tn + fp + fn + tp
    return (f"AUC {m['roc_auc']:.3f} | P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} | "
            f"pred+ {100 * (tp + fp) / n:.0f}% (true+ {100 * (tp + fn) / n:.0f}%) | TN/FP/FN/TP {tn}/{fp}/{fn}/{tp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled", default="data/processed/uni_v2_luad_slide_pooled.npz")
    ap.add_argument("--coords", default="data/processed/patch_coords_tumor_only.csv")
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--n-seeds", type=int, default=5, help="inner-CV seeds for the C search (data/split fixed)")
    ap.add_argument("--out-dir", default="runs/slide_level/R0b_fixed_split_logreg")
    args = ap.parse_args()

    split, case_labels = get_fixed_split(args.coords, args.labels, args.gene)
    z = np.load(args.pooled)
    patients, X = build_patient_matrix(args.pooled, "mean")
    keep = [i for i, p in enumerate(patients) if p in split]
    patients, X = patients[keep], X[keep]
    y = np.array([case_labels[p] for p in patients])
    part = np.array([split[p] for p in patients])

    slide_case = z["case_ids"]
    slide_keep = np.array([c in split for c in slide_case])
    S = z["mean"][slide_keep]
    s_case = slide_case[slide_keep]
    s_y = np.array([case_labels[c] for c in s_case])
    s_part = np.array([split[c] for c in s_case])

    print(f"{args.gene}: patients with features -> " + ", ".join(
        f"{k} {int((part == k).sum())} ({int(y[part == k].sum())}+)" for k in ["train", "val", "test"]))
    print("slides with features -> " + ", ".join(
        f"{k} {int((s_part == k).sum())} ({int(s_y[s_part == k].sum())}+)" for k in ["train", "val", "test"]))

    tr = part == "train"
    results = {"gene": args.gene, "seeds": []}
    for seed in range(args.n_seeds):
        model = make_model(seed).fit(X[tr], y[tr])
        r = {"seed": seed}
        for view, (XX, yy, pp) in {"patient": (X, y, part), "slide": (S, s_y, s_part)}.items():
            score = model.predict_proba(XX)[:, 1]
            v, t = pp == "val", pp == "test"
            thr = find_best_threshold(yy[v], score[v])
            r[view] = {
                "val_auc": float(roc_auc_score(yy[v], score[v])), "threshold_fit_on_val": float(thr),
                "test_at_val_threshold": binary_metrics(yy[t], score[t], thr),
                "test_at_0.5": binary_metrics(yy[t], score[t], 0.5),
            }
        results["seeds"].append(r)
        print(f"\n--- seed {seed} ---")
        for view in ["slide", "patient"]:
            rv = r[view]
            print(f"[{view}] val AUC {rv['val_auc']:.3f}, threshold(val) {rv['threshold_fit_on_val']:.3f}")
            print("   test @val-thr:", _fmt(rv["test_at_val_threshold"]))
            print("   test @0.5    :", _fmt(rv["test_at_0.5"]))

    for view in ["slide", "patient"]:
        a = [s[view]["test_at_val_threshold"]["roc_auc"] for s in results["seeds"]]
        results[f"{view}_test_auc_mean"], results[f"{view}_test_auc_std"] = float(np.mean(a)), float(np.std(a))
        print(f"\n{view}-level TEST AUC over {args.n_seeds} inner-CV seeds: {np.mean(a):.3f} +/- {np.std(a):.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
