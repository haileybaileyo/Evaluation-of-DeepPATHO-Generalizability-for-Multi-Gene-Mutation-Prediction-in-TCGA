"""Patch-level vs patient-level training, everything else identical.

Question: is training on individual patches (each patch inherits the patient's
label) the reason EGFR looks weak, compared with patient-level models?

Same UNI2-h features, same label file, same patients, and the SAME repeated
stratified 5-fold splits as cv_baseline.py / abmil_cv.py (patients sorted,
StratifiedKFold(5, shuffle, random_state=42+repeat)). Only the unit of training differs:

  patch   : a linear or small-MLP head is trained on individual patches (up to
            --cap per slide, label copied from the patient, early stopping on a
            stratified 15% hold-out of the TRAINING patients). The score of a
            held-out patient is the mean predicted probability over all of
            that patient's patches (the Stage A aggregation).
  patient : reference numbers from cv_baseline.py (mean-pooled logistic
            regression) and abmil_cv.py (attention MIL), read from their
            results.json when present.

Usage:
    python -m deeppatho_gen.slide_level.patch_vs_patient_cv --features-dir <.../uni_v2/TCGA_LUAD> \
        --labels data/raw/egfr_variants/act_all.csv --tag act_all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .cv_baseline import bootstrap_auc_ci
from .patch_head_uni2 import predict, train_head


def load_patches(features_dir, pooled, patients, cap, seed):
    z = np.load(pooled)
    rng = np.random.default_rng(seed)
    keep = set(patients)
    X, case = [], []
    for sid, cid in zip(z["slide_ids"], z["case_ids"]):
        if cid not in keep:
            continue
        with h5py.File(Path(features_dir) / f"{sid}.h5", "r") as h:
            f = h["features"][:]
        if len(f) > cap:
            f = f[np.sort(rng.choice(len(f), cap, replace=False))]
        X.append(f.astype(np.float16))
        case += [cid] * len(f)
    return np.concatenate(X), np.array(case)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--pooled", default="data/processed/uni_v2_luad_slide_pooled.npz")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--gene", default="EGFR")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=800)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/slide_level/R6_patch_vs_patient")
    args = ap.parse_args()
    dev = torch.device("cuda")

    lab = pd.read_csv(args.labels).groupby("case_id")[args.gene].max()
    z = np.load(args.pooled)
    patients = np.array(sorted(c for c in set(z["case_ids"]) if c in lab.index))  # same order as cv_baseline
    y = lab.loc[patients].values.astype(int)
    t0 = time.time()
    X, case = load_patches(args.features_dir, args.pooled, patients, args.cap, 0)
    print(f"[{args.tag}] {len(patients)} patients ({y.sum()} positive), {len(X):,} patches, loaded in {time.time() - t0:.0f}s", flush=True)
    Xg = torch.from_numpy(X).to(dev)
    case_idx = {c: np.where(case == c)[0] for c in patients}
    pos_of = {c: i for i, c in enumerate(patients)}
    yp = torch.from_numpy(np.array([y[pos_of[c]] for c in case], dtype=np.float32)).unsqueeze(1).to(dev)

    def patch_ids(cs):
        return torch.from_numpy(np.concatenate([case_idx[c] for c in cs])).to(dev)

    res = {}
    for kind in ["linear", "mlp"]:
        oofs, aucs = [], []
        for r in range(args.n_repeats):
            oof = np.zeros(len(patients))
            for f, (tr, te) in enumerate(StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed + r).split(patients, y)):
                fit_p, val_p = train_test_split(patients[tr], test_size=0.15, random_state=args.seed + r, stratify=y[tr])
                fi, vi, ti = patch_ids(fit_p), patch_ids(val_p), patch_ids(patients[te])
                mu = Xg[fi].float().mean(0); sd = Xg[fi].float().std(0) + 1e-6
                net, _, _ = train_head(kind, Xg[fi], yp[fi], Xg[vi], yp[vi], mu, sd, args.seed + 100 * r + f, dev)
                prob = predict(net, Xg[ti], mu, sd, dev)[:, 0]
                owner = case[ti.cpu().numpy()]
                s = pd.Series(prob).groupby(owner).mean()
                oof[te] = s.loc[patients[te]].values
            oofs.append(oof); aucs.append(roc_auc_score(y, oof))
            print(f"[{args.tag}/{kind}] repeat {r}: patient-level OOF AUC {aucs[-1]:.3f}", flush=True)
        lo, hi = bootstrap_auc_ci(y, np.mean(oofs, 0))
        res[kind] = {"auc_mean": float(np.mean(aucs)), "auc_std_over_repeats": float(np.std(aucs)), "auc_bootstrap_ci95": [lo, hi],
                     "pr_auc_mean": float(np.mean([average_precision_score(y, o) for o in oofs])), "pr_auc_baseline": float(y.mean()),
                     "auc_per_repeat": [float(a) for a in aucs], "n_patients": len(y), "n_positive": int(y.sum())}
        print(f"=== [{args.tag}] PATCH-level {kind}: AUC {res[kind]['auc_mean']:.3f} (CI {lo:.3f}-{hi:.3f}), "
              f"PR-AUC {res[kind]['pr_auc_mean']:.3f} (chance {res[kind]['pr_auc_baseline']:.3f})", flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps({"config": vars(args), "patch_level": res}, indent=2))


if __name__ == "__main__":
    main()
