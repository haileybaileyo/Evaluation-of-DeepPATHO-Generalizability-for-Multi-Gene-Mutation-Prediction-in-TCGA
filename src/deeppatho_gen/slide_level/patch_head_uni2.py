"""UNI2-h patch features + the Stage A protocol (10 genes, same split, same evaluation).

Question: in the DeepPATH-style setting (patch-level training, label inherited
from the patient, tile probabilities averaged per slide), what changes if the
image encoder is a strong frozen pathology model (UNI2-h) instead of an
Inception v3 trained from scratch?

Data: the TRIDENT-extracted UNI2-h patch features already on disk (20x, 1536-d,
all tissue patches of each slide). Per slide at most --cap patches are kept at
random (default 800, the cap used for the Stage A tiles). Heads are trained on
the frozen features, so a run takes minutes.

Same as Stage A (deeppath_style/train_multilabel.py): the 10 genes and labels
(mutation_labels_10gene.csv), the patient split (stratified 70/15/15, seed 42 ->
the same 71 test patients / 86 test slides), tile probabilities averaged per
slide, per-gene slide AUC with bootstrap CI (the very same summarize()), and
EGFR Precision/Recall/F1 at a val-fit threshold and at 0.5.

Different from Stage A: the patches (TRIDENT 20x patches, not our 512px tiles),
the encoder (frozen UNI2-h, not trained), and the trainable part (a linear or
small-MLP head instead of a whole Inception v3).

Variants (all patch-level, early stopping on validation BCE, 5 seeds each):
  linear_10     Linear(1536 -> 10)             all genes jointly
  mlp_10        MLP(1536 -> 512 -> 10)         all genes jointly
  linear_egfr   Linear(1536 -> 1)              EGFR only  (multi-task control)
  mlp_egfr      MLP(1536 -> 512 -> 1)          EGFR only  (multi-task control)

Usage:
    python -m deeppatho_gen.slide_level.patch_head_uni2 --features-dir <.../uni_v2/TCGA_LUAD>
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..deeppath_style.train_multilabel import GENES, summarize, slide_table
from ..metrics import binary_metrics, find_best_threshold
from .fixed_split_logreg import get_fixed_split

EGFR = GENES.index("EGFR")


def load_data(features_dir, pooled, split, lab10, cap, seed):
    z = np.load(pooled)
    rng = np.random.default_rng(seed)
    X, slide, case = [], [], []
    t0 = time.time()
    for sid, cid in zip(z["slide_ids"], z["case_ids"]):
        if cid not in split or cid not in lab10.index:
            continue
        with h5py.File(Path(features_dir) / f"{sid}.h5", "r") as h:
            f = h["features"][:]
        if len(f) > cap:
            f = f[np.sort(rng.choice(len(f), cap, replace=False))]
        X.append(f.astype(np.float16))
        slide += [sid] * len(f)
        case += [cid] * len(f)
    X = np.concatenate(X)
    print(f"loaded {len(X):,} patches from {len(set(slide))} slides ({len(set(case))} patients) in {time.time() - t0:.0f}s", flush=True)
    return X, np.array(slide), np.array(case)


def make_head(kind, d, n_out):
    if kind == "linear":
        return nn.Linear(d, n_out)
    return nn.Sequential(nn.Dropout(0.25), nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.25), nn.Linear(512, n_out))


def train_head(kind, Xtr, Ytr, Xva, Yva, mu, sd, seed, dev, epochs=40, patience=4, lr=1e-3, wd=1e-2, bs=4096):
    torch.manual_seed(seed)
    net = make_head(kind, Xtr.shape[1], Ytr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    best, state, bad, ep_used = float("inf"), None, 0, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            loss = F.binary_cross_entropy_with_logits(net((Xtr[idx].float() - mu) / sd), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        # exact mean BCE over all val patches and outputs
        with torch.no_grad():
            tot = sum(F.binary_cross_entropy_with_logits(net((Xva[i:i + 20000].float() - mu) / sd), Yva[i:i + 20000],
                                                         reduction="sum").item() for i in range(0, len(Xva), 20000))
        vl = tot / Yva.numel()
        ep_used = ep + 1
        if vl < best:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net.eval(), best, ep_used


@torch.no_grad()
def predict(net, X, mu, sd, dev):
    return torch.cat([torch.sigmoid(net((X[i:i + 20000].to(dev).float() - mu) / sd)).cpu()
                      for i in range(0, len(X), 20000)]).numpy()


def egfr_protocol(val_df, val_p, val_Y, test_df, test_p, test_Y):
    mv, _, lv, _ = slide_table(val_df, val_p, val_Y)
    mt, _, lt, _ = slide_table(test_df, test_p, test_Y)
    yv, yt = lv["y_EGFR"].values.astype(int), lt["y_EGFR"].values.astype(int)
    thr = find_best_threshold(yv, mv["EGFR"].values)
    out = {"threshold_fit_on_val": float(thr)}
    for name, t in [("at_val_threshold", thr), ("at_0.5", 0.5)]:
        m = binary_metrics(yt, mt["EGFR"].values, t)
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        m["predicted_positive_ratio"] = (tp + fp) / len(yt)
        out[name] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--pooled", default="data/processed/uni_v2_luad_slide_pooled.npz")
    ap.add_argument("--labels10", default="data/raw/mutation_labels_10gene.csv")
    ap.add_argument("--coords", default="data/processed/patch_coords_tumor_only.csv")
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--cap", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out-dir", default="runs/slide_level/R5_uni2_patch_heads")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    lab10 = pd.read_csv(args.labels10).groupby("case_id")[GENES].max()
    split, _ = get_fixed_split(args.coords, args.labels, "EGFR")
    X, slide, case = load_data(args.features_dir, args.pooled, split, lab10, args.cap, 0)
    Y = lab10.loc[case, GENES].values.astype(np.float32)
    part = np.array([split[c] for c in case])
    ix = {k: np.where(part == k)[0] for k in ["train", "val", "test"]}
    for k, i in ix.items():
        print(f"{k:5s}: {len(set(case[i])):3d} patients, {len(set(slide[i])):3d} slides, {len(i):7,} patches", flush=True)
    Xt = {k: torch.from_numpy(X[i]).to(dev) for k, i in ix.items()}
    Yt = {k: torch.from_numpy(Y[i]).to(dev) for k, i in ix.items()}
    mu = Xt["train"].float().mean(0); sd = Xt["train"].float().std(0) + 1e-6
    dfs = {k: pd.DataFrame({"file_id": slide[i], "case_id": case[i]}) for k, i in ix.items()}

    variants = [("linear", "10"), ("mlp", "10"), ("linear", "egfr"), ("mlp", "egfr")]
    results = {"config": vars(args), "variants": {}}
    for kind, target in variants:
        name = f"{kind}_{target}"
        cols = list(range(10)) if target == "10" else [EGFR]
        per_seed, ens_val, ens_test = [], 0, 0
        for seed in range(args.seeds):
            t0 = time.time()
            net, vbce, ep = train_head(kind, Xt["train"], Yt["train"][:, cols], Xt["val"], Yt["val"][:, cols], mu, sd, seed, dev)
            pv, pt = predict(net, Xt["val"], mu, sd, dev), predict(net, Xt["test"], mu, sd, dev)
            full = lambda p: (p if target == "10" else np.concatenate([np.full((len(p), EGFR), 0.5), p, np.full((len(p), 9 - EGFR), 0.5)], 1))  # noqa: E731
            pv, pt = full(pv), full(pt)
            ens_val, ens_test = ens_val + pv / args.seeds, ens_test + pt / args.seeds
            s = summarize(dfs["test"], pt, Y[ix["test"]], n_boot=1)
            per_seed.append({"seed": seed, "epochs": ep, "val_bce": vbce,
                             "test_slide_auc": {g: s[g]["slide_auc_mean_prob"] for g in GENES},
                             "test_tile_auc": {g: s[g]["tile_auc"] for g in GENES}})
            print(f"[{name}] seed {seed}: {ep} ep, val BCE {vbce:.4f}, EGFR test slide AUC {s['EGFR']['slide_auc_mean_prob']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        ens = summarize(dfs["test"], ens_test, Y[ix["test"]])
        results["variants"][name] = {"per_seed": per_seed, "ensemble_test": ens,
                                     "EGFR_protocol_ensemble": egfr_protocol(dfs["val"], ens_val, Y[ix["val"]], dfs["test"], ens_test, Y[ix["test"]])}
        (out / "results.json").write_text(json.dumps(results, indent=2))

    # ---- summary table
    stageA = json.load(open("runs/deeppath_style/stageA_dx/results.json"))["test"]
    print("\n=== TEST slide-level AUC (평균 ± 시드 표준편차; 시드 5개 평균 예측의 앙상블 AUC는 괄호) ===")
    print(f"{'':14s}" + "".join(f"{g:>13s}" for g in GENES))
    print(f"{'StageA(Incep.)':14s}" + "".join(f"{stageA[g]['slide_auc_mean_prob']:>13.3f}" for g in GENES))
    for name, v in results["variants"].items():
        row = f"{name:14s}"
        for g in GENES:
            a = [s["test_slide_auc"][g] for s in v["per_seed"]]
            row += f"{np.mean(a):>7.3f}±{np.std(a):.2f} " if (name.endswith("10") or g == "EGFR") else f"{'-':>13s}"
        print(row)
    print("\nEGFR (시드 앙상블, 슬라이드 86장, 양성 11장):")
    for name, v in results["variants"].items():
        e, m5 = v["EGFR_protocol_ensemble"], v["EGFR_protocol_ensemble"]["at_0.5"]
        mv = e["at_val_threshold"]
        print(f"  {name:12s} AUC {v['ensemble_test']['EGFR']['slide_auc_mean_prob']:.3f} "
              f"[{v['ensemble_test']['EGFR']['slide_auc_ci95'][0]:.3f}-{v['ensemble_test']['EGFR']['slide_auc_ci95'][1]:.3f}] | "
              f"@val-thr {mv['threshold']:.3f}: P {mv['precision']:.3f} R {mv['recall']:.3f} F1 {mv['f1']:.3f} pred+ {100 * mv['predicted_positive_ratio']:.0f}% | "
              f"@0.5: P {m5['precision']:.3f} R {m5['recall']:.3f} F1 {m5['f1']:.3f} pred+ {100 * m5['predicted_positive_ratio']:.0f}%")
    print(f"\nWrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
