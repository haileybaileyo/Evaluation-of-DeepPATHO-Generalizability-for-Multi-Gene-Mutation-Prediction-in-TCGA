"""R3: gated attention MIL (ABMIL, Ilse et al. 2018) on frozen UNI-v2 patch
features -- one bag per PATIENT (all patches of all of that patient's slides).

Where R1/R0b average the patch features (every patch counts equally), ABMIL
learns a per-patch attention weight, so the slide embedding can concentrate
on the patches that matter. The label is still per patient; no patch labels.

Two evaluations, both patient-level:
  cv    repeated stratified K-fold over all patients (same folds logic as
        cv_baseline.py; early stopping on a stratified 15% hold-out of each
        training fold). First gene in --genes gets --n-repeats, the rest
        --ctrl-repeats (positive controls). Optional label-permutation control.
  fixed the exact 70/15/15 split used by the patch experiments (train ->
        fit, val -> early stopping AND F1-maximizing threshold, test ->
        reported at that threshold and at 0.5), repeated over training seeds.

Class imbalance: BCE with pos_weight = n_neg / n_pos of the fitting patients.

Usage:
    python -m deeppatho_gen.slide_level.abmil_cv --features-dir <.../uni_v2/TCGA_LUAD>
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..metrics import binary_metrics, find_best_threshold
from .cv_baseline import bootstrap_auc_ci
from .fixed_split_logreg import get_fixed_split


class GatedABMIL(nn.Module):
    def __init__(self, in_dim: int = 1536, hid: int = 256, att: int = 128, drop: float = 0.25):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(drop))
        self.V, self.U, self.w = nn.Linear(hid, att), nn.Linear(hid, att), nn.Linear(att, 1)
        self.att_drop = nn.Dropout(drop)
        self.cls = nn.Linear(hid, 1)

    def forward(self, x: torch.Tensor):
        """x: (n_patches, in_dim) -> (logit scalar, attention (n_patches,))."""
        h = self.fc(self.norm(x))
        a = self.w(self.att_drop(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))).squeeze(-1)
        a = torch.softmax(a, dim=0)
        return self.cls((a.unsqueeze(-1) * h).sum(0)).squeeze(-1), a


def load_bags(features_dir: str, pooled_npz: str, patients: list[str]) -> dict[str, torch.Tensor]:
    z = np.load(pooled_npz)
    slide_ids, case_ids = z["slide_ids"], z["case_ids"]
    bags = {}
    t0 = time.time()
    for k, c in enumerate(patients):
        parts = []
        for i in np.where(case_ids == c)[0]:
            with h5py.File(Path(features_dir) / f"{slide_ids[i]}.h5", "r") as h:
                parts.append(h["features"][:].astype(np.float16))
        bags[c] = torch.from_numpy(np.concatenate(parts, 0))
        if (k + 1) % 100 == 0:
            print(f"  loaded {k + 1}/{len(patients)} patients ({time.time() - t0:.0f}s)", flush=True)
    n = sum(len(b) for b in bags.values())
    print(f"{len(bags)} patient bags, {n:,} patches, {n * 1536 * 2 / 1e9:.1f} GB (fp16)", flush=True)
    return bags


def fit_predict(fit_ids, val_ids, pred_ids, y, bags, seed, device, cfg):
    """Train on fit_ids (early stopping on val_ids' loss), return {id: prob}
    for pred_ids (val_ids too, so callers can fit a threshold on them)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedABMIL().to(device)
    n_pos = int(sum(y[c] for c in fit_ids))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(fit_ids) - n_pos) / max(n_pos, 1), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    target = lambda c: torch.tensor(float(y[c]), device=device)  # noqa: E731

    best_loss, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        order = rng.permutation(len(fit_ids))
        for k, i in enumerate(order, 1):
            c = fit_ids[i]
            x = bags[c]
            if len(x) > cfg["max_patches"]:
                x = x[torch.from_numpy(rng.choice(len(x), cfg["max_patches"], replace=False))]
            logit, _ = model(x.to(device).float())
            (loss_fn(logit, target(c)) / cfg["accum"]).backward()
            if k % cfg["accum"] == 0 or k == len(order):
                opt.step()
                opt.zero_grad()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(model(bags[c].to(device).float())[0], target(c)).item() for c in val_ids]))
        if vl < best_loss:
            best_loss, best_state, bad = vl, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = {c: float(torch.sigmoid(model(bags[c].to(device).float())[0])) for c in set(pred_ids) | set(val_ids)}
    return probs, ep + 1, best_loss


def run_cv(patients, y, bags, n_splits, n_repeats, seed, device, cfg, tag):
    ids = np.array(patients)
    yy = np.array([y[c] for c in patients])
    oofs, aucs = [], []
    for r in range(n_repeats):
        oof = np.zeros(len(ids))
        for f, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(ids, yy)):
            fit, val = train_test_split(ids[tr], test_size=0.15, random_state=seed + r, stratify=yy[tr])
            t0 = time.time()
            p, ep, bl = fit_predict(list(fit), list(val), list(ids[te]), y, bags, seed + 100 * r + f, device, cfg)
            oof[te] = [p[c] for c in ids[te]]
            print(f"  [{tag}] repeat {r} fold {f}: {ep} epochs, best val loss {bl:.3f}, fold AUC "
                  f"{roc_auc_score(yy[te], oof[te]):.3f}  ({time.time() - t0:.0f}s)", flush=True)
        oofs.append(oof)
        aucs.append(roc_auc_score(yy, oof))
        print(f"[{tag}] repeat {r}: pooled OOF AUC {aucs[-1]:.3f}", flush=True)
    lo, hi = bootstrap_auc_ci(yy, np.mean(oofs, 0))
    return {
        "n_patients": len(yy), "n_positive": int(yy.sum()), "auc_mean": float(np.mean(aucs)),
        "auc_std_over_repeats": float(np.std(aucs)), "auc_bootstrap_ci95": [lo, hi],
        "pr_auc_mean": float(np.mean([average_precision_score(yy, o) for o in oofs])),
        "pr_auc_baseline": float(yy.mean()), "auc_per_repeat": [float(a) for a in aucs],
    }, np.stack(oofs)


def run_fixed(patients, y, split, bags, n_seeds, device, cfg):
    ids = np.array(patients)
    part = np.array([split[c] for c in patients])
    fit, val, test = list(ids[part == "train"]), list(ids[part == "val"]), list(ids[part == "test"])
    print(f"fixed split: train {len(fit)} ({int(sum(y[c] for c in fit))}+), val {len(val)} "
          f"({int(sum(y[c] for c in val))}+), test {len(test)} ({int(sum(y[c] for c in test))}+)", flush=True)
    yv, yt = np.array([y[c] for c in val]), np.array([y[c] for c in test])
    out = []
    for s in range(n_seeds):
        t0 = time.time()
        p, ep, bl = fit_predict(fit, val, test, y, bags, s, device, cfg)
        sv, st = np.array([p[c] for c in val]), np.array([p[c] for c in test])
        thr = find_best_threshold(yv, sv)
        r = {"seed": s, "epochs": ep, "best_val_loss": bl, "val_auc": float(roc_auc_score(yv, sv)),
             "threshold_fit_on_val": float(thr), "test_at_val_threshold": binary_metrics(yt, st, thr),
             "test_at_0.5": binary_metrics(yt, st, 0.5)}
        out.append(r)
        m, m5 = r["test_at_val_threshold"], r["test_at_0.5"]
        print(f"  seed {s}: {ep} ep, val AUC {r['val_auc']:.3f} | TEST AUC {m['roc_auc']:.3f} | "
              f"@val-thr {thr:.3f}: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
              f"pred+ {100 * (m['confusion_matrix'][0][1] + m['confusion_matrix'][1][1]) / len(yt):.0f}% | "
              f"@0.5: P {m5['precision']:.3f} R {m5['recall']:.3f} F1 {m5['f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--pooled", default="data/processed/uni_v2_luad_slide_pooled.npz")
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--coords", default="data/processed/patch_coords_tumor_only.csv")
    ap.add_argument("--mode", choices=["cv", "fixed", "both"], default="both")
    ap.add_argument("--genes", nargs="+", default=["EGFR", "STK11"])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--ctrl-repeats", type=int, default=1)
    ap.add_argument("--n-perm", type=int, default=1)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-patches", type=int, default=4000)
    ap.add_argument("--out-dir", default="runs/slide_level/R3_abmil")
    args = ap.parse_args()
    cfg = {k: getattr(args, k.replace("-", "_")) for k in ["epochs", "patience", "lr", "wd", "accum"]}
    cfg["max_patches"] = args.max_patches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    print(f"device {device}; cfg {cfg}", flush=True)

    labels = pd.read_csv(args.labels).groupby("case_id")[args.genes].max()
    z = np.load(args.pooled)
    patients = sorted(c for c in set(z["case_ids"]) if c in labels.index)
    bags = load_bags(args.features_dir, args.pooled, patients)
    results = {"config": {**vars(args), **cfg}, "cv": {}, "permutation": {}, "fixed": {}}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save():
        (out / "results.json").write_text(json.dumps(results, indent=2))

    if args.mode in ("cv", "both"):
        for gi, g in enumerate(args.genes):
            y = {c: int(labels.loc[c, g]) for c in patients}
            res, oofs = run_cv(patients, y, bags, args.n_splits, args.n_repeats if gi == 0 else args.ctrl_repeats,
                               args.seed, device, cfg, g)
            results["cv"][g] = res
            np.savez(out / f"oof_{g}.npz", patients=np.array(patients), oof=oofs, y=np.array([y[c] for c in patients]))
            print(f"=== CV {g}: AUC {res['auc_mean']:.3f} (CI {res['auc_bootstrap_ci95'][0]:.3f}-"
                  f"{res['auc_bootstrap_ci95'][1]:.3f}), PR-AUC {res['pr_auc_mean']:.3f} (chance {res['pr_auc_baseline']:.3f})", flush=True)
            save()
        g = args.genes[0]
        for p in range(args.n_perm):
            yp = dict(zip(patients, np.random.default_rng(args.seed + 1000 + p).permutation([int(labels.loc[c, g]) for c in patients])))
            res, _ = run_cv(patients, yp, bags, args.n_splits, 1, args.seed + 7 * p, device, cfg, f"perm-{g}")
            results["permutation"].setdefault(g, []).append(res["auc_mean"])
            print(f"=== permuted-label control {g} #{p}: AUC {res['auc_mean']:.3f} (expect ~0.5)", flush=True)
            save()

    if args.mode in ("fixed", "both"):
        g = args.genes[0]
        split, case_labels = get_fixed_split(args.coords, args.labels, g)
        pat = [c for c in patients if c in split]
        results["fixed"][g] = run_fixed(pat, case_labels, split, bags, args.n_seeds, device, cfg)
        a = [r["test_at_val_threshold"]["roc_auc"] for r in results["fixed"][g]]
        print(f"=== FIXED split {g}: test AUC {np.mean(a):.3f} +/- {np.std(a):.3f} over {len(a)} seeds", flush=True)
        save()
    print(f"Wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
