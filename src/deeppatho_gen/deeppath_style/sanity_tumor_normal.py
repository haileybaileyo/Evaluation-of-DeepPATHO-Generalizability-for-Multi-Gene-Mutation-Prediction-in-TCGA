"""Sanity check of the Stage A training code on an EASY target: tumor vs normal tiles.

Purpose: show that the very same machinery used for the 10-gene run (TileDataset
with random-crop/flip/colour augmentation, Inception v3 from scratch, TF-style
RMSProp, weight EMA, BN settings, evaluation by predict_tiles + per-slide mean
probability) actually LEARNS when the label is learnable. It is a test of the
code, not of biology: tumor tiles come from the DX (FFPE) slides and normal
tiles from the solid-tissue-normal (frozen) slides, so slide type alone makes the
task easy -- an AUC near 1.0 therefore says "the training/evaluation code
works", nothing about EGFR.

Data: tumor tiles = the existing Stage A tiles (dx_20x_512), at most
--tumor-cap per slide; normal tiles = normal_20x_512 (tile_512.py on
gdc_manifest_normal.csv), at most --normal-cap per slide.
Split (patient level, no patient in two splits): patients already in the
project's EGFR split keep that assignment; normal-only patients are assigned
randomly 70/15/15 with a fixed seed.

Usage:
    python -m deeppatho_gen.deeppath_style.sanity_tumor_normal --config configs/deeppath_style/stageA_dx.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..metrics import binary_metrics
from ..slide_level.fixed_split_logreg import get_fixed_split
from .train_multilabel import EMA, TFRMSprop, TileDataset, build_model, loss_fn, predict_tiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tumor-cap", type=int, default=150)
    ap.add_argument("--normal-cap", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--out-dir", default="runs/deeppath_style/sanity_tumor_normal")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path("data/processed/deeppath_tiles")

    tum = pd.read_csv(root / "dx_20x_512/tile_index.csv")
    nor = pd.read_csv(root / "normal_20x_512/tile_index.csv")
    rng = np.random.default_rng(0)
    def cap(df, k):
        return df.groupby("file_id", group_keys=False).apply(lambda g: g if len(g) <= k else g.sample(k, random_state=0))
    tum, nor = cap(tum, args.tumor_cap).copy(), cap(nor, args.normal_cap).copy()
    tum["path"] = "dx_20x_512/" + tum["path"]; nor["path"] = "normal_20x_512/" + nor["path"]
    tum["y"], nor["y"] = 1.0, 0.0
    df = pd.concat([tum, nor], ignore_index=True)

    split, _ = get_fixed_split(cfg["split_coords"], cfg["split_labels"], "EGFR", cfg["train_frac"], cfg["val_frac"], cfg["split_seed"])
    extra = sorted(set(df.case_id) - set(split))
    r = np.random.default_rng(123); perm = r.permutation(len(extra))
    for i, j in enumerate(perm):
        split[extra[j]] = "train" if i < 0.7 * len(extra) else ("val" if i < 0.85 * len(extra) else "test")
    df["split"] = df.case_id.map(split)
    parts = {k: np.where(df.split.values == k)[0] for k in ["train", "val", "test"]}
    for k, ix in parts.items():
        d = df.iloc[ix]
        print(f"{k:5s}: {d.case_id.nunique():3d} patients, {d.file_id.nunique():3d} slides, {len(d):6,} tiles "
              f"(tumor {int((d.y == 1).sum()):,} / normal {int((d.y == 0).sum()):,})", flush=True)
    assert all(set(df.iloc[parts[a]].case_id).isdisjoint(set(df.iloc[parts[b]].case_id)) for a, b in [("train", "val"), ("train", "test"), ("val", "test")])
    Y = df[["y"]].values.astype(np.float32)

    train_ds = TileDataset(df.iloc[parts["train"]], root, Y[parts["train"]], train=True, size=cfg["image_size"])
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"],
                        pin_memory=True, persistent_workers=True, prefetch_factor=4)
    model = build_model(1, cfg["bn_momentum"]).to(dev).to(memory_format=torch.channels_last)
    opt = TFRMSprop(model.parameters(), cfg["rms_decay"], cfg["momentum"], cfg["rms_eps"], cfg["weight_decay"])
    ema = EMA(model, cfg["ema_decay"])
    vdf = df.iloc[parts["val"]]
    vdf = vdf.groupby("file_id", group_keys=False).apply(lambda g: g if len(g) <= 60 else g.sample(60, random_state=0))
    vY = vdf[["y"]].values.astype(np.float32)

    def tile_auc(m, d, y):
        p = predict_tiles(m, d, root, y, dev, cfg["image_size"], cfg["num_workers"])[:, 0]
        return p, float(roc_auc_score(y[:, 0], p))

    history, step, t0, run = [], 0, time.time(), 0.0
    while step < args.max_steps:
        for x, y, _ in loader:
            model.train()
            x = x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.to(dev, non_blocking=True)
            loss = loss_fn(model(x), y, cfg["label_smoothing"], cfg["aux_weight"])
            opt.zero_grad(); loss.backward(); opt.step(cfg["lr"])
            step += 1; ema.update(model, step); run += loss.item()
            if step % 200 == 0:
                print(f"step {step} loss {run / 200:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True); run = 0.0
            if step % args.eval_every == 0:
                p, auc = tile_auc(ema.shadow, vdf, vY)
                acc = float(((p >= 0.5) == (vY[:, 0] == 1)).mean())
                history.append({"step": step, "val_tile_auc": auc, "val_tile_acc": acc})
                print("EVAL", json.dumps({k: round(v, 4) for k, v in history[-1].items()}), flush=True)
            if step >= args.max_steps:
                break

    # ---- final test with the EMA weights (all test tiles)
    tdf = df.iloc[parts["test"]]; tY = Y[parts["test"]]
    p, auc = tile_auc(ema.shadow, tdf, tY)
    slide = pd.DataFrame({"file_id": tdf.file_id.values, "p": p, "y": tdf.y.values}).groupby("file_id").agg(p=("p", "mean"), y=("y", "first"))
    m_tile = binary_metrics(tY[:, 0], p, 0.5); m_slide = binary_metrics(slide.y.values, slide.p.values, 0.5)
    res = {"history": history, "test_tile_auc": auc, "test_tile_at_0.5": m_tile, "test_slide_auc": float(roc_auc_score(slide.y, slide.p)),
           "test_slide_at_0.5": m_slide, "n_test_slides": len(slide), "config": vars(args)}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\n=== TEST tile AUC {auc:.4f} | slide AUC {res['test_slide_auc']:.4f} ({len(slide)} slides)")
    for name, m in [("tile", m_tile), ("slide", m_slide)]:
        print(f"{name}: P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} cm {m['confusion_matrix']}")
    print(f"Wrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
