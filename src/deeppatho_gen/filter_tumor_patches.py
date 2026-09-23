"""Scores every patch in patch_coords.csv with the trained tumor/normal
classifier and writes a filtered index containing only tumor-predicted
patches -- this is what train.py should point at for the actual mutation
task, instead of the raw (unfiltered) tiling output.

Usage:
    python -m deeppatho_gen.filter_tumor_patches \
        --patch-coords data/processed/patch_coords.csv \
        --tumor-model runs/tumor_classifier/best_model.pt \
        --tumor-config configs/tumor_classifier_config.yaml \
        --out data/processed/patch_coords_tumor_only.csv \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchPairDataset
from .model.backbone import DeepPATHOClassifier
from .train import get_device, limit_cpu_threads


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-coords", default="data/processed/patch_coords.csv")
    ap.add_argument("--tumor-model", default="runs/tumor_classifier/best_model.pt")
    ap.add_argument("--tumor-config", default="configs/tumor_classifier_config.yaml")
    ap.add_argument("--out", default="data/processed/patch_coords_tumor_only.csv")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.tumor_config).read_text())
    limit_cpu_threads(cfg["data"].get("torch_threads", 2))
    device = get_device()

    coords = pd.read_csv(args.patch_coords)
    print(f"Scoring {len(coords)} patches from {coords.file_id.nunique()} slides ...")

    ds = PatchPairDataset(coords.assign(is_tumor=0.0), ["is_tumor"], augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = DeepPATHOClassifier(
        num_classes=1,
        dropout_local=cfg["model"]["dropout_local"],
        dropout_global=cfg["model"]["dropout_global"],
    ).to(device)
    model.load_state_dict(torch.load(args.tumor_model, map_location=device))
    model.eval()

    tumor_probs = []
    with torch.no_grad():
        for local_t, global_t, _labels, _meta in tqdm(loader, desc="scoring"):
            probs = torch.sigmoid(model(local_t.to(device), global_t.to(device))).cpu().numpy()
            tumor_probs.extend(probs[:, 0].tolist())

    coords["tumor_prob"] = tumor_probs
    kept = coords[coords["tumor_prob"] >= args.threshold].drop(columns=["tumor_prob"])
    dropped_frac = 1 - len(kept) / len(coords)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out_path, index=False)

    print(f"Kept {len(kept)}/{len(coords)} patches ({dropped_frac:.1%} dropped as tumor_prob < {args.threshold})")
    print(f"Slides remaining: {kept.file_id.nunique()}/{coords.file_id.nunique()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
