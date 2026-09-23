"""E: MMA (DeepPATHOClassifier) vs 20x-only single-magnification baseline,
under conditions identical to D (fixed_validation.py) -- same patient split,
same TP53 labels, same tumor-filtered patch set, same patch budget, same
optimizer/schedule, same seed. Only the model architecture differs, so any
metric gap is attributable to MMA itself rather than a confound.

Question this answers: does the cross-magnification attention mechanism
help at all, or is it dead weight (or worse) relative to a plain
single-tower classifier on the same data?

Usage:
    python -m deeppatho_gen.diagnostics.mma_vs_baseline --config configs/diagnostics/mma_vs_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from ..dataset import PatchPairDataset, case_level_split
from ..model.backbone import DeepPATHOClassifier
from ..model.single_mag_baseline import SingleMagnificationClassifier
from ..train import get_device, limit_cpu_threads
from .fixed_validation import cap_patches_per_slide, git_commit_hash, run_epoch, slide_level_metrics


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_one_model(name: str, model: nn.Module, cfg: dict, index_df: pd.DataFrame, device, out_dir: Path) -> dict:
    train_ds = PatchPairDataset(index_df[index_df.split == "train"], ["label"], augment=cfg["augment"])
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], ["label"], augment=False)

    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    metric_key = cfg["checkpoint_selection_metric"]
    best_metric = float("inf") if metric_key == "val_loss" else -float("inf")
    patience_left = cfg["early_stopping_patience"]
    history = []
    t0 = time.time()

    for epoch in range(cfg["max_epochs"]):
        train_loss, train_patch_df = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_patch_df = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        train_m, _ = slide_level_metrics(train_patch_df)
        val_m, _ = slide_level_metrics(val_patch_df)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_auc": train_m["auc"], "val_auc": val_m["auc"], "val_balanced_acc": val_m["balanced_acc"]}
        history.append(row)
        print(f"[{name}] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_auc={train_m['auc']:.4f} val_auc={val_m['auc']:.4f}")

        current = val_loss if metric_key == "val_loss" else val_m["auc"]
        improved = (current < best_metric) if metric_key == "val_loss" else (current > best_metric)
        if improved:
            best_metric = current
            patience_left = cfg["early_stopping_patience"]
            torch.save(model.state_dict(), out_dir / f"{name}_best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{name}] Early stopping.")
                break

    elapsed = time.time() - t0
    pd.DataFrame(history).to_csv(out_dir / f"{name}_epoch_metrics.csv", index=False)

    best_val_auc = max((h["val_auc"] for h in history if h["val_auc"] == h["val_auc"]), default=float("nan"))  # nan-safe max
    return {
        "name": name, "n_params": count_params(model), "n_epochs_run": len(history),
        "best_metric_value": best_metric, "best_val_auc_seen": best_val_auc,
        "final_train_auc": history[-1]["train_auc"], "elapsed_seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/diagnostics/mma_vs_baseline.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    limit_cpu_threads(cfg["torch_threads"])

    coords = pd.read_csv(cfg["patch_coords"])
    labels = pd.read_csv(cfg["labels"])
    labels_by_case = labels.groupby("case_id")[cfg["gene"]].max().reset_index().rename(columns={cfg["gene"]: "label"})
    index_df = coords.merge(labels_by_case, on="case_id", how="inner")
    split = case_level_split(index_df["case_id"].tolist(), train_frac=cfg["train_frac"], val_frac=cfg["val_frac"], seed=cfg["split_seed"])
    index_df["split"] = index_df["case_id"].map(split)
    index_df = cap_patches_per_slide(index_df, cfg["max_patches_per_slide"], cfg["seed"])

    device = get_device()
    print(f"train: {len(index_df[index_df.split=='train'])} patches / {index_df[index_df.split=='train'].case_id.nunique()} patients")
    print(f"val:   {len(index_df[index_df.split=='val'])} patches / {index_df[index_df.split=='val'].case_id.nunique()} patients")

    results = []
    for name, model in [
        ("mma", DeepPATHOClassifier(num_classes=1)),
        ("single_mag_20x", SingleMagnificationClassifier(num_classes=1)),
    ]:
        torch.manual_seed(cfg["seed"])  # same init seed for both, applied right before each model's training
        model = model.to(device)
        print(f"\n=== Training {name} ({count_params(model)/1e6:.2f}M params) ===")
        results.append(run_one_model(name, model, cfg, index_df, device, out_dir))

    comparison = pd.DataFrame(results)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    meta = {"gene": cfg["gene"], "git_commit": git_commit_hash(), "config": cfg, "results": results}
    (out_dir / "comparison_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n=== E: MMA vs single-magnification baseline ===")
    print(comparison.to_string(index=False))
    mma_row = comparison[comparison.name == "mma"].iloc[0]
    base_row = comparison[comparison.name == "single_mag_20x"].iloc[0]
    delta = mma_row["best_val_auc_seen"] - base_row["best_val_auc_seen"]
    print(f"\nbest_val_auc_seen: MMA={mma_row['best_val_auc_seen']:.4f} vs "
          f"single_mag={base_row['best_val_auc_seen']:.4f}  (delta={delta:+.4f})")
    print(f"Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
