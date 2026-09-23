"""Train a binary tumor-vs-normal patch classifier.

This is the missing step both source papers actually use before their
mutation-prediction task: Coudray et al. only fed their mutation model
tiles their own classifier called "LUAD"; Kosaraju et al. used pathologist
tumor annotations. Our first two full training attempts skipped this
entirely (fed the mutation model every tissue patch, tumor or not) and
overfit immediately -- this classifier's output is what
filter_tumor_patches.py uses to restrict the mutation-training set to
tumor-predicted patches only.

Much easier task than mutation prediction (tumor vs. normal lung tissue is
visually obvious -- Coudray et al. got ~0.99 AUC on this exact task), so it
should converge in far fewer epochs.

Usage:
    python -m deeppatho_gen.train_tumor_classifier --config configs/tumor_classifier_config.yaml
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
from tqdm import tqdm

from .dataset import PatchPairDataset, case_level_split
from .model.backbone import DeepPATHOClassifier
from .train import get_device, limit_cpu_threads, run_epoch


def build_tumor_normal_index(tumor_coords_csv: str, normal_coords_csv: str) -> pd.DataFrame:
    tumor = pd.read_csv(tumor_coords_csv)
    normal = pd.read_csv(normal_coords_csv)
    tumor["is_tumor"] = 1.0
    normal["is_tumor"] = 0.0
    return pd.concat([tumor, normal], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tumor_classifier_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    limit_cpu_threads(cfg["data"].get("torch_threads", 2))

    index_df = build_tumor_normal_index(cfg["data"]["tumor_coords_csv"], cfg["data"]["normal_coords_csv"])
    print(f"{len(index_df)} patches: {int(index_df.is_tumor.sum())} tumor, "
          f"{int((1 - index_df.is_tumor).sum())} normal, "
          f"from {index_df.file_id.nunique()} slides / {index_df.case_id.nunique()} cases")

    split = case_level_split(index_df["case_id"].tolist(), train_frac=cfg["data"]["train_frac"],
                              val_frac=cfg["data"]["val_frac"])
    index_df["split"] = index_df["case_id"].map(split)

    device = get_device()
    print(f"Using device: {device}")

    gene_cols = ["is_tumor"]  # reuse PatchPairDataset's label-column mechanism for a single binary label
    train_ds = PatchPairDataset(index_df[index_df.split == "train"], gene_cols, augment=True)
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], gene_cols, augment=False)

    num_workers = cfg["data"].get("num_workers", 6)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"),
                          persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["optim"]["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["optim"]["batch_size"], shuffle=False, **loader_kwargs)

    model = DeepPATHOClassifier(
        num_classes=1,
        dropout_local=cfg["model"]["dropout_local"],
        dropout_global=cfg["model"]["dropout_global"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_left = cfg["optim"]["early_stopping_patience"]
    history = []
    max_hours = cfg["optim"].get("max_hours")
    deadline = time.time() + max_hours * 3600 if max_hours else None
    t_start = time.time()

    for epoch in range(cfg["optim"]["epochs"]):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        val_acc = ((val_probs > 0.5).float() == val_labels).float().mean().item()
        elapsed_h = (time.time() - t_start) / 3600
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  elapsed={elapsed_h:.2f}h")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_acc": val_acc, "elapsed_hours": elapsed_h})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg["optim"]["early_stopping_patience"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping (no val improvement).")
                break

        if deadline and time.time() > deadline:
            print(f"Stopping: hit the {max_hours}h time budget after epoch {epoch}.")
            break

    print(f"Best val loss: {best_val_loss:.4f}. Model checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
