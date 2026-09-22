"""Train DeepPATHOClassifier as a multi-label (multi-gene) instance-level
classifier on matched 20x/5x LUAD patch pairs.

This is the step that actually requires a GPU and the downloaded/tiled WSI
data from the earlier phases. Everything upstream of this script (label
fetch, manifest, tiling coordinates, model smoke test) already runs and has
been verified on this machine without either.

Usage:
    python -m deeppatho_gen.train --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchPairDataset, build_patch_index, case_level_split
from .model.backbone import DeepPATHOClassifier


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple silicon
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for local_t, global_t, labels, _meta in tqdm(loader, leave=False):
            local_t, global_t, labels = local_t.to(device), global_t.to(device), labels.to(device)
            logits = model(local_t, global_t)
            loss = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_probs.append(torch.sigmoid(logits).detach().cpu())
            all_labels.append(labels.detach().cpu())

    return total_loss / max(n_batches, 1), torch.cat(all_probs), torch.cat(all_labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--genes-config", default="configs/genes.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    genes_cfg = yaml.safe_load(Path(args.genes_config).read_text())
    gene_cols = list(genes_cfg["phase1_genes"])

    index_df = build_patch_index(
        patch_coords_csv="data/processed/patch_coords.csv",
        labels_csv="data/raw/mutation_labels.csv",
    )

    split = case_level_split(
        index_df["case_id"].tolist(),
        train_frac=cfg["data"]["train_frac"],
        val_frac=cfg["data"]["val_frac"],
    )
    index_df["split"] = index_df["case_id"].map(split)

    device = get_device()
    print(f"Using device: {device}")

    train_ds = PatchPairDataset(index_df[index_df.split == "train"], gene_cols)
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], gene_cols)

    train_loader = DataLoader(train_ds, batch_size=cfg["optim"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["optim"]["batch_size"], shuffle=False, num_workers=4)

    model = DeepPATHOClassifier(
        num_classes=len(gene_cols),
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

    for epoch in range(cfg["optim"]["epochs"]):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg["optim"]["early_stopping_patience"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best val loss: {best_val_loss:.4f}. Model checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
