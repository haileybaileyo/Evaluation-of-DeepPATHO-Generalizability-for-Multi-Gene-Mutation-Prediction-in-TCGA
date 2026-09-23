"""EGFR roadmap, step 4: train MMA + FC head on top of a frozen Phikon-v2
backbone, on the EGFR mutation label (fresh run -- no reuse of the earlier
TP53/6-gene model weights or their conclusions, only of already-downloaded
public TCGA slides and already-fetched EGFR labels).

Mirrors train.py's resume/checkpoint/test-prediction pattern (same lessons
learned: LRU-bounded slide cache already in dataset.py, CPU thread limits,
resumable checkpointing) but uses PhikonDeepPATHOClassifier and a single
gene label instead of the from-scratch multi-gene model.

Usage:
    python -m deeppatho_gen.roadmap_egfr_train --config configs/egfr_roadmap/train_config.yaml
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
from .diagnostics.fixed_validation import cap_patches_per_slide, slide_level_metrics
from .model.phikon_backbone import PhikonDeepPATHOClassifier
from .train import get_device, limit_cpu_threads


def build_index(cfg: dict) -> pd.DataFrame:
    coords = pd.read_csv(cfg["patch_coords"])
    labels = pd.read_csv(cfg["labels"])
    labels_by_case = labels.groupby("case_id")[cfg["gene"]].max().reset_index().rename(columns={cfg["gene"]: "label"})
    index_df = coords.merge(labels_by_case, on="case_id", how="inner")
    split = case_level_split(index_df["case_id"].tolist(), train_frac=cfg["train_frac"], val_frac=cfg["val_frac"], seed=cfg["split_seed"])
    index_df["split"] = index_df["case_id"].map(split)
    index_df = cap_patches_per_slide(index_df, cfg["max_patches_per_slide"], cfg["seed"])
    return index_df


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels, all_file_ids, all_case_ids = [], [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for local_t, global_t, y, meta in tqdm(loader, leave=False):
            local_t, global_t, y = local_t.to(device), global_t.to(device), y.to(device)
            logits = model(local_t, global_t)
            loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy()[:, 0].tolist())
            all_labels.extend(y.detach().cpu().numpy()[:, 0].tolist())
            all_file_ids.extend(meta["file_id"])
            all_case_ids.extend(meta["case_id"])

    patch_df = pd.DataFrame({"file_id": all_file_ids, "case_id": all_case_ids, "prob": all_probs, "label": all_labels})
    return total_loss / max(n_batches, 1), patch_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/egfr_roadmap/train_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    limit_cpu_threads(cfg["torch_threads"])
    torch.manual_seed(cfg["seed"])

    index_df = build_index(cfg)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = PatchPairDataset(index_df[index_df.split == "train"], ["label"], augment=cfg["augment"])
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], ["label"], augment=False)
    test_df_raw = index_df[index_df.split == "test"]

    for name, df in [("train", index_df[index_df.split == "train"]), ("val", index_df[index_df.split == "val"]), ("test", test_df_raw)]:
        n_pos = df.groupby("case_id")["label"].max().sum()
        n_pat = df.case_id.nunique()
        print(f"{name}: {len(df)} patches / {n_pat} patients ({int(n_pos)} EGFR+)")

    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    model = PhikonDeepPATHOClassifier(
        num_classes=cfg["model"]["num_classes"], freeze_backbone=cfg["model"]["freeze_backbone"]
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total/1e6:.1f}M total params, {n_trainable/1e6:.2f}M trainable")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_left = cfg["early_stopping_patience"]
    history = []
    start_epoch = 0

    resume_path = out_dir / "resume_checkpoint.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val_loss = ckpt["best_val_loss"]
        patience_left = ckpt["patience_left"]
        start_epoch = ckpt["epoch"] + 1
        history_path = out_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
        print(f"Resumed from {resume_path}: starting at epoch {start_epoch}")

    t_start = time.time()
    for epoch in range(start_epoch, cfg["max_epochs"]):
        train_loss, train_patch_df = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_patch_df = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        train_m, _ = slide_level_metrics(train_patch_df)
        val_m, _ = slide_level_metrics(val_patch_df)
        elapsed_h = (time.time() - t_start) / 3600

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_auc": train_m["auc"], "val_auc": val_m["auc"], "elapsed_hours": elapsed_h}
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_auc={train_m['auc']:.4f} val_auc={val_m['auc']:.4f} elapsed={elapsed_h:.2f}h")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg["early_stopping_patience"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_val_loss": best_val_loss, "patience_left": patience_left,
        }, resume_path)

    print(f"\nBest val loss: {best_val_loss:.4f}")

    # Final test-set evaluation (single pass, per task discipline: never touched before now)
    if len(test_df_raw) > 0:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
        test_ds = PatchPairDataset(test_df_raw, ["label"], augment=False)
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        _, test_patch_df = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)
        test_m, test_slide_df = slide_level_metrics(test_patch_df)
        test_patch_df.to_csv(out_dir / "test_patch_scores.csv", index=False)
        test_slide_df.reset_index().to_csv(out_dir / "test_slide_scores.csv", index=False)
        print(f"TEST slide-level AUC: {test_m['auc']:.4f} (n_slides={test_m['n_slides']})")
        (out_dir / "test_result.json").write_text(json.dumps(test_m, indent=2, default=str))

    if resume_path.exists() and epoch >= cfg["max_epochs"] - 1:
        resume_path.unlink()


if __name__ == "__main__":
    main()
