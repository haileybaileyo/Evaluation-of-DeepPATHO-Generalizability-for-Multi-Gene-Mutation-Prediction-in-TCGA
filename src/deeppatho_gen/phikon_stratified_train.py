"""EGFR, Phikon-v2 backbone + MMA (Experiment C: frozen transfer learning),
with the SAME stratified split/metrics fixes as egfr_stratified_train.py
and uni_egfr_train.py -- the missing fair-comparison counterpart to those
two scripts.

Relation to existing code: roadmap_egfr_train.py already trains a frozen
Phikon-v2 + MMA model on EGFR, but on the OLD case_level_split (8:1:1,
non-stratified) -- its test set is a *different* set of patients than
egfr_stratified_train.py / uni_egfr_train.py use, so its numbers are not
directly comparable in a side-by-side table. This script is that same
frozen-Phikon setup re-run on the stratified 70/15/15 split (split_seed=42,
same as baseline/UNI), with metrics.py's full Precision/Recall/F1/ROC-AUC/
PR-AUC at patch and slide level and a val-fit decision threshold (same
methodology as egfr_stratified_train.py), so Experiment A/B/C become a fair
three-way comparison on identical train/val/test patients.

Usage:
    python -m deeppatho_gen.phikon_stratified_train --config configs/egfr_phikon/train_config.yaml
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

from .dataset import PatchPairDataset, split_summary, stratified_case_level_split
from .diagnostics.fixed_validation import cap_patches_per_slide
from .metrics import find_best_threshold, patch_level_metrics, slide_level_metrics
from .model.phikon_backbone import PhikonDeepPATHOClassifier
from .train import get_device, limit_cpu_threads


def build_index(cfg: dict) -> pd.DataFrame:
    coords = pd.read_csv(cfg["patch_coords"])
    labels = pd.read_csv(cfg["labels"])
    labels_by_case = labels.groupby("case_id")[cfg["gene"]].max().reset_index().rename(columns={cfg["gene"]: "label"})
    index_df = coords.merge(labels_by_case, on="case_id", how="inner")

    case_labels = labels_by_case.set_index("case_id")["label"].to_dict()
    split = stratified_case_level_split(
        index_df["case_id"].tolist(), case_labels,
        train_frac=cfg["train_frac"], val_frac=cfg["val_frac"], seed=cfg["split_seed"],
    )
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
    ap.add_argument("--config", default="configs/egfr_phikon/train_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    limit_cpu_threads(cfg["torch_threads"])
    torch.manual_seed(cfg["seed"])

    index_df = build_index(cfg)
    device = get_device()
    print(f"Using device: {device}")

    summary = split_summary(index_df)
    print("\n=== Patient-level split summary (stratified on EGFR) ===")
    print(summary.to_string())
    print()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "split_summary.csv")

    train_ds = PatchPairDataset(index_df[index_df.split == "train"], ["label"], augment=cfg["augment"])
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], ["label"], augment=False)
    test_df_raw = index_df[index_df.split == "test"]

    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    model = PhikonDeepPATHOClassifier(num_classes=cfg["model"]["num_classes"], freeze_backbone=cfg["model"]["freeze_backbone"]).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total/1e6:.1f}M total params, {n_trainable/1e6:.2f}M trainable (frozen backbone)")

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

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
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg["max_epochs"]):
        train_loss, train_patch_df = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_patch_df = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        val_patch_m = patch_level_metrics(val_patch_df)
        val_slide_m, _ = slide_level_metrics(val_patch_df)
        elapsed_h = (time.time() - t_start) / 3600

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_patch_auc": val_patch_m["roc_auc"], "val_slide_auc": val_slide_m["roc_auc"],
               "val_slide_f1": val_slide_m["f1"], "elapsed_hours": elapsed_h}
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_patch_auc={val_patch_m['roc_auc']:.4f} val_slide_auc={val_slide_m['roc_auc']:.4f} "
              f"val_slide_f1={val_slide_m['f1']:.4f} elapsed={elapsed_h:.2f}h")

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

    result = {
        "config": cfg, "seed": cfg["seed"], "split_seed": cfg["split_seed"],
        "model": "PhikonDeepPATHOClassifier (freeze_backbone=True)",
        "n_epochs_run": len(history), "best_val_loss": best_val_loss,
        "split_summary": summary.reset_index().to_dict(orient="records"),
    }

    if len(test_df_raw) > 0:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

        val_loader_eval = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        _, best_val_patch_df = run_epoch(model, val_loader_eval, optimizer, loss_fn, device, train=False)
        patch_threshold = find_best_threshold(best_val_patch_df["label"].values, best_val_patch_df["prob"].values)
        val_slide_df = best_val_patch_df.groupby("file_id").agg(prob=("prob", "mean"), label=("label", "first"))
        slide_threshold = find_best_threshold(val_slide_df["label"].values, val_slide_df["prob"].values)
        print(f"Threshold fit on val (F1-maximizing): patch={patch_threshold:.4f} slide={slide_threshold:.4f}")

        test_ds = PatchPairDataset(test_df_raw, ["label"], augment=False)
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        _, test_patch_df = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)

        test_patch_m = patch_level_metrics(test_patch_df, threshold=patch_threshold)
        test_slide_m, test_slide_df = slide_level_metrics(test_patch_df, threshold=slide_threshold)

        test_patch_df.to_csv(out_dir / "test_patch_scores.csv", index=False)
        test_slide_df.reset_index().to_csv(out_dir / "test_slide_scores.csv", index=False)

        result["patch_threshold"] = patch_threshold
        result["slide_threshold"] = slide_threshold
        result["test_patch_metrics"] = test_patch_m
        result["test_slide_metrics"] = test_slide_m

        print(f"\n=== TEST (patch-level, threshold={patch_threshold:.4f}) ===")
        print(json.dumps(test_patch_m, indent=2))
        print(f"\n=== TEST (slide-level, threshold={slide_threshold:.4f}) ===")
        print(json.dumps(test_slide_m, indent=2))

    (out_dir / "test_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote full results to {out_dir / 'test_result.json'}")

    if resume_path.exists() and epoch >= cfg["max_epochs"] - 1:
        resume_path.unlink()


if __name__ == "__main__":
    main()
