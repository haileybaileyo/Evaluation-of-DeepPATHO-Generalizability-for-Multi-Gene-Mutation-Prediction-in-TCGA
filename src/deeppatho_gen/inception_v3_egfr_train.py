"""EGFR plain-Inception-v3 baseline (20x only, no MMA).

Relation to existing code: this is egfr_stratified_train.py's protocol
(patient-level stratified split, pos_weight for class imbalance, F1-maximizing
decision threshold fit on VALIDATION only, patch- and slide-level metrics)
with exactly one change: the model is torchvision's Inception v3 on the 20x
patch alone (model/inception_v3_baseline.py) instead of the twin-Inception +
MMA DeepPATHOClassifier. build_index() and run_epoch() are imported unchanged
from egfr_stratified_train.py, so data, split and sampled patches are
identical to the other stratified EGFR experiments.

Additions over egfr_stratified_train.py: a "prediction_bias" block in the
result JSON (how many patches/slides are predicted positive vs. how many are
truly positive), to check the model is not predicting one class almost
exclusively.

Usage:
    python -m deeppatho_gen.inception_v3_egfr_train --config configs/egfr_inception_v3/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .dataset import PatchPairDataset, split_summary
from .egfr_stratified_train import build_index, run_epoch
from .metrics import find_best_threshold, patch_level_metrics, slide_level_metrics
from .model.inception_v3_baseline import InceptionV3Classifier
from .train import get_device, limit_cpu_threads


def prediction_bias(y_true, y_score, threshold: float) -> dict:
    """Fraction predicted positive vs. fraction truly positive. A model that
    collapses to one class shows pred_pos_rate ~0 (or ~1)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    pred = y_score >= threshold
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_pred_positive": int(pred.sum()),
        "pred_pos_rate": float(pred.mean()),
        "true_pos_rate": float(y_true.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/egfr_inception_v3/train_config.yaml")
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

    model = InceptionV3Classifier(num_classes=cfg["model"]["num_classes"], pretrained=cfg["model"]["pretrained"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: torchvision Inception v3, {n_params/1e6:.2f}M params, all trainable "
          f"({'ImageNet-pretrained' if cfg['model']['pretrained'] else 'from scratch'}), 20x patch only")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    n_pos = int((index_df[index_df.split == "train"]["label"] == 1).sum())
    n_neg = int((index_df[index_df.split == "train"]["label"] == 0).sum())
    pos_weight_value = n_neg / n_pos
    print(f"Train patches: {n_pos} positive / {n_neg} negative -> pos_weight={pos_weight_value:.3f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

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

        train_m = patch_level_metrics(train_patch_df)
        val_patch_m = patch_level_metrics(val_patch_df)
        val_slide_m, _ = slide_level_metrics(val_patch_df)
        elapsed_h = (time.time() - t_start) / 3600

        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_patch_auc": train_m["roc_auc"], "val_patch_auc": val_patch_m["roc_auc"],
            "val_patch_f1": val_patch_m["f1"], "val_slide_auc": val_slide_m["roc_auc"],
            "val_slide_f1": val_slide_m["f1"], "elapsed_hours": elapsed_h,
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_patch_auc={train_m['roc_auc']:.4f} val_patch_auc={val_patch_m['roc_auc']:.4f} "
              f"val_slide_auc={val_slide_m['roc_auc']:.4f} val_slide_f1={val_slide_m['f1']:.4f} "
              f"elapsed={elapsed_h:.2f}h", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = cfg["early_stopping_patience"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping (no val_loss improvement).")
                break

        torch.save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_val_loss": best_val_loss, "patience_left": patience_left,
        }, resume_path)

    print(f"\nBest val loss: {best_val_loss:.4f}")

    result = {
        "config": cfg, "seed": cfg["seed"], "split_seed": cfg["split_seed"],
        "model": "torchvision Inception v3, 20x only, no MMA "
                 f"({'ImageNet-pretrained' if cfg['model']['pretrained'] else 'from scratch'})",
        "pos_weight": pos_weight_value,
        "n_epochs_run": len(history), "best_val_loss": best_val_loss,
        "split_summary": summary.reset_index().to_dict(orient="records"),
    }

    if len(test_df_raw) > 0:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

        # Threshold is fit on VAL only (never test), using the best checkpoint.
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
        test_patch_m_05 = patch_level_metrics(test_patch_df, threshold=0.5)
        test_slide_m_05, _ = slide_level_metrics(test_patch_df, threshold=0.5)

        test_patch_df.to_csv(out_dir / "test_patch_scores.csv", index=False)
        test_slide_df.reset_index().to_csv(out_dir / "test_slide_scores.csv", index=False)

        result["patch_threshold"] = patch_threshold
        result["slide_threshold"] = slide_threshold
        result["test_patch_metrics"] = test_patch_m
        result["test_slide_metrics"] = test_slide_m
        result["test_patch_metrics_at_0.5"] = test_patch_m_05
        result["test_slide_metrics_at_0.5"] = test_slide_m_05
        result["prediction_bias"] = {
            "test_patch_at_val_threshold": prediction_bias(test_patch_df["label"], test_patch_df["prob"], patch_threshold),
            "test_patch_at_0.5": prediction_bias(test_patch_df["label"], test_patch_df["prob"], 0.5),
            "test_slide_at_val_threshold": prediction_bias(test_slide_df["label"], test_slide_df["prob"], slide_threshold),
            "test_slide_at_0.5": prediction_bias(test_slide_df["label"], test_slide_df["prob"], 0.5),
        }

        print(f"\n=== TEST (patch-level, threshold={patch_threshold:.4f}) ===")
        print(json.dumps(test_patch_m, indent=2))
        print(f"\n=== TEST (slide-level, mean-probability aggregation, threshold={slide_threshold:.4f}) ===")
        print(json.dumps(test_slide_m, indent=2))
        print("\n=== Prediction bias ===")
        print(json.dumps(result["prediction_bias"], indent=2))

    (out_dir / "test_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote full results to {out_dir / 'test_result.json'}")

    if resume_path.exists() and epoch >= cfg["max_epochs"] - 1:
        resume_path.unlink()


if __name__ == "__main__":
    main()
