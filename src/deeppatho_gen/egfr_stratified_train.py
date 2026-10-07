"""EGFR baseline, patch-level, with a PATIENT-LEVEL STRATIFIED split.

Relation to existing code: this is egfr_scratch_train.py's from-scratch
twin-Inception + MMA model (model/backbone.py's DeepPATHOClassifier) and
training loop, with four changes:

  1. Split: stratified_case_level_split (dataset.py) instead of
     case_level_split. The latter is a plain random shuffle of case_ids and,
     for a label as imbalanced as EGFR (~12% positive), produced very
     different positive ratios across train/val/test by chance (previously
     observed 13%/15%/6%). Stratifying on the case-level EGFR label keeps
     each split close to the overall ratio while still splitting on
     case_id, so no patient's patches cross a split boundary.

  2. Metrics: metrics.py's binary_metrics (Precision/Recall/F1/ROC-AUC/
     PR-AUC), computed at both patch level (the actual training/prediction
     unit) and slide level (mean-probability aggregation), instead of just
     AUC + balanced_accuracy.

  3. pos_weight: this project's first stratified run (no pos_weight, fixed
     0.5 threshold -- results archived under
     runs/egfr_stratified/mma_from_scratch_no_posweight/) collapsed to
     predicting "negative" for every single patch: test precision/recall/F1
     all ~0, slide-level AUC 0.41 (worse than chance). Root cause: with
     train patches at 93999 negative / 16379 positive (ratio 5.74), plain
     BCEWithLogitsLoss's easiest way to minimize average loss is to push
     every prediction toward the ~12% base rate, which never crosses a 0.5
     threshold. pos_weight=5.74 (n_neg/n_pos on the train split, computed
     automatically in build_index/main below) rebalances the gradient so
     misclassifying a positive costs as much as misclassifying ~5.7
     negatives.

  4. Threshold: the previous run's threshold (0.5) was never actually
     justified -- it's a default, not a fit value. find_best_threshold
     (metrics.py) now picks the F1-maximizing threshold on the VALIDATION
     set only (never test, which would leak test information into the
     reported result), and that fitted threshold -- not 0.5 -- is what test
     precision/recall/F1 are computed at. ROC-AUC/PR-AUC stay
     threshold-free as before.

Everything else (resume/checkpoint support, tumor-only patch source,
max_patches_per_slide capping, CPU thread limiting) is unchanged from
egfr_scratch_train.py.

Usage:
    python -m deeppatho_gen.egfr_stratified_train --config configs/egfr_stratified/train_config.yaml
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
from .metrics import binary_metrics, find_best_threshold, patch_level_metrics, slide_level_metrics
from .model.backbone import DeepPATHOClassifier
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
    ap.add_argument("--config", default="configs/egfr_stratified/train_config.yaml")
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

    model = DeepPATHOClassifier(num_classes=cfg["model"]["num_classes"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params, all trainable (from-scratch)")

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
              f"val_patch_auc={val_patch_m['roc_auc']:.4f} val_slide_auc={val_slide_m['roc_auc']:.4f} "
              f"val_slide_f1={val_slide_m['f1']:.4f} elapsed={elapsed_h:.2f}h")

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
        "model": "DeepPATHOClassifier (from-scratch twin-Inception + MMA)",
        "pos_weight": pos_weight_value,
        "n_epochs_run": len(history), "best_val_loss": best_val_loss,
        "split_summary": summary.reset_index().to_dict(orient="records"),
    }

    if len(test_df_raw) > 0:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))

        # Fit the decision threshold on VAL only (never test) -- reusing the
        # best checkpoint's own validation pass so this doesn't require a
        # second forward pass over val.
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
        # Also report the old fixed-0.5 numbers side by side, so the effect of
        # threshold tuning (vs. pos_weight alone) is visible in the results.
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

        print(f"\n=== TEST (patch-level, threshold={patch_threshold:.4f}) ===")
        print(json.dumps(test_patch_m, indent=2))
        print(f"\n=== TEST (slide-level, mean-probability aggregation, threshold={slide_threshold:.4f}) ===")
        print(json.dumps(test_slide_m, indent=2))

    (out_dir / "test_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote full results to {out_dir / 'test_result.json'}")

    if resume_path.exists() and epoch >= cfg["max_epochs"] - 1:
        resume_path.unlink()


if __name__ == "__main__":
    main()
