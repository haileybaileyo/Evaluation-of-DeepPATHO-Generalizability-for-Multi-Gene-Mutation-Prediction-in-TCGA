"""EGFR, UNI backbone + MMA (Experiment B: frozen transfer learning, or
Experiment D: fine-tuning, selected by `model.freeze_backbone` in the
config) -- the UNI counterpart to roadmap_egfr_train.py's Phikon-v2 run.

Relation to existing code: same PatchPairDataset / MMA head structure as
roadmap_egfr_train.py, with three differences:
  1. UNIDeepPATHOClassifier (model/uni_backbone.py) instead of
     PhikonDeepPATHOClassifier -- a different frozen ViT backbone.
  2. stratified_case_level_split (dataset.py), not case_level_split --
     same fix as egfr_stratified_train.py, for the same reason (EGFR's
     ~12% positive rate needs stratification to avoid split ratio drift).
  3. metrics.py's full Precision/Recall/F1/ROC-AUC/PR-AUC at both patch
     and slide level, not just AUC/balanced_accuracy.

`model.freeze_backbone: true`  -> Experiment B (frozen UNI features, only
                                   MMA + FC head trained: transfer learning).
`model.freeze_backbone: false` -> Experiment D (full fine-tuning: UNI's own
                                   weights are also updated). Needs a much
                                   lower backbone LR than the frozen run's
                                   head-only LR or the pretrained features
                                   collapse in a few steps -- see
                                   `backbone_lr` in the config.

Usage:
    python -m deeppatho_gen.uni_egfr_train --config configs/egfr_uni/train_config.yaml
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
from .metrics import patch_level_metrics, slide_level_metrics
from .model.uni_backbone import UNIDeepPATHOClassifier
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


def build_optimizer(model: UNIDeepPATHOClassifier, cfg: dict):
    """Frozen run: single param group (head + MMA) at cfg['lr'].
    Fine-tune run: two param groups -- head+MMA at cfg['lr'], UNI backbone
    at cfg['backbone_lr'] (must be much smaller, or the pretrained features
    are destroyed in the first few steps of BCE loss on 1.88M-parameter-
    scale gradients flowing into a 303M-parameter pretrained ViT)."""
    if cfg["model"]["freeze_backbone"]:
        return torch.optim.Adam(model.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    head_params = list(model.mma.parameters()) + list(model.head.parameters())
    backbone_params = list(model.encoder.parameters())
    return torch.optim.Adam([
        {"params": head_params, "lr": cfg["lr"]},
        {"params": backbone_params, "lr": cfg["backbone_lr"]},
    ], weight_decay=cfg["weight_decay"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/egfr_uni/train_config.yaml")
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

    freeze_backbone = cfg["model"]["freeze_backbone"]
    train_ds = PatchPairDataset(index_df[index_df.split == "train"], ["label"], augment=cfg["augment"])
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], ["label"], augment=False)
    test_df_raw = index_df[index_df.split == "test"]

    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    model = UNIDeepPATHOClassifier(num_classes=cfg["model"]["num_classes"], freeze_backbone=freeze_backbone).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total/1e6:.1f}M total params, {n_trainable/1e6:.2f}M trainable "
          f"({'frozen backbone' if freeze_backbone else 'fine-tuning backbone'})")

    optimizer = build_optimizer(model, cfg)
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
        "model": f"UNIDeepPATHOClassifier (freeze_backbone={freeze_backbone})",
        "n_epochs_run": len(history), "best_val_loss": best_val_loss,
        "split_summary": summary.reset_index().to_dict(orient="records"),
    }

    if len(test_df_raw) > 0:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
        test_ds = PatchPairDataset(test_df_raw, ["label"], augment=False)
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
        _, test_patch_df = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)

        test_patch_m = patch_level_metrics(test_patch_df)
        test_slide_m, test_slide_df = slide_level_metrics(test_patch_df)

        test_patch_df.to_csv(out_dir / "test_patch_scores.csv", index=False)
        test_slide_df.reset_index().to_csv(out_dir / "test_slide_scores.csv", index=False)

        result["test_patch_metrics"] = test_patch_m
        result["test_slide_metrics"] = test_slide_m

        print("\n=== TEST (patch-level) ===")
        print(json.dumps(test_patch_m, indent=2))
        print("\n=== TEST (slide-level, mean-probability aggregation) ===")
        print(json.dumps(test_slide_m, indent=2))

    (out_dir / "test_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote full results to {out_dir / 'test_result.json'}")

    if resume_path.exists() and epoch >= cfg["max_epochs"] - 1:
        resume_path.unlink()


if __name__ == "__main__":
    main()
