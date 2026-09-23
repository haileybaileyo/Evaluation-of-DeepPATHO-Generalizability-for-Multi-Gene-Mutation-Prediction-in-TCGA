"""D: fixed patient-level train/val run for a single gene (TP53), using the
exact same case_level_split as production train.py.

Only runs after B and C (overfit_test.py) both PASS -- this script asserts
that and refuses to run otherwise, per the task's dependency structure.

Question this answers: given the implementation is confirmed capable of
memorizing (B/C both passed), does it generalize across held-out patients
at all, under production-like conditions (real train/val split, real class
balance, tumor filtering, augmentation)? Test set is never touched here --
this is diagnostic only.

Usage:
    python -m deeppatho_gen.diagnostics.fixed_validation --config configs/diagnostics/fixed_validation.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from ..dataset import PatchPairDataset, case_level_split
from ..model.backbone import DeepPATHOClassifier
from ..train import get_device, limit_cpu_threads


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3]).decode().strip()
    except Exception:
        return "unknown"


def require_bc_pass(overfit_dir: Path) -> None:
    for mode in ("clean", "shuffled_label"):
        result_path = overfit_dir / mode / "result.json"
        if not result_path.exists():
            raise SystemExit(f"Prerequisite missing: {result_path} not found. Run overfit_test.py --mode {mode} first.")
        result = json.loads(result_path.read_text())
        if result["result"] != "PASS":
            raise SystemExit(f"Prerequisite failed: {mode} overfit test result was {result['result']}, not PASS. "
                              f"Per task spec, fixed_validation must not run until B and C both PASS.")
    print("Prerequisite check: B (clean) and C (shuffled_label) both PASS -- proceeding.")


def cap_patches_per_slide(coords: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def sample_group(g):
        if len(g) <= cap:
            return g
        idx = rng.choice(g.index, size=cap, replace=False)
        return g.loc[idx]

    return coords.groupby("file_id", group_keys=False).apply(sample_group).reset_index(drop=True)


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_probs, all_labels, all_file_ids = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for local_t, global_t, y, meta in loader:
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

    return total_loss / max(n_batches, 1), pd.DataFrame({"file_id": all_file_ids, "prob": all_probs, "label": all_labels})


def slide_level_metrics(patch_df: pd.DataFrame) -> dict:
    slide = patch_df.groupby("file_id").agg(prob=("prob", "mean"), label=("label", "first"))
    has_both = slide["label"].nunique() == 2
    auc = roc_auc_score(slide["label"], slide["prob"]) if has_both else float("nan")
    preds = (slide["prob"] >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(slide["label"], preds) if has_both else float("nan")
    cm = confusion_matrix(slide["label"], preds, labels=[0, 1]).tolist() if has_both else None
    return {"auc": auc, "balanced_acc": bal_acc, "confusion_matrix": cm, "n_slides": len(slide)}, slide


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/diagnostics/fixed_validation.yaml")
    ap.add_argument("--overfit-dir", default="runs/diagnostics/overfit_tp53")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    require_bc_pass(Path(args.overfit_dir))

    torch.manual_seed(cfg["seed"])
    limit_cpu_threads(cfg["torch_threads"])

    coords = pd.read_csv(cfg["patch_coords"])
    labels = pd.read_csv(cfg["labels"])
    labels_by_case = labels.groupby("case_id")[cfg["gene"]].max().reset_index().rename(columns={cfg["gene"]: "label"})

    index_df = coords.merge(labels_by_case, on="case_id", how="inner")
    split = case_level_split(index_df["case_id"].tolist(), train_frac=cfg["train_frac"], val_frac=cfg["val_frac"], seed=cfg["split_seed"])
    index_df["split"] = index_df["case_id"].map(split)

    index_df = cap_patches_per_slide(index_df, cfg["max_patches_per_slide"], cfg["seed"])

    device = get_device()
    train_ds = PatchPairDataset(index_df[index_df.split == "train"], ["label"], augment=cfg["augment"])
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], ["label"], augment=False)

    print(f"train: {len(train_ds)} patches / {index_df[index_df.split=='train'].case_id.nunique()} patients")
    print(f"val:   {len(val_ds)} patches / {index_df[index_df.split=='val'].case_id.nunique()} patients")

    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    model = DeepPATHOClassifier(num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_key = cfg["checkpoint_selection_metric"]
    assert metric_key in ("val_loss", "val_auc"), "checkpoint_selection_metric must be exactly one of val_loss/val_auc"
    best_metric = float("inf") if metric_key == "val_loss" else -float("inf")
    patience_left = cfg["early_stopping_patience"]
    history = []
    best_val_slide_df = None

    for epoch in range(cfg["max_epochs"]):
        train_loss, train_patch_df = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_patch_df = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        train_m, _ = slide_level_metrics(train_patch_df)
        val_m, val_slide_df = slide_level_metrics(val_patch_df)

        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_auc": train_m["auc"], "val_auc": val_m["auc"],
            "train_balanced_acc": train_m["balanced_acc"], "val_balanced_acc": val_m["balanced_acc"],
            "val_confusion_matrix": val_m["confusion_matrix"], "n_val_slides": val_m["n_slides"],
        }
        history.append(row)
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_auc={train_m['auc']:.4f} val_auc={val_m['auc']:.4f} val_bal_acc={val_m['balanced_acc']:.4f}")

        current = val_loss if metric_key == "val_loss" else val_m["auc"]
        improved = (current < best_metric) if metric_key == "val_loss" else (current > best_metric)
        if improved:
            best_metric = current
            patience_left = cfg["early_stopping_patience"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            best_val_slide_df = val_slide_df.copy()
            best_val_slide_df["epoch"] = epoch
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_json(out_dir / "epoch_metrics.json", orient="records", indent=2)
    pd.DataFrame(history).to_csv(out_dir / "epoch_metrics.csv", index=False)
    if best_val_slide_df is not None:
        best_val_slide_df.reset_index().to_csv(out_dir / "best_val_slide_predictions.csv", index=False)

    meta = {
        "gene": cfg["gene"], "git_commit": git_commit_hash(), "config": cfg,
        "n_epochs_run": len(history), "checkpoint_selection_metric": metric_key,
        "best_metric_value": best_metric,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nBest {metric_key}: {best_metric:.4f}. Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
