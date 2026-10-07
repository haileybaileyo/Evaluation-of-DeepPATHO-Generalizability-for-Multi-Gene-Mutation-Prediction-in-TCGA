"""B & C: small-subset overfit test (clean labels) and shuffled-label control.

Question this answers: can the model/data-path/loss/gradient plumbing
memorize a tiny, fixed training set at all? If B fails, the bug is
upstream of "does the architecture generalize" -- something in the
data path, loss, or optimization is broken. If B passes but C (labels
permuted) also passes trivially the same way, that's expected (a model
that can memorize real labels can memorize random ones too -- C's real
purpose is to confirm the loader->label->loss wiring is truly live, not
silently a no-op that would let loss appear to drop for an unrelated
reason). If B FAILS while C also fails identically, suspect a
data-path/optimization bug rather than a generalization limit.

Reproducibility: the 8 selected slides and every chosen patch index are
written to disk on first run and reloaded on subsequent runs with the
same selection_seed, so re-running this script (e.g. for --mode
shuffled_label) trains on the exact same patches as the clean run.

Usage:
    python -m deeppatho_gen.diagnostics.overfit_test --mode clean --config configs/diagnostics/overfit_test.yaml
    python -m deeppatho_gen.diagnostics.overfit_test --mode shuffled_label --config configs/diagnostics/overfit_test.yaml
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..dataset import PatchPairDataset
from ..model.backbone import DeepPATHOClassifier
from ..train import get_device, limit_cpu_threads


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3]).decode().strip()
    except Exception:
        return "unknown"


def select_slides(coords: pd.DataFrame, labels: pd.DataFrame, gene: str, n_per_class: int, seed: int) -> pd.DataFrame:
    """Pick n_per_class positive-patient and n_per_class negative-patient
    slides, preferring distinct patients, one slide per patient."""
    labels_by_case = labels.groupby("case_id")[gene].max()
    slide_to_case = coords[["file_id", "case_id"]].drop_duplicates().set_index("file_id")["case_id"]

    per_slide_label = slide_to_case.map(labels_by_case)
    candidates = pd.DataFrame({"file_id": slide_to_case.index, "case_id": slide_to_case.values, "label": per_slide_label.values}).dropna()
    candidates["label"] = candidates["label"].astype(int)

    rng = np.random.default_rng(seed)

    def pick(label_value: int) -> pd.DataFrame:
        pool = candidates[candidates.label == label_value].drop_duplicates(subset="case_id")  # 1 slide/patient
        if len(pool) < n_per_class:
            raise SystemExit(f"Not enough distinct-patient slides with label={label_value}: have {len(pool)}, need {n_per_class}")
        idx = rng.choice(pool.index, size=n_per_class, replace=False)
        return pool.loc[idx]

    chosen = pd.concat([pick(1), pick(0)], ignore_index=True)
    return chosen


def select_patches(coords: pd.DataFrame, slide_ids: list[str], per_slide: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)  # offset from slide-selection seed
    rows = []
    for i, fid in enumerate(slide_ids):
        pool = coords[coords.file_id == fid]
        n = min(per_slide, len(pool))
        idx = rng.choice(pool.index, size=n, replace=False)
        rows.append(pool.loc[idx])
    return pd.concat(rows, ignore_index=True)


def compute_grad_norms(model: DeepPATHOClassifier) -> dict[str, float]:
    def norm(params) -> float:
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += p.grad.detach().float().norm() ** 2
        return float(sq ** 0.5)

    return {
        "grad_norm_total": norm(model.parameters()),
        "grad_norm_mma": norm(model.mma.parameters()),
        "grad_norm_head": norm(model.head.parameters()),
        "grad_norm_towers": norm(list(model.tower_local.parameters()) + list(model.tower_global.parameters())),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/diagnostics/overfit_test.yaml")
    ap.add_argument("--mode", choices=["clean", "shuffled_label"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["output_dir"]) / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["selection_seed"])
    limit_cpu_threads(2)

    coords = pd.read_csv(cfg["patch_coords"])
    labels = pd.read_csv(cfg["labels"])

    selection_path = Path(cfg["output_dir"]) / "selected_slides.csv"
    patches_path = Path(cfg["output_dir"]) / "selected_patches.csv"
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if selection_path.exists():
        chosen_slides = pd.read_csv(selection_path)
        print(f"Reusing existing slide selection from {selection_path}")
    else:
        chosen_slides = select_slides(coords, labels, cfg["gene"], cfg["n_slides_per_class"], cfg["selection_seed"])
        chosen_slides.to_csv(selection_path, index=False)

    if patches_path.exists():
        patch_df = pd.read_csv(patches_path)
        print(f"Reusing existing patch selection from {patches_path}")
    else:
        patch_df = select_patches(coords, chosen_slides["file_id"].tolist(), cfg["patches_per_slide"], cfg["selection_seed"])
        patch_df.to_csv(patches_path, index=False)

    true_label_by_slide = chosen_slides.set_index("file_id")["label"].to_dict()

    if args.mode == "shuffled_label":
        rng = np.random.default_rng(cfg["selection_seed"] + 999)
        slide_ids = list(true_label_by_slide.keys())
        shuffled_values = rng.permutation(list(true_label_by_slide.values()))
        label_by_slide = dict(zip(slide_ids, shuffled_values))
        pd.DataFrame({"file_id": slide_ids,
                       "true_label": [true_label_by_slide[s] for s in slide_ids],
                       "shuffled_label": [label_by_slide[s] for s in slide_ids]}).to_csv(out_dir / "label_permutation.csv", index=False)
    else:
        label_by_slide = true_label_by_slide

    patch_df = patch_df.copy()
    patch_df["label"] = patch_df["file_id"].map(label_by_slide)
    gene_cols = ["label"]

    device = get_device()
    train_ds = PatchPairDataset(patch_df, gene_cols, augment=cfg["augment"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)

    model = DeepPATHOClassifier(num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    consecutive_pass = 0
    crit = cfg["success_criteria"]

    for epoch in range(cfg["max_epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        all_probs, all_labels, all_file_ids = [], [], []
        grad_norms_epoch = []
        nan_or_inf = False

        for local_t, global_t, y, meta in train_loader:
            local_t, global_t, y = local_t.to(device), global_t.to(device), y.to(device)
            logits = model(local_t, global_t)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            grad_norms_epoch.append(compute_grad_norms(model))
            optimizer.step()

            if torch.isnan(loss) or torch.isinf(loss):
                nan_or_inf = True

            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            all_probs.extend(probs.tolist())
            all_labels.extend(y.detach().cpu().numpy()[:, 0].tolist())
            all_file_ids.extend(meta["file_id"])

        train_loss = total_loss / max(n_batches, 1)

        # slide-level aggregation (mean probability) for the AUC/report
        pred_df = pd.DataFrame({"file_id": all_file_ids, "prob": all_probs, "label": all_labels})
        slide_probs = pred_df.groupby("file_id").agg(prob=("prob", "mean"), label=("label", "first"))
        has_both_classes = slide_probs["label"].nunique() == 2
        train_auc = roc_auc_score(slide_probs["label"], slide_probs["prob"]) if has_both_classes else float("nan")
        train_acc = ((slide_probs["prob"] > 0.5).astype(int) == slide_probs["label"]).mean()

        avg_grad = {k: float(np.mean([g[k] for g in grad_norms_epoch])) for k in grad_norms_epoch[0]}

        row = {
            "epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc": float(train_acc),
            "lr": cfg["lr"], "nan_or_inf": nan_or_inf, **avg_grad,
        }
        for fid, p in slide_probs["prob"].items():
            row[f"prob_{fid[:8]}"] = float(p)
        history.append(row)

        if epoch % 5 == 0 or epoch == cfg["max_epochs"] - 1:
            print(f"[{args.mode}] epoch {epoch:03d} loss={train_loss:.4f} auc={train_auc:.4f} acc={train_acc:.4f} "
                  f"grad_total={avg_grad['grad_norm_total']:.4f}")

        # early-success check -- only eligible once we've run enough epochs
        # to actually evaluate the loss-improvement criterion. (Bug found
        # during this diagnostic's own first run: defaulting to
        # loss_improved=True before epoch>=n_back let AUC-only epochs count
        # toward early success, stopping training at epoch 5 with loss still
        # far from the target -- a false "criteria held" before the loss
        # criterion had ever actually been checked.)
        n_back = crit["loss_improve_over_last_n"]
        eligible = epoch >= n_back
        loss_improved = eligible and history[epoch]["train_loss"] < history[epoch - n_back]["train_loss"]
        is_pass = eligible and (not nan_or_inf) and has_both_classes and train_auc >= crit["min_train_auc"] and loss_improved
        consecutive_pass = consecutive_pass + 1 if is_pass else 0
        if consecutive_pass >= cfg["early_success_patience"]:
            print(f"Early success: PASS criteria held for {consecutive_pass} consecutive epochs.")
            break

    pd.DataFrame(history).to_csv(out_dir / "epoch_metrics.csv", index=False)

    final = history[-1]
    final_pass = (
        not final["nan_or_inf"]
        and not np.isnan(final["train_auc"]) and final["train_auc"] >= crit["min_train_auc"]
        and final["train_loss"] <= crit["max_final_loss"]
    )

    slide_report = slide_probs.reset_index()
    slide_report["mode"] = args.mode
    slide_report.to_csv(out_dir / "slide_predictions.csv", index=False)

    meta = {
        "mode": args.mode, "gene": cfg["gene"], "git_commit": git_commit_hash(),
        "config": cfg, "n_epochs_run": len(history),
        "final_train_loss": final["train_loss"], "final_train_auc": final["train_auc"],
        "final_train_acc": final["train_acc"], "any_nan_or_inf": bool(pd.DataFrame(history)["nan_or_inf"].any()),
        "result": "PASS" if final_pass else "FAIL",
    }
    (out_dir / "result.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"\n=== {args.mode.upper()} RESULT: {meta['result']} ===")
    print(f"final_train_loss={final['train_loss']:.4f}  final_train_auc={final['train_auc']:.4f}  "
          f"final_train_acc={final['train_acc']:.4f}  nan_or_inf_ever={meta['any_nan_or_inf']}")
    print(f"Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
