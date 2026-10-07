"""Ablation: does DeepPATHO's Multi-Magnification Attention (MMA) actually
add anything over UNI's raw pooled embeddings for EGFR, or is UNI's
transfer-learning advantage (Experiment B/UNI-frozen: slide F1=0.250,
AUC=0.615) coming entirely from the backbone, with MMA contributing little?

Model: concat(UNI local 1024-dim pooled embedding, UNI global 1024-dim
pooled embedding) -> small MLP -> sigmoid. No MMA, no spatial attention --
the two magnifications' pooled vectors are simply concatenated. This is the
"naive" version of transfer learning (frozen feature extractor + linear/MLP
head) that the project's original plan called out as a separate condition
from the full MMA-based transfer learning already run in uni_egfr_train.py.

Uses cache_uni_embeddings.py's cached (1024,) pooled UNI embeddings instead
of re-reading patches from .svs, since the backbone is frozen and its
output never changes -- this is what makes training here take seconds
instead of ~3.5h, and is what makes a real multi-seed run affordable.

Runs N_SEEDS training seeds automatically (same train/val/test patients
each time -- only the MLP's weight init and minibatch order vary) and
reports mean +/- std, addressing "this whole project is one seed" the
direct way: an actual repeated-seeds variance estimate, not just a single
run's bootstrap CI over the fixed test predictions.

Usage:
    python -m deeppatho_gen.uni_embedding_ablation --embeddings data/processed/uni_embeddings_egfr.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .metrics import find_best_threshold, patch_level_metrics, slide_level_metrics

N_SEEDS = 5
SEEDS = [20240924, 1, 2, 3, 4]


class EmbeddingMLP(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden=(256, 64)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_one_seed(data: dict, seed: int, device, max_epochs=200, patience=15, lr=1e-3, weight_decay=1e-4) -> dict:
    torch.manual_seed(seed)
    x = torch.cat([data["local_emb"], data["global_emb"]], dim=1).to(device)
    y = data["label"].float().unsqueeze(1).to(device)
    split = np.array(data["split"])

    idx_train = np.where(split == "train")[0]
    idx_val = np.where(split == "val")[0]
    idx_test = np.where(split == "test")[0]

    n_pos = int(y[idx_train].sum().item())
    n_neg = len(idx_train) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, device=device)

    model = EmbeddingMLP(in_dim=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    best_val_loss, best_state, patience_left = float("inf"), None, patience
    rng = np.random.default_rng(seed)
    batch_size = 512

    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(len(idx_train))
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            xb, yb = x_train[b], y_train[b]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = loss_fn(val_logits, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(x_val)).cpu().numpy()[:, 0]
        test_probs = torch.sigmoid(model(x_test)).cpu().numpy()[:, 0]

    val_file_ids = np.array(data["file_id"])[idx_val]
    val_patch_df = pd.DataFrame({"file_id": val_file_ids, "prob": val_probs, "label": y_val.cpu().numpy()[:, 0]})
    patch_thr = find_best_threshold(val_patch_df["label"].values, val_patch_df["prob"].values)
    val_slide = val_patch_df.groupby("file_id").agg(prob=("prob", "mean"), label=("label", "first"))
    slide_thr = find_best_threshold(val_slide["label"].values, val_slide["prob"].values)

    test_file_ids = np.array(data["file_id"])[idx_test]
    test_patch_df = pd.DataFrame({"file_id": test_file_ids, "prob": test_probs, "label": y_test.cpu().numpy()[:, 0]})
    patch_m = patch_level_metrics(test_patch_df, threshold=patch_thr)
    slide_m, _ = slide_level_metrics(test_patch_df, threshold=slide_thr)

    return {"seed": seed, "n_epochs": epoch + 1, "best_val_loss": best_val_loss,
            "patch_threshold": patch_thr, "slide_threshold": slide_thr,
            "test_patch_metrics": patch_m, "test_slide_metrics": slide_m}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", default="data/processed/uni_embeddings_egfr.pt")
    ap.add_argument("--out", default="runs/egfr_uni_embedding_ablation/results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    data = torch.load(args.embeddings)
    print(f"Loaded {len(data['file_id'])} cached embeddings "
          f"(local {tuple(data['local_emb'].shape)}, global {tuple(data['global_emb'].shape)})")

    runs = []
    for seed in SEEDS:
        r = run_one_seed(data, seed, device)
        runs.append(r)
        pm, sm = r["test_patch_metrics"], r["test_slide_metrics"]
        print(f"seed={seed} epochs={r['n_epochs']:3d} | "
              f"patch F1={pm['f1']:.3f} AUC={pm['roc_auc']:.3f} | "
              f"slide F1={sm['f1']:.3f} AUC={sm['roc_auc']:.3f}")

    def agg(key, metric):
        vals = [r[key][metric] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for level in ["test_patch_metrics", "test_slide_metrics"]:
        summary[level] = {}
        for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
            mean, std = agg(level, metric)
            summary[level][metric] = {"mean": round(mean, 4), "std": round(std, 4)}

    print("\n=== Summary across", len(SEEDS), "seeds (mean +/- std) ===")
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
