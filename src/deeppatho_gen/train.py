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
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchPairDataset, build_patch_index, case_level_split
from .model.backbone import DeepPATHOClassifier


def limit_cpu_threads(n: int = 2) -> None:
    """Cap PyTorch's intra-op BLAS thread pool.

    Diagnosed on the pilot run: with num_workers=0, this process alone was
    pegging ~12 CPU cores (per `top`) despite the model being tiny (2M
    params) -- BLAS was over-threading a CPU-bound *I/O* workload (openslide
    reads), not compute. That starves everyone else on this shared server
    without actually making training faster, since the real bottleneck is
    disk/openslide latency per patch, which extra BLAS threads can't help.
    The actual fix is more DataLoader *workers* (parallel I/O), each with a
    small thread budget so per-worker BLAS calls don't themselves oversubscribe.
    """
    torch.set_num_threads(n)


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

    limit_cpu_threads(cfg["data"].get("torch_threads", 2))

    index_df = build_patch_index(
        patch_coords_csv=cfg["data"]["index_csv"],
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

    train_ds = PatchPairDataset(index_df[index_df.split == "train"], gene_cols, augment=True)
    val_ds = PatchPairDataset(index_df[index_df.split == "val"], gene_cols, augment=False)
    test_df = index_df[index_df.split == "test"]

    # PatchPairDataset opens openslide handles lazily inside __getitem__, so
    # each forked worker process opens its own handles from a cold cache --
    # safe to parallelize. This is what actually speeds things up: the pilot
    # run's bottleneck was serial per-patch disk/openslide I/O in the main
    # process, not GPU compute (see limit_cpu_threads() above).
    num_workers = cfg["data"].get("num_workers", 4)
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["optim"]["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["optim"]["batch_size"], shuffle=False, **loader_kwargs)

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
    start_epoch = 0

    # Resume support: if a crash/kill/OOM interrupted a previous run in this
    # same output_dir, a watchdog can just re-invoke this script and it
    # picks back up from the last completed epoch instead of losing all
    # progress and starting a fresh random model.
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
        print(f"Resumed from {resume_path}: starting at epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}, patience_left={patience_left}")

    # Hard wall-clock budget: under a deadline, we need *a* result no matter
    # what epoch we're on, so this takes priority over early-stopping
    # patience and the configured epoch count. Checked between epochs (not
    # mid-epoch) so every epoch that starts finishes and gets a checkpoint.
    max_hours = cfg["optim"].get("max_hours")
    deadline = time.time() + max_hours * 3600 if max_hours else None
    t_start = time.time()

    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        train_loss, _, _ = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)

        elapsed_h = (time.time() - t_start) / 3600
        print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  elapsed={elapsed_h:.2f}h")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_hours": elapsed_h})
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

        torch.save(model.state_dict(), out_dir / "last_model.pt")  # always available even if best_model is stale
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "patience_left": patience_left,
        }, out_dir / "resume_checkpoint.pt")

        if deadline and time.time() > deadline:
            print(f"Stopping: hit the {max_hours}h training time budget after epoch {epoch}.")
            break

    print(f"Best val loss: {best_val_loss:.4f}. Model checkpoint: {out_dir / 'best_model.pt'}")
    if resume_path.exists() and epoch >= cfg["optim"]["epochs"] - 1:
        resume_path.unlink()  # naturally finished all epochs; no need to resume further

    # Auto-generate test-set predictions from whichever checkpoint is best,
    # so a downstream `evaluate.py` call always has something to consume --
    # "training finished" should always mean "a result exists", even if we
    # got cut short by max_hours rather than converging or early-stopping.
    if len(test_df) > 0:
        print(f"Generating test-set patch predictions ({len(test_df)} patches) ...")
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
        model.eval()
        test_ds = PatchPairDataset(test_df, gene_cols)
        test_loader = DataLoader(test_ds, batch_size=cfg["optim"]["batch_size"], shuffle=False,
                                  num_workers=num_workers, pin_memory=(device.type == "cuda"))
        rows = []
        with torch.no_grad():
            for local_t, global_t, labels, meta in tqdm(test_loader, desc="test predictions"):
                probs = torch.sigmoid(model(local_t.to(device), global_t.to(device))).cpu().numpy()
                for i in range(len(probs)):
                    row = {"file_id": meta["file_id"][i], "case_id": meta["case_id"][i]}
                    row.update({g: float(probs[i, j]) for j, g in enumerate(gene_cols)})
                    rows.append(row)
        import pandas as pd  # local import: only needed on this path
        pd.DataFrame(rows).to_csv(out_dir / "test_patch_scores.csv", index=False)
        print(f"Wrote {out_dir / 'test_patch_scores.csv'} -- feed this to evaluate.py")


if __name__ == "__main__":
    main()
