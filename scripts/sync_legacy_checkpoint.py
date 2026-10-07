"""Keeps resume_checkpoint.pt + history.json fresh while the CURRENTLY
RUNNING train.py process (started before resume support existed, so it has
no idea this file should exist) is still alive.

Rationale: editing a running process's already-loaded Python code has no
effect on that process, so the live training run will keep saving
best_model.pt/last_model.pt each epoch but never write its own
resume_checkpoint.pt. If it crashes, a restart would otherwise lose all
progress and start from a random model. This script polls the training log
for newly completed epochs and reconstructs an equivalent resume checkpoint
after each one, so a crash at any point only loses at most one epoch, not
the whole run. Stops on its own once the training log shows the run
finished (successfully or not) -- at that point either a fresh v2-managed
run's own native resume-writing takes over, or there's nothing left to sync.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "src")
from deeppatho_gen.model.backbone import DeepPATHOClassifier  # noqa: E402

TRAIN_LOG = Path("runs/full_experiment/logs/03_train.log")
OUT_DIR = Path("runs/phase1_gene_mutation")
EPOCH_RE = re.compile(r"epoch (\d+)\s+train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+elapsed=([\d.]+)h")


def load_model(cfg, gene_cols):
    return DeepPATHOClassifier(
        num_classes=len(gene_cols),
        dropout_local=cfg["model"]["dropout_local"],
        dropout_global=cfg["model"]["dropout_global"],
    )


def main():
    cfg = yaml.safe_load(open("configs/train_config_runtime.yaml"))
    genes_cfg = yaml.safe_load(open("configs/genes.yaml"))
    gene_cols = list(genes_cfg["phase1_genes"])
    patience_cfg = cfg["optim"]["early_stopping_patience"]

    seen_epochs = set()
    history = json.loads((OUT_DIR / "history.json").read_text()) if (OUT_DIR / "history.json").exists() else []
    for h in history:
        seen_epochs.add(h["epoch"])

    print(f"Syncer watching {TRAIN_LOG}, already have epochs {sorted(seen_epochs)}", flush=True)

    while True:
        if not TRAIN_LOG.exists():
            time.sleep(30)
            continue
        text = TRAIN_LOG.read_text(errors="ignore")
        matches = EPOCH_RE.findall(text)
        new_ones = [m for m in matches if int(m[0]) not in seen_epochs]

        for epoch_s, train_loss_s, val_loss_s, elapsed_s in new_ones:
            epoch, train_loss, val_loss, elapsed = int(epoch_s), float(train_loss_s), float(val_loss_s), float(elapsed_s)
            seen_epochs.add(epoch)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_hours": elapsed})
            history.sort(key=lambda h: h["epoch"])
            (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

            best_val_loss = min(h["val_loss"] for h in history)
            # Recompute patience the same way train.py does: resets to full
            # patience on every new best, decrements otherwise.
            patience_left = patience_cfg
            running_best = float("inf")
            for h in history:
                if h["val_loss"] < running_best:
                    running_best = h["val_loss"]
                    patience_left = patience_cfg
                else:
                    patience_left -= 1

            best_ckpt = OUT_DIR / "best_model.pt"
            if best_ckpt.exists():
                model = load_model(cfg, gene_cols)
                model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "patience_left": patience_left,
                }, OUT_DIR / "resume_checkpoint.pt")
                print(f"Synced resume_checkpoint.pt after epoch {epoch} "
                      f"(best_val_loss={best_val_loss:.4f}, patience_left={patience_left})", flush=True)

        if "Best val loss:" in text or "Early stopping" in text:
            print("Training log shows the run finished; syncer exiting.", flush=True)
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
