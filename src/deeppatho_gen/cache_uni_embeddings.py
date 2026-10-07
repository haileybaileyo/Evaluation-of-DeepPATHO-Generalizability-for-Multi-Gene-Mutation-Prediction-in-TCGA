"""One-time extraction of pooled (1024-dim) UNI embeddings for every patch
in the stratified EGFR index (same split as egfr_uni/train_config.yaml),
saved to a single tensor file.

Why: uni_egfr_train.py re-runs the frozen 303M-param UNI ViT forward pass
on every patch, every epoch -- that forward pass (not disk I/O) is why a
full run takes ~3.5h even though only the 8.5M-param MMA+head is actually
learning anything. Since the backbone is frozen, its output for a given
patch never changes across epochs or across reruns with a different
training seed. Pre-computing it once and caching the pooled vector lets:
  1. A "UNI embedding + MLP, no MMA" ablation (uni_embedding_ablation.py)
     train in seconds instead of hours.
  2. Multiple training seeds of that ablation to actually be affordable
     within a few-hour budget.

Only the GLOBALLY POOLED embedding (1024,) is cached, not the (1024,14,14)
spatial map MMA needs -- caching the spatial map for ~160k patches x 2
magnifications would be ~500GB (infeasible on this disk: 350GB free). This
means the cache is only usable for the no-MMA ablation, not for re-running
the full UNIDeepPATHOClassifier faster.

Usage:
    python -m deeppatho_gen.cache_uni_embeddings --config configs/egfr_uni/train_config.yaml \
        --out data/processed/uni_embeddings_egfr.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PatchPairDataset
from .model.uni_encoder import UNIPatchEncoder
from .train import get_device, limit_cpu_threads
from .uni_egfr_train import build_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/egfr_uni/train_config.yaml")
    ap.add_argument("--out", default="data/processed/uni_embeddings_egfr.pt")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=3)  # modest: shares the box with any other running job
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    limit_cpu_threads(cfg["torch_threads"])
    device = get_device()
    print(f"Using device: {device}")

    index_df = build_index(cfg)
    print(f"Total patches to embed: {len(index_df)}")

    encoder = UNIPatchEncoder(freeze=True).to(device).eval()
    gap = torch.nn.AdaptiveAvgPool2d(1)

    ds = PatchPairDataset(index_df, ["label"], augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    local_embs, global_embs, labels, file_ids, case_ids, splits = [], [], [], [], [], []
    split_by_pos = index_df["split"].tolist()  # aligned with ds/loader order (no shuffle)

    idx = 0
    with torch.no_grad():
        for local_t, global_t, y, meta in tqdm(loader):
            local_t, global_t = local_t.to(device), global_t.to(device)
            fl = gap(encoder(local_t)).flatten(1)   # (B, 1024)
            fg = gap(encoder(global_t)).flatten(1)  # (B, 1024)
            local_embs.append(fl.cpu())
            global_embs.append(fg.cpu())
            labels.append(y[:, 0].cpu())
            file_ids.extend(meta["file_id"])
            case_ids.extend(meta["case_id"])
            b = local_t.shape[0]
            splits.extend(split_by_pos[idx: idx + b])
            idx += b

    out = {
        "local_emb": torch.cat(local_embs),
        "global_emb": torch.cat(global_embs),
        "label": torch.cat(labels),
        "file_id": file_ids,
        "case_id": case_ids,
        "split": splits,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Wrote {len(file_ids)} embeddings ({out['local_emb'].shape}, {out['global_emb'].shape}) to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
