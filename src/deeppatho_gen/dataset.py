"""PyTorch Dataset: reads matched (20x, 5x) patch pairs on the fly from the
open .svs files, using the coordinate index from preprocessing/tile_wsi.py
joined against the multi-gene label table from data/fetch_mutation_labels.py.

Only tiles from slides whose case survives the case-level train/val/test
split are ever loaded by a given split's Dataset, so no patient's tiles
leak across sets (Coudray et al.'s "never trained and tested on tiles from
the same tumor sample" requirement, generalized to case-level).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import openslide
except ImportError:  # pragma: no cover
    openslide = None

from .preprocessing.tile_wsi import PATCH_PX, GLOBAL_MAG, LOCAL_MAG, read_patch, SlideGeometry


def case_level_split(
    case_ids: list[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Deterministic case_id -> {'train','val','test'} assignment."""
    rng = np.random.default_rng(seed)
    ids = sorted(set(case_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    assignment = {}
    for i, cid in enumerate(ids):
        if i < n_train:
            assignment[cid] = "train"
        elif i < n_train + n_val:
            assignment[cid] = "val"
        else:
            assignment[cid] = "test"
    return assignment


def build_patch_index(patch_coords_csv: str, labels_csv: str) -> pd.DataFrame:
    coords = pd.read_csv(patch_coords_csv)
    labels = pd.read_csv(labels_csv)
    gene_cols = [c for c in labels.columns if c not in ("case_id", "sample_id")]
    labels_by_case = labels.groupby("case_id")[gene_cols].max().reset_index()  # any sample mutated -> case positive
    merged = coords.merge(labels_by_case, on="case_id", how="inner")
    merged.attrs["gene_cols"] = gene_cols
    return merged


class PatchPairDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, gene_cols: list[str], augment: bool = False):
        """augment: random horizontal/vertical flips, applied IDENTICALLY to
        the local and global patch of a sample (not independently -- they
        must stay spatially consistent, since the MMA module assumes a
        shared center point and orientation between the two magnifications).

        This directly addresses a confirmed gap vs. both source papers: the
        official (if broken) DeepPATHO repo's DataGenetors.py shows they
        used `horizontal_flip=True`; our first two training attempts had
        zero augmentation and overfit immediately (train loss dropping
        smoothly while val loss diverged from epoch 0 on), consistent with
        a model that could just memorize exact patches.
        """
        if openslide is None:
            raise RuntimeError("openslide-python is required to read patches from .svs files")
        self.df = index_df.reset_index(drop=True)
        self.gene_cols = gene_cols
        self.augment = augment
        self._slide_cache: dict[str, "openslide.OpenSlide"] = {}
        self._geom_cache: dict[str, SlideGeometry] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _get_slide(self, slide_path: str):
        if slide_path not in self._slide_cache:
            slide = openslide.OpenSlide(slide_path)
            self._slide_cache[slide_path] = slide
            self._geom_cache[slide_path] = SlideGeometry.from_slide(slide)
        return self._slide_cache[slide_path], self._geom_cache[slide_path]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        slide, geom = self._get_slide(row["slide_path"])
        cx, cy = int(row["center_x"]), int(row["center_y"])

        local_patch = read_patch(slide, cx, cy, geom.local_downsample)
        global_patch = read_patch(slide, cx, cy, geom.global_downsample)

        local_t = torch.from_numpy(local_patch).permute(2, 0, 1).float() / 255.0
        global_t = torch.from_numpy(global_patch).permute(2, 0, 1).float() / 255.0

        if self.augment:
            if np.random.rand() < 0.5:
                local_t = local_t.flip(-1)
                global_t = global_t.flip(-1)
            if np.random.rand() < 0.5:
                local_t = local_t.flip(-2)
                global_t = global_t.flip(-2)

        label = torch.tensor(row[self.gene_cols].values.astype(np.float32))
        meta = {"file_id": row["file_id"], "case_id": row["case_id"]}
        return local_t, global_t, label, meta

    def close(self):
        for s in self._slide_cache.values():
            s.close()
        self._slide_cache.clear()
