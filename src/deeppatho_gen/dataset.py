"""PyTorch Dataset: reads matched (20x, 5x) patch pairs on the fly from the
open .svs files, using the coordinate index from preprocessing/tile_wsi.py
joined against the multi-gene label table from data/fetch_mutation_labels.py.

Only tiles from slides whose case survives the case-level train/val/test
split are ever loaded by a given split's Dataset, so no patient's tiles
leak across sets (Coudray et al.'s "never trained and tested on tiles from
the same tumor sample" requirement, generalized to case-level).
"""
from __future__ import annotations

from collections import OrderedDict
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
    """Deterministic case_id -> {'train','val','test'} assignment.

    NOT stratified: a pure random shuffle of case_ids. For an imbalanced
    binary label (e.g. EGFR at ~12% positive), this can and did produce
    wildly different positive ratios per split by chance (observed 13%/15%/6%
    train/val/test on one run). Kept only for callers that split on
    something other than a single binary label (e.g. train.py's multi-gene
    setup, where no single stratification target is well-defined). For a
    single binary target, use stratified_case_level_split instead.
    """
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


def stratified_case_level_split(
    case_ids: list[str],
    case_labels: dict[str, int],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Deterministic case_id -> {'train','val','test'}, stratified on a
    per-case binary label so each split preserves (as closely as integer
    case counts allow) the overall positive ratio.

    This is patient/case-level, not patch-level: the unit being split and
    stratified is one row per case_id, so no single patient's patches can
    ever appear in two splits, and the positive-case *fraction* (not just
    the count) is balanced across splits -- the two conditions required to
    avoid the leakage and the split-imbalance problems that plain
    case_level_split can produce on a rare, imbalanced label like EGFR.
    """
    from sklearn.model_selection import train_test_split

    ids = np.array(sorted(set(case_ids)))
    y = np.array([case_labels[cid] for cid in ids])

    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 1e-9, "train_frac + val_frac must be < 1.0"

    ids_trainval, ids_test, y_trainval, _ = train_test_split(
        ids, y, test_size=test_frac, random_state=seed, stratify=y
    )
    rel_val_frac = val_frac / (train_frac + val_frac)
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=rel_val_frac, random_state=seed, stratify=y_trainval
    )

    assignment = {cid: "train" for cid in ids_train}
    assignment.update({cid: "val" for cid in ids_val})
    assignment.update({cid: "test" for cid in ids_test})
    return assignment


def split_summary(index_df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Per-split patient count and positive ratio, computed at the case
    level (one row per patient, not per patch) -- what section 4/6 of the
    experiment plan requires printing after every split."""
    rows = []
    for split_name, sub in index_df.groupby("split"):
        case_labels = sub.groupby("case_id")[label_col].max()
        n_pos = int(case_labels.sum())
        n_pat = len(case_labels)
        rows.append({
            "split": split_name,
            "n_patients": n_pat,
            "n_positive": n_pos,
            "n_negative": n_pat - n_pos,
            "positive_ratio": round(n_pos / n_pat, 4) if n_pat else float("nan"),
            "n_patches": len(sub),
        })
    return pd.DataFrame(rows).set_index("split").loc[["train", "val", "test"]]


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
        # LRU-bounded, NOT an unbounded cache: this dataset's index typically
        # spans hundreds of distinct slides (532 tumor + 244 normal for the
        # tumor classifier), and each worker process (persistent_workers,
        # shuffled) ends up lazily touching most of them over an epoch. An
        # unbounded cache of OpenSlide handles leaked memory until the OOM
        # killer took out DataLoader workers mid-run -- confirmed via
        # `dmesg`-style "DataLoader worker ... killed by signal: Killed" and
        # `free` showing 122/125GB used. Bounding this to a small number of
        # concurrently-open slides per worker fixes it; re-opening a slide
        # that was evicted is cheap relative to actually reading a region.
        self._slide_cache: "OrderedDict[str, openslide.OpenSlide]" = OrderedDict()
        self._geom_cache: dict[str, SlideGeometry] = {}
        self._max_open_slides = 16

    def __len__(self) -> int:
        return len(self.df)

    def _get_slide(self, slide_path: str):
        if slide_path in self._slide_cache:
            self._slide_cache.move_to_end(slide_path)  # mark as most-recently-used
            return self._slide_cache[slide_path], self._geom_cache[slide_path]

        slide = openslide.OpenSlide(slide_path)
        self._slide_cache[slide_path] = slide
        self._geom_cache[slide_path] = SlideGeometry.from_slide(slide)

        if len(self._slide_cache) > self._max_open_slides:
            oldest_path, oldest_slide = self._slide_cache.popitem(last=False)
            oldest_slide.close()
            del self._geom_cache[oldest_path]

        return slide, self._geom_cache[slide_path]

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
