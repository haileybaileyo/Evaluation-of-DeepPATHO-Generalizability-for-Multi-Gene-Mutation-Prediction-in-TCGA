"""Build a coordinate index of matched (5x, 20x) patch pairs from WSIs.

Phase 1, Step 3 -- the first step that actually needs the downloaded slide
files (.svs) and a GPU-less machine is fine for this too; it's CPU/IO bound.

Design choice: we do NOT rasterize and save ~1-2M individual patch PNGs to
disk (that's ~100s of GB and mirrors of I/O overhead). Instead we record
*coordinates* (slide path, level-0 x, y, tissue fraction) in a CSV, and the
PyTorch Dataset (see dataset.py) re-reads the exact 5x/20x patch pair from
the open .svs with openslide at train time. This matches the tiling logic
described in Coudray et al. 2018 (non-overlapping 256x256, <=50% background
discarded) and Kosaraju et al. 2025 (>=20% tissue kept; global and local
patches share the same center point).

Speed: most of a WSI's area is white background, and reading a full-res
256x256 region just to check its background fraction is expensive at scale
(hundreds of thousands of reads per slide-set). So we first build one cheap
low-resolution tissue mask per slide (a single get_thumbnail() call) and use
it to skip full-resolution reads for candidate tiles that are clearly
background, only paying the expensive read_region() for tiles the thumbnail
says probably contain tissue. Slides are also processed in parallel across
a process pool (each worker opens its own openslide handle).

Usage:
    python -m deeppatho_gen.preprocessing.tile_wsi \
        --manifest data/raw/gdc_manifest.csv \
        --slide-dir data/raw/svs \
        --out data/processed/patch_coords.csv \
        --min-tissue-fraction 0.20 \
        --workers 8

Requires: openslide-python + the system libopenslide (`apt install
openslide-tools` / `brew install openslide`).
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import openslide
except ImportError:  # pragma: no cover - only hit on machines without the system lib
    openslide = None

PATCH_PX = 256
LOCAL_MAG = 20.0
GLOBAL_MAG = 5.0
THUMBNAIL_MAX_SIDE = 2048


@dataclass
class SlideGeometry:
    """Pixel bookkeeping for one slide at the two target magnifications."""

    base_mag: float
    local_downsample: float   # level-0 px per output px at 20x
    global_downsample: float  # level-0 px per output px at 5x

    @classmethod
    def from_slide(cls, slide: "openslide.OpenSlide") -> "SlideGeometry":
        base_mag = float(slide.properties.get("openslide.objective-power", 40))
        return cls(
            base_mag=base_mag,
            local_downsample=base_mag / LOCAL_MAG,
            global_downsample=base_mag / GLOBAL_MAG,
        )

    def local_step_level0(self) -> int:
        """Level-0 pixel stride between adjacent non-overlapping 20x tiles."""
        return round(PATCH_PX * self.local_downsample)

    def global_extent_level0(self) -> int:
        """Level-0 pixel width/height covered by one 5x patch."""
        return round(PATCH_PX * self.global_downsample)


def tissue_fraction(rgb: np.ndarray, bg_threshold: int = 220) -> float:
    """Fraction of non-background pixels, following Coudray et al.'s rule
    (a pixel counts as background if all RGB channels are >= bg_threshold,
    i.e. near-white slide background)."""
    is_bg = np.all(rgb >= bg_threshold, axis=-1)
    return 1.0 - is_bg.mean()


def read_patch(slide: "openslide.OpenSlide", cx: int, cy: int, downsample: float) -> np.ndarray:
    """Read a PATCH_PX x PATCH_PX RGB patch centered at (cx, cy) in level-0
    coordinates, at the given downsample factor, using the best available
    pyramid level and resizing to the exact output size."""
    level = slide.get_best_level_for_downsample(downsample)
    level_downsample = slide.level_downsamples[level]
    extent_level0 = round(PATCH_PX * downsample)
    top_left_level0 = (cx - extent_level0 // 2, cy - extent_level0 // 2)
    read_size = round(extent_level0 / level_downsample)
    region = slide.read_region(top_left_level0, level, (read_size, read_size)).convert("RGB")
    if read_size != PATCH_PX:
        region = region.resize((PATCH_PX, PATCH_PX))
    return np.array(region)  # copy (not asarray): PIL's buffer isn't writable, but downstream
    # code (torch.from_numpy in dataset.py) needs a writable array


class ThumbnailTissueMask:
    """Cheap low-res tissue/background mask, used to skip expensive
    full-resolution reads for tiles that are obviously background."""

    def __init__(self, slide: "openslide.OpenSlide", bg_threshold: int = 220):
        w, h = slide.dimensions
        scale = THUMBNAIL_MAX_SIDE / max(w, h)
        thumb_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        thumb = np.array(slide.get_thumbnail(thumb_size).convert("RGB"))
        self.is_tissue = ~np.all(thumb >= bg_threshold, axis=-1)
        self.scale_x = w / thumb.shape[1]
        self.scale_y = h / thumb.shape[0]

    def coarse_tissue_fraction(self, cx: int, cy: int, extent_level0: int) -> float:
        x0 = max(0, int((cx - extent_level0 / 2) / self.scale_x))
        x1 = min(self.is_tissue.shape[1], int((cx + extent_level0 / 2) / self.scale_x))
        y0 = max(0, int((cy - extent_level0 / 2) / self.scale_y))
        y1 = min(self.is_tissue.shape[0], int((cy + extent_level0 / 2) / self.scale_y))
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return float(self.is_tissue[y0:y1, x0:x1].mean())


def enumerate_slide_patches(
    slide_path: Path, min_tissue_fraction: float, max_patches: int | None = None, seed: int = 0
) -> list[dict]:
    slide = openslide.OpenSlide(str(slide_path))
    geom = SlideGeometry.from_slide(slide)
    width, height = slide.dimensions

    step = geom.local_step_level0()
    global_extent = geom.global_extent_level0()
    margin = global_extent // 2  # keep the (larger) global patch on-slide

    mask = ThumbnailTissueMask(slide)
    # Coarse pre-filter threshold is deliberately looser than the real
    # threshold (thumbnail is low-res and can under/over-estimate tissue at
    # tile boundaries) -- it only needs to rule out tiles that are
    # confidently all-background, not make the final accept/reject call.
    coarse_threshold = max(0.0, min_tissue_fraction - 0.15)

    records = []
    for cy in range(margin, height - margin, step):
        for cx in range(margin, width - margin, step):
            if mask.coarse_tissue_fraction(cx, cy, step) < coarse_threshold:
                continue
            local_patch = read_patch(slide, cx, cy, geom.local_downsample)
            frac = tissue_fraction(local_patch)
            if frac < min_tissue_fraction:
                continue
            records.append(
                {
                    "slide_path": str(slide_path),
                    "center_x": cx,
                    "center_y": cy,
                    "tissue_fraction": round(float(frac), 4),
                }
            )
    slide.close()

    if max_patches is not None and len(records) > max_patches:
        # Cap per-slide patch count so no single (usually larger) slide
        # dominates an epoch, and so total dataset size / epoch time stays
        # tractable at full (534-slide) scale. Coudray et al.'s TCGA-LUAD
        # tiling averaged ~600 tiles/slide across the whole cohort (987,931
        # tiles / 1634 slides); we cap similarly rather than keep every tile
        # from slides with thousands of tissue-covered tiles.
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(records), size=max_patches, replace=False)
        records = [records[i] for i in sorted(keep_idx)]

    return records


def _tile_one(args) -> tuple[str, str, list[dict], str | None]:
    file_id, case_id, slide_path, min_tissue_fraction, max_patches = args
    try:
        recs = enumerate_slide_patches(Path(slide_path), min_tissue_fraction, max_patches)
        for r in recs:
            r["file_id"] = file_id
            r["case_id"] = case_id
        return file_id, case_id, recs, None
    except Exception as e:  # noqa: BLE001 -- report and move on, don't kill the whole pool
        return file_id, case_id, [], str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/raw/gdc_manifest.csv")
    ap.add_argument("--slide-dir", default="data/raw/svs",
                     help="directory containing downloaded .svs files named <file_id>/<file_name>, "
                          "as produced by gdc-client")
    ap.add_argument("--out", default="data/processed/patch_coords.csv")
    ap.add_argument("--min-tissue-fraction", type=float, default=0.20)
    ap.add_argument("--max-patches-per-slide", type=int, default=None,
                     help="randomly subsample down to this many tissue patches per slide "
                          "(keeps epoch time bounded at full-cohort scale)")
    ap.add_argument("--resume", action="store_true",
                     help="skip slides whose file_id is already in --out (append new slides' "
                          "patches instead of recomputing everything from scratch)")
    ap.add_argument("--workers", type=int, default=8, help="slides tiled in parallel")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                     help="stop submitting new slides once this much wall-clock time has "
                          "elapsed (in-flight slides still finish); guarantees the stage "
                          "terminates within a deadline instead of processing all slides")
    args = ap.parse_args()

    if openslide is None:
        raise SystemExit(
            "openslide-python / libopenslide is not installed on this machine.\n"
            "Install with: pip install openslide-python  (+ system libopenslide)\n"
            "This step must run wherever the .svs files actually live."
        )

    manifest = pd.read_csv(args.manifest)
    slide_dir = Path(args.slide_dir)
    out_path = Path(args.out)

    already_done: set[str] = set()
    existing_df = None
    if args.resume and out_path.exists():
        existing_df = pd.read_csv(out_path)
        already_done = set(existing_df["file_id"].unique())
        print(f"Resuming: {len(already_done)} slides already tiled in {out_path}", flush=True)

    jobs = []
    n_skipped_missing = 0
    for _, row in manifest.iterrows():
        if row["file_id"] in already_done:
            continue
        slide_path = slide_dir / row["file_id"] / row["file_name"]
        if not slide_path.exists():
            n_skipped_missing += 1
            continue
        jobs.append((row["file_id"], row["case_id"], str(slide_path), args.min_tissue_fraction, args.max_patches_per_slide))

    print(f"{len(jobs)} slides to tile ({n_skipped_missing} not yet downloaded, "
          f"{len(already_done)} already done), {args.workers} parallel workers", flush=True)

    t0 = time.time()
    deadline = t0 + args.time_budget_hours * 3600 if args.time_budget_hours else None
    n_done, n_failed, n_timed_out = 0, 0, 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        job_iter = iter(jobs)

        def submit_more():
            nonlocal n_timed_out
            while len(futures) < args.workers * 2:
                if deadline and time.time() > deadline:
                    n_timed_out = len(jobs) - n_done - n_failed - len(futures)
                    return False
                try:
                    job = next(job_iter)
                except StopIteration:
                    return False
                futures[pool.submit(_tile_one, job)] = job[0]
            return True

        submit_more()
        pbar = tqdm(total=len(jobs), desc="slides")
        while futures:
            done_future = next(as_completed(futures))
            file_id, case_id, recs, err = done_future.result()
            del futures[done_future]

            if err:
                n_failed += 1
                tqdm.write(f"FAILED {file_id}: {err}")
            else:
                n_done += 1
                if recs:
                    partial_df = pd.DataFrame(recs)
                    existing_df = pd.concat([existing_df, partial_df]) if existing_df is not None else partial_df
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    existing_df.to_csv(out_path, index=False)
            pbar.update(1)

            if not (deadline and time.time() > deadline):
                submit_more()
        pbar.close()

    final_n = len(existing_df) if existing_df is not None else 0
    final_slides = existing_df["file_id"].nunique() if existing_df is not None else 0
    elapsed_min = (time.time() - t0) / 60
    print(f"\nWrote {final_n} patch coordinates from {final_slides} slides to {out_path} in {elapsed_min:.1f}min", flush=True)
    print(f"Done={n_done} Failed={n_failed} SkippedByDeadline={n_timed_out} NotYetDownloaded={n_skipped_missing}", flush=True)


if __name__ == "__main__":
    main()
