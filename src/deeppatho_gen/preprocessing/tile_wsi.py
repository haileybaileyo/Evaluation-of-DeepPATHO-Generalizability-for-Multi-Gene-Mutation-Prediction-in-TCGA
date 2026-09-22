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

Usage:
    python -m deeppatho_gen.preprocessing.tile_wsi \
        --manifest data/raw/gdc_manifest.csv \
        --slide-dir data/raw/svs \
        --out data/processed/patch_coords.csv \
        --min-tissue-fraction 0.20

Requires: openslide-python + the system libopenslide (`apt install
openslide-tools` / `brew install openslide`).
"""
from __future__ import annotations

import argparse
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


def enumerate_slide_patches(slide_path: Path, min_tissue_fraction: float) -> list[dict]:
    slide = openslide.OpenSlide(str(slide_path))
    geom = SlideGeometry.from_slide(slide)
    width, height = slide.dimensions

    step = geom.local_step_level0()
    global_extent = geom.global_extent_level0()
    # Keep enough margin that the (larger) global patch never runs off the slide.
    margin = global_extent // 2

    records = []
    for cy in range(margin, height - margin, step):
        for cx in range(margin, width - margin, step):
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
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/raw/gdc_manifest.csv")
    ap.add_argument("--slide-dir", default="data/raw/svs",
                     help="directory containing downloaded .svs files named <file_id>/<file_name>, "
                          "as produced by gdc-client")
    ap.add_argument("--out", default="data/processed/patch_coords.csv")
    ap.add_argument("--min-tissue-fraction", type=float, default=0.20)
    args = ap.parse_args()

    if openslide is None:
        raise SystemExit(
            "openslide-python / libopenslide is not installed on this machine.\n"
            "Install with: pip install openslide-python  (+ system libopenslide)\n"
            "This step must run wherever the .svs files actually live."
        )

    manifest = pd.read_csv(args.manifest)
    slide_dir = Path(args.slide_dir)

    all_records = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="slides"):
        slide_path = slide_dir / row["file_id"] / row["file_name"]
        if not slide_path.exists():
            tqdm.write(f"MISSING (not downloaded yet): {slide_path}")
            continue
        recs = enumerate_slide_patches(slide_path, args.min_tissue_fraction)
        for r in recs:
            r["file_id"] = row["file_id"]
            r["case_id"] = row["case_id"]
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} patch coordinates from {df['file_id'].nunique() if len(df) else 0} slides to {out_path}")


if __name__ == "__main__":
    main()
