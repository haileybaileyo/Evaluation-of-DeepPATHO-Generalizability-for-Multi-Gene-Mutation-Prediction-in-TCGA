"""DeepPATH-style tiling: non-overlapping 512x512 tiles at 20x, saved as JPEG.

Follows the tiling of Coudray et al. 2018 (DeepPATH/0b_tileLoop_deepzoom4.py,
`-s 512 -M 20 -B 50`):
  * 512x512 px tiles, non-overlapping, at 20x (0.5 um/px)
  * a pixel is background if its grayscale value (PIL 'L') is >= 220; a tile is
    kept only if at most 50% of its pixels are background
  * tiles are written to disk (unlike the repo's other patch pipeline, which
    only stores coordinates) so that a DeepPATH-style classifier can train on
    small JPEGs instead of re-reading .svs files every epoch

Deliberate deviation (documented): the base magnification of each slide is
taken from its pixel size (mpp) when present, because 23/534 TCGA-LUAD DX
slides carry objective-power=20 with mpp~0.23 (i.e. actually 40x resolution),
and 9 carry no magnification metadata at all. Order of preference:
  mpp  ->  downsample = 0.5 / mpp
  objective-power -> downsample = objective_power / 20
  neither -> assume 40x (flagged in the index as mag_source="assumed_40x")

Output: <out-dir>/<file_id>/<x>_<y>.jpg and, once all slides are done,
<out-dir>/tile_index.csv with columns
  file_id, case_id, x, y (level-0 top-left), downsample, bg_frac, mag_source, path

Resumable: a slide whose folder contains `_done` is skipped.

Usage:
    python -m deeppatho_gen.deeppath_style.tile_512 --out-dir data/processed/deeppath_tiles/dx_20x_512
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import openslide
except ImportError:  # pragma: no cover
    openslide = None

TILE = 512
TARGET_MPP = 0.5  # 20x
BG_GRAY = 220
THUMB_MAX = 2048


def slide_downsample(slide) -> tuple[float, str]:
    pr = slide.properties
    for key in ("openslide.mpp-x", "aperio.MPP"):
        try:
            mpp = float(pr[key])
            if mpp > 0:
                return TARGET_MPP / mpp, "mpp"
        except (KeyError, ValueError):
            pass
    try:
        return float(pr["openslide.objective-power"]) / 20.0, "objective_power"
    except (KeyError, ValueError):
        return 2.0, "assumed_40x"


def bg_fraction(img: Image.Image) -> float:
    return float((np.asarray(img.convert("L")) >= BG_GRAY).mean())


def tile_one(args) -> tuple[str, int, str | None]:
    file_id, case_id, slide_path, out_dir, max_tiles, quality, seed = args
    d = Path(out_dir) / file_id
    if (d / "_done").exists():
        n = len(list(d.glob("*.jpg")))
        return file_id, n, None
    try:
        d.mkdir(parents=True, exist_ok=True)
        s = openslide.OpenSlide(slide_path)
        ds, src = slide_downsample(s)
        ext = int(round(TILE * ds))  # tile edge in level-0 pixels
        level = s.get_best_level_for_downsample(ds)
        lvl_ds = s.level_downsamples[level]
        read_px = int(round(ext / lvl_ds))
        W, H = s.dimensions

        # cheap thumbnail pre-filter: skip cells that are clearly all background
        scale = THUMB_MAX / max(W, H)
        th = s.get_thumbnail((max(1, round(W * scale)), max(1, round(H * scale)))).convert("L")
        bg = np.asarray(th) >= BG_GRAY
        sx, sy = bg.shape[1] / W, bg.shape[0] / H

        cells = []
        for y in range(0, H - ext + 1, ext):
            for x in range(0, W - ext + 1, ext):
                x0, x1 = int(x * sx), max(int((x + ext) * sx), int(x * sx) + 1)
                y0, y1 = int(y * sy), max(int((y + ext) * sy), int(y * sy) + 1)
                if bg[y0:y1, x0:x1].mean() > 0.9:  # looser than the real 0.5 rule
                    continue
                cells.append((x, y))
        if max_tiles and len(cells) > max_tiles:
            rng = np.random.default_rng(seed)
            cells = [cells[i] for i in sorted(rng.choice(len(cells), max_tiles, replace=False))]

        rows = []
        for x, y in cells:
            img = s.read_region((x, y), level, (read_px, read_px)).convert("RGB")
            if img.size != (TILE, TILE):
                img = img.resize((TILE, TILE), Image.LANCZOS)
            frac = bg_fraction(img)
            if frac > 0.5:  # DeepPATH "-B 50"
                continue
            fn = f"{x}_{y}.jpg"
            img.save(d / fn, quality=quality)
            rows.append((file_id, case_id, x, y, round(ds, 4), round(frac, 4), src, f"{file_id}/{fn}"))
        s.close()
        pd.DataFrame(rows, columns=["file_id", "case_id", "x", "y", "downsample", "bg_frac", "mag_source", "path"]).to_csv(
            d / "index.csv", index=False)
        (d / "_done").write_text("ok")
        return file_id, len(rows), None
    except Exception as e:  # noqa: BLE001
        return file_id, 0, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/raw/gdc_manifest.csv")
    ap.add_argument("--slide-dir", default="data/raw/svs")
    ap.add_argument("--out-dir", default="data/processed/deeppath_tiles/dx_20x_512")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--max-tiles-per-slide", type=int, default=0, help="0 = keep every tile (DeepPATH behaviour)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N slides (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    if args.limit:
        man = man.head(args.limit)
    jobs = [(r.file_id, r.case_id, str(Path(args.slide_dir) / r.file_id / r.file_name), args.out_dir,
             args.max_tiles_per_slide, args.quality, args.seed) for r in man.itertuples()]
    print(f"{len(jobs)} slides, {args.workers} workers -> {args.out_dir}", flush=True)
    t0, done, fail, total = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(tile_one, j) for j in jobs]
        for f in as_completed(futs):
            fid, n, err = f.result()
            done += 1
            total += n
            if err:
                fail += 1
                print(f"FAILED {fid}: {err}", flush=True)
            if done % 20 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] tiles so far {total:,}  failed {fail}  {(time.time() - t0) / 60:.1f} min", flush=True)

    parts = [pd.read_csv(p) for p in sorted(Path(args.out_dir).glob("*/index.csv")) if p.parent.name in set(man.file_id)]
    if parts:
        idx = pd.concat(parts, ignore_index=True)
        idx.to_csv(Path(args.out_dir) / "tile_index.csv", index=False)
        print(f"\nWrote tile_index.csv: {len(idx):,} tiles from {idx.file_id.nunique()} slides "
              f"({idx.case_id.nunique()} patients); mag_source counts: {idx.groupby('file_id').mag_source.first().value_counts().to_dict()}")


if __name__ == "__main__":
    main()
