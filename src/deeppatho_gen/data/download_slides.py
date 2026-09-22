"""Parallel, resumable downloader for GDC diagnostic slides.

TCGA diagnostic H&E slides are open-access (no token needed). This hits
GDC's public /data/{file_id} endpoint directly with a thread pool, since a
single connection is bandwidth-capped well below this network's actual
throughput (observed ~2-3MB/s per connection during the initial pilot
download; parallelizing is a straightforward multi-x speedup here, not a
premature optimization).

Resumable: skips any file already on disk with the correct size (from the
manifest's file_size column), so a killed/disconnected run can just be
re-launched with the same command.

Usage:
    python -m deeppatho_gen.data.download_slides \
        --manifest data/raw/gdc_manifest.csv \
        --out-dir data/raw/svs \
        --workers 8
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{file_id}"


def download_one(file_id: str, file_name: str, expected_size: int, out_dir: Path, max_retries: int = 3) -> tuple[str, str]:
    dest_dir = out_dir / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name

    if dest.exists() and dest.stat().st_size == expected_size:
        return file_id, "skipped (already complete)"

    url = GDC_DATA_URL.format(file_id=file_id)
    last_err = None
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
            if dest.stat().st_size != expected_size:
                raise IOError(f"size mismatch: got {dest.stat().st_size}, expected {expected_size}")
            return file_id, "downloaded"
        except Exception as e:  # noqa: BLE001 -- retry loop, want to catch broadly here
            last_err = e
            time.sleep(2 * (attempt + 1))
    return file_id, f"FAILED after {max_retries} attempts: {last_err}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/raw/gdc_manifest.csv")
    ap.add_argument("--out-dir", default="data/raw/svs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)

    total_bytes = manifest["file_size"].sum()
    print(f"{len(manifest)} slides, {total_bytes / 1e9:.1f} GB total, {args.workers} parallel workers")

    t0 = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row["file_id"], row["file_name"], row["file_size"], out_dir): row["file_id"]
            for _, row in manifest.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            file_id, status = fut.result()
            if status.startswith("FAILED"):
                n_fail += 1
                print(f"[{i}/{len(futures)}] {file_id}: {status}")
            elif status.startswith("skipped"):
                n_skip += 1
            else:
                n_ok += 1
            if i % 20 == 0 or i == len(futures):
                elapsed = time.time() - t0
                print(f"[{i}/{len(futures)}] ok={n_ok} skipped={n_skip} failed={n_fail}  elapsed={elapsed/60:.1f}min")

    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail}  total_time={(time.time()-t0)/60:.1f}min")
    if n_fail:
        print("Re-run the same command to retry failed downloads (skip logic will leave completed ones alone).")


if __name__ == "__main__":
    main()
