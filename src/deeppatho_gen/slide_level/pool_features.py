"""Pool each slide's UNI-v2 patch features (TRIDENT .h5 files) into one vector
per slide and cache them in a single small .npz.

Why: the TRIDENT feature files already contain every tissue patch's frozen
UNI-v2 embedding (N_patches x 1536) for the TCGA-LUAD DX slides. Pooling them
once means slide-/patient-level experiments (logistic regression, repeated
cross-validation, positive/negative controls) run in seconds instead of the
~3.5h the patch-level runs need to re-read .svs files every epoch.

Read-only on the feature files. Output: mean-pooled and max-pooled vectors,
patch counts, slide ids and case ids.

Usage:
    python -m deeppatho_gen.slide_level.pool_features \
        --features-dir ../uncertainty-aware-histopathology-survival-analysis/data/processed/uni_v2/TCGA_LUAD \
        --out data/processed/uni_v2_luad_slide_pooled.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def case_id_from_name(name: str) -> str:
    """'TCGA-05-4244-01Z-00-DX1.<uuid>.h5' -> 'TCGA-05-4244'."""
    return "-".join(name.split("-")[:3])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", default="data/processed/uni_v2_luad_slide_pooled.npz")
    args = ap.parse_args()

    files = sorted(Path(args.features_dir).glob("*.h5"))
    print(f"{len(files)} feature files in {args.features_dir}", flush=True)

    slide_ids, case_ids, n_patches, means, maxs, failed = [], [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            with h5py.File(f, "r") as h:
                x = h["features"][:]
            if x.ndim != 2 or len(x) == 0:
                raise ValueError(f"unexpected feature shape {x.shape}")
            means.append(x.mean(axis=0, dtype=np.float64).astype(np.float32))
            maxs.append(x.max(axis=0).astype(np.float32))
            n_patches.append(len(x))
            slide_ids.append(f.name[: -len(".h5")])
            case_ids.append(case_id_from_name(f.name))
        except Exception as e:  # noqa: BLE001 -- report and continue
            failed.append((f.name, str(e)))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mean=np.stack(means), max=np.stack(maxs), n_patches=np.array(n_patches),
        slide_ids=np.array(slide_ids), case_ids=np.array(case_ids),
    )
    print(f"Wrote {len(slide_ids)} slides ({len(set(case_ids))} patients) to {out} in {time.time() - t0:.0f}s")
    if failed:
        print(f"FAILED {len(failed)} files:")
        for name, err in failed:
            print(" ", name, err)


if __name__ == "__main__":
    main()
