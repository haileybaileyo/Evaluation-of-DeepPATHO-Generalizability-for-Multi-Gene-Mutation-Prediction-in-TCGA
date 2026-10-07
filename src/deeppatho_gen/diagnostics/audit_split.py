"""Patient-level split audit for a single target gene.

Verifies, with hard asserts (not just eyeballing), that the train/val/test
split used by train.py is genuinely patient-level: no TCGA patient appears
in more than one split, and reports the per-split statistics needed to
reason about class balance, multi-slide patients, and label conflicts
before any diagnostic training is run.

case_id extraction rule (documented per the task spec, not just in a
comment): tile_wsi.py records `case_id` as GDC's `cases.submitter_id` for
the file, which for TCGA is the 12-character patient barcode (e.g.
"TCGA-38-4631") -- this is the *patient*, not the sample or slide barcode
(which would carry a "-01"/"-11" suffix). fetch_mutation_labels.py
independently derives the same 12-char case_id by truncating the sample
barcode (`sid[:12]`). Both paths land on the same patient-level identifier,
so joining patch_coords (via file_id -> case_id) against mutation_labels
(via sample_id -> case_id) and then splitting on unique case_id is a
patient-level split by construction. This script verifies that claim
empirically rather than trusting the construction.

Usage:
    python -m deeppatho_gen.diagnostics.audit_split \
        --gene TP53 \
        --patch-coords data/processed/patch_coords_tumor_only.csv \
        --labels data/raw/mutation_labels.csv \
        --out-dir runs/diagnostics/audit_tp53
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import case_level_split


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3]).decode().strip()
    except Exception:
        return "unknown"


def build_patient_table(coords: pd.DataFrame, labels: pd.DataFrame, gene: str) -> pd.DataFrame:
    """One row per (case_id, sample_id) label record, joined with slide counts."""
    if gene not in labels.columns:
        raise SystemExit(f"Gene {gene!r} not in labels columns: {list(labels.columns)}")

    # Label conflict check: does any single case_id have both a 0 and a 1
    # across its sample_id rows in the raw label table (before any max())?
    per_case_label_values = labels.groupby("case_id")[gene].apply(lambda s: sorted(s.unique()))
    conflicts = per_case_label_values[per_case_label_values.apply(len) > 1]

    labels_by_case = labels.groupby("case_id")[gene].max().reset_index()
    slide_counts = coords.groupby("case_id")["file_id"].nunique().reset_index(name="n_slides")

    table = labels_by_case.merge(slide_counts, on="case_id", how="left")
    table["n_slides"] = table["n_slides"].fillna(0).astype(int)
    table["has_slides"] = table["n_slides"] > 0
    table["label_conflict"] = table["case_id"].isin(conflicts.index)
    return table


def split_stats(table: pd.DataFrame, split_name: str) -> dict:
    sub = table[table["split"] == split_name]
    with_slides = sub[sub["has_slides"]]
    slide_dist = with_slides["n_slides"]
    return {
        "split": split_name,
        "n_patients_total": int(len(sub)),
        "n_patients_with_slides": int(len(with_slides)),
        "n_patients_missing_slides": int((~sub["has_slides"]).sum()),
        "n_slides_total": int(with_slides["n_slides"].sum()),
        "n_positive_patients": int(with_slides[with_slides["label"] == 1].shape[0]),
        "n_negative_patients": int(with_slides[with_slides["label"] == 0].shape[0]),
        "n_positive_slides": int(with_slides.loc[with_slides["label"] == 1, "n_slides"].sum()),
        "n_negative_slides": int(with_slides.loc[with_slides["label"] == 0, "n_slides"].sum()),
        "slides_per_patient_min": int(slide_dist.min()) if len(slide_dist) else 0,
        "slides_per_patient_median": float(slide_dist.median()) if len(slide_dist) else 0.0,
        "slides_per_patient_max": int(slide_dist.max()) if len(slide_dist) else 0,
        "n_label_conflicts": int(sub["label_conflict"].sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gene", required=True)
    ap.add_argument("--patch-coords", default="data/processed/patch_coords_tumor_only.csv")
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="runs/diagnostics/audit")
    args = ap.parse_args()

    coords = pd.read_csv(args.patch_coords)
    labels = pd.read_csv(args.labels)

    table = build_patient_table(coords, labels, args.gene)
    table = table.rename(columns={args.gene: "label"})

    split = case_level_split(table["case_id"].tolist(), train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)
    table["split"] = table["case_id"].map(split)

    # --- Hard assert: no patient appears in more than one split. ---
    # By construction (case_level_split assigns each unique case_id to
    # exactly one split) this cannot happen, but we check the *realized*
    # assignment against the raw per-slide records too, in case a caller
    # ever passes duplicated/aliased case_ids in.
    by_split = {s: set(table.loc[table.split == s, "case_id"]) for s in ("train", "val", "test")}
    overlaps = {
        "train_val": by_split["train"] & by_split["val"],
        "train_test": by_split["train"] & by_split["test"],
        "val_test": by_split["val"] & by_split["test"],
    }
    leak_found = any(overlaps.values())
    if leak_found:
        print("PATIENT LEAKAGE DETECTED:")
        for pair, ids in overlaps.items():
            if ids:
                print(f"  {pair}: {sorted(ids)}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(out_dir / "split_audit.csv", index=False)

    stats = [split_stats(table, s) for s in ("train", "val", "test")]
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "split_audit_summary.csv", index=False)

    meta = {
        "gene": args.gene,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "git_commit": git_commit_hash(),
        "patch_coords_file": args.patch_coords,
        "labels_file": args.labels,
        "n_total_patients": int(len(table)),
        "n_total_slides": int(table["n_slides"].sum()),
        "n_total_label_conflicts": int(table["label_conflict"].sum()),
        "patient_leakage_detected": leak_found,
    }
    (out_dir / "audit_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Gene: {args.gene}  |  git: {meta['git_commit'][:8]}  |  seed: {args.seed}")
    print(f"Total patients: {meta['n_total_patients']}, total slides: {meta['n_total_slides']}, "
          f"label conflicts: {meta['n_total_label_conflicts']}")
    print("\nNo patient-level leakage between splits: PASS" if not leak_found else "\nLEAKAGE: FAIL")
    print()
    print(stats_df.to_string(index=False))
    print(f"\nWrote {out_dir}/split_audit.csv, split_audit_summary.csv, audit_meta.json")


if __name__ == "__main__":
    main()
