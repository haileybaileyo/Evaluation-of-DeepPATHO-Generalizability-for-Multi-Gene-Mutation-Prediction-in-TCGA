"""Verify, with hard asserts, the patient-level split that every EGFR experiment uses.

It rebuilds the split with the SAME call the training scripts make
(get_fixed_split -> dataset.stratified_case_level_split: 70/15/15, seed 42,
stratified on the EGFR label) and checks:

  1. split sizes: patients / slides / EGFR-positive patients per split
  2. no patient is in two splits
  3. no slide is in two splits
  4. no slide belongs to two patients
  5. every patient with several slides has ALL slides in ONE split
  6. (optional) the rebuilt test slides are exactly the test slides stored by an
     earlier training run (proof that this reconstruction is the split that was
     really used)

Usage:
    python -m deeppatho_gen.diagnostics.check_egfr_split_overlap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..slide_level.fixed_split_logreg import get_fixed_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coords", default="data/processed/patch_coords_tumor_only.csv")
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv")
    ap.add_argument("--saved-test-slides", default="runs/egfr_uni/frozen/test_slide_scores.csv")
    args = ap.parse_args()

    split, case_labels = get_fixed_split(args.coords, args.labels, "EGFR")
    coords = pd.read_csv(args.coords, usecols=["file_id", "case_id"]).drop_duplicates()
    coords = coords[coords.case_id.isin(split)].copy()
    coords["split"] = coords.case_id.map(split)

    print("1) split sizes")
    for k in ["train", "val", "test"]:
        d = coords[coords.split == k]
        pats = d.case_id.unique()
        print(f"   {k:5s} patients {len(pats):3d} (EGFR+ {sum(case_labels[c] for c in pats):2d}) | slides {d.file_id.nunique():3d}")

    P = {k: set(coords[coords.split == k].case_id) for k in ["train", "val", "test"]}
    S = {k: set(coords[coords.split == k].file_id) for k in ["train", "val", "test"]}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (P[a] & P[b]), f"patient overlap {a}/{b}"
        assert not (S[a] & S[b]), f"slide overlap {a}/{b}"
    print("2) patient overlap between splits: 0   OK")
    print("3) slide overlap between splits:   0   OK")

    assert (coords.groupby("file_id").case_id.nunique() == 1).all(), "a slide maps to several patients"
    print("4) every slide belongs to exactly one patient   OK")

    n_slides = coords.groupby("case_id").file_id.nunique()
    multi = n_slides[n_slides > 1].index
    assert (coords[coords.case_id.isin(multi)].groupby("case_id").split.nunique() == 1).all()
    print(f"5) {len(multi)} patients with several slides, each entirely inside one split   OK")

    saved = Path(args.saved_test_slides)
    if saved.exists():
        stored = set(pd.read_csv(saved).file_id)
        assert stored == S["test"], "rebuilt test slides differ from the stored run"
        print(f"6) rebuilt test slides == test slides stored in {saved} ({len(stored)} slides)   OK")


if __name__ == "__main__":
    main()
