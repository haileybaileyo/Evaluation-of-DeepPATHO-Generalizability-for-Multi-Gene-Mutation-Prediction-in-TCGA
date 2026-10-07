"""Download the DETAILED EGFR mutation records (not just 0/1) from cBioPortal.

fetch_mutation_labels.py reduces every (sample, gene) to 0/1 and throws the
details away. To decide which EGFR variants are known activating mutations we
need the details: which amino acid changed (proteinChange), what kind of change
(mutationType: missense / in-frame deletion / ...), and where (proteinPosStart).

Source: cBioPortal public API, study luad_tcga_pan_can_atlas_2018, mutation
profile "..._mutations" (whole-exome sequencing, MAF), gene EGFR (Entrez 1956).
This is the same source and the same call as the 0/1 label table, so the 86
records here are exactly the ones behind the 70 "EGFR = 1" patients.

Output: data/raw/egfr_mutation_details.csv (one row per mutation record)

Usage:
    python -m deeppatho_gen.data.fetch_egfr_mutation_details
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .fetch_mutation_labels import (
    fetch_gene_ids,
    fetch_mutation_profile_id,
    fetch_mutations,
    fetch_sample_list,
)

STUDY = "luad_tcga_pan_can_atlas_2018"
KEEP = ["sampleId", "patientId", "proteinChange", "mutationType", "variantType", "startPosition", "endPosition",
        "keyword", "referenceAllele", "variantAllele", "mutationStatus", "validationStatus", "aminoAcidChange",
        "proteinPosStart", "proteinPosEnd", "refseqMrnaId", "tumorAltCount", "tumorRefCount"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/raw/egfr_mutation_details.csv")
    ap.add_argument("--study", default=STUDY)
    args = ap.parse_args()

    gene_ids = fetch_gene_ids(["EGFR"])
    profile = fetch_mutation_profile_id(args.study)
    samples = fetch_sample_list(args.study)
    records = fetch_mutations(profile, samples, list(gene_ids.values()))
    df = pd.DataFrame(records)
    df = df[[c for c in KEEP if c in df.columns]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} EGFR mutation records from {df.patientId.nunique()} patients -> {args.out}")
    print(df.mutationType.value_counts().to_string())


if __name__ == "__main__":
    main()
