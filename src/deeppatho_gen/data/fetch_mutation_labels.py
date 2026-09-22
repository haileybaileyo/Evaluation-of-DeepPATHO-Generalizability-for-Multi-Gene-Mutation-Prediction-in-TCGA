"""Fetch per-sample mutation status for target genes from cBioPortal.

This is Phase 1, Step 1 of the plan: it needs only network access (no GPU,
no WSI downloads) and produces the ground-truth label table used later to
supervise the model.

Usage:
    python -m deeppatho_gen.data.fetch_mutation_labels \
        --genes-config configs/genes.yaml \
        --out data/raw/mutation_labels.csv

Output columns:
    case_id, sample_id, <GENE_1>, <GENE_2>, ... (0/1 mutated flags)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

API = "https://www.cbioportal.org/api"


def _get(path: str, **params) -> dict | list:
    url = f"{API}/{path}"
    for attempt in range(3):
        r = requests.get(url, params=params, timeout=60)
        if r.ok:
            return r.json()
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()


def _post(path: str, json_body: dict, **params) -> list:
    url = f"{API}/{path}"
    for attempt in range(3):
        r = requests.post(url, params=params, json=json_body, timeout=120)
        if r.ok:
            return r.json()
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()


def fetch_gene_ids(hugo_symbols: list[str]) -> dict[str, int]:
    genes = _post("genes/fetch", json_body=hugo_symbols, geneIdType="HUGO_GENE_SYMBOL")
    missing = set(hugo_symbols) - {g["hugoGeneSymbol"] for g in genes}
    if missing:
        print(f"WARNING: cBioPortal did not resolve gene symbols: {missing}", file=sys.stderr)
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in genes}


def fetch_mutation_profile_id(study_id: str) -> str:
    profiles = _get(f"studies/{study_id}/molecular-profiles")
    for p in profiles:
        if p.get("molecularAlterationType") == "MUTATION_EXTENDED":
            return p["molecularProfileId"]
    raise RuntimeError(f"No mutation profile found for study {study_id}")


def fetch_sample_list(study_id: str) -> list[str]:
    samples = _get(f"studies/{study_id}/samples", pageSize=10_000)
    return [s["sampleId"] for s in samples]


def fetch_mutations(profile_id: str, sample_ids: list[str], entrez_ids: list[int]) -> list[dict]:
    body = {"entrezGeneIds": entrez_ids, "sampleIds": sample_ids}
    return _post(
        f"molecular-profiles/{profile_id}/mutations/fetch",
        json_body=body,
        projection="SUMMARY",
    )


def build_label_table(study_id: str, genes: list[str]) -> pd.DataFrame:
    print(f"Resolving {len(genes)} gene symbols ...")
    gene_ids = fetch_gene_ids(genes)
    entrez_to_symbol = {v: k for k, v in gene_ids.items()}

    print(f"Fetching sample list for study {study_id} ...")
    sample_ids = fetch_sample_list(study_id)
    print(f"  {len(sample_ids)} samples")

    print("Locating mutation molecular profile ...")
    profile_id = fetch_mutation_profile_id(study_id)

    print("Fetching mutation calls (this can take a minute) ...")
    mutations = fetch_mutations(profile_id, sample_ids, list(gene_ids.values()))
    print(f"  {len(mutations)} mutation records")

    mutated = {(m["sampleId"], entrez_to_symbol[m["entrezGeneId"]]) for m in mutations}

    rows = []
    for sid in sample_ids:
        case_id = sid[:12]  # TCGA barcode: patient id is first 12 chars
        row = {"case_id": case_id, "sample_id": sid}
        for g in genes:
            row[g] = int((sid, g) in mutated)
        rows.append(row)

    df = pd.DataFrame(rows).drop_duplicates(subset="sample_id")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genes-config", default="configs/genes.yaml")
    ap.add_argument("--out", default="data/raw/mutation_labels.csv")
    ap.add_argument(
        "--include-phase2", action="store_true",
        help="Also fetch phase2_genes (e.g. ALK) in the same table.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.genes_config).read_text())
    genes = list(cfg["phase1_genes"])
    if args.include_phase2:
        genes += list(cfg.get("phase2_genes", []))

    df = build_label_table(cfg["study_id"], genes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} samples to {out_path}")
    print("\nMutation frequency per gene:")
    print(df[genes].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
