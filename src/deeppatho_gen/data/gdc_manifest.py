"""Build a GDC manifest of diagnostic H&E whole-slide images for TCGA-LUAD.

Phase 1, Step 2: still no GPU needed. This queries the public GDC REST API
for "Diagnostic Slide" SVS files (matching the Coudray et al. / Kosaraju
et al. use of frozen-section H&E diagnostic slides), restricted to the case
IDs present in the mutation-label table so we don't download slides we can't
label.

Usage:
    python -m deeppatho_gen.data.gdc_manifest \
        --labels data/raw/mutation_labels.csv \
        --project TCGA-LUAD \
        --out data/raw/gdc_manifest.csv

The resulting CSV has one row per slide file with file_id, file_name,
case_id, and can be fed directly to `gdc-client download -m <manifest>`
(the GDC Data Transfer Tool) or downloaded over HTTPS with
`data/gdc_manifest.py --download` (slow; prefer gdc-client for bulk).

NOTE ON ACCESS: TCGA diagnostic slides are "open" access (no dbGaP
application needed), unlike raw sequencing reads. You do NOT need a GDC
token for this step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

GDC_API = "https://api.gdc.cancer.gov"


def query_slides(project: str, case_ids: list[str] | None = None, page_size: int = 2000) -> list[dict]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
        ],
    }
    if case_ids:
        filters["content"].append(
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}}
        )

    fields = ",".join(
        [
            "file_id",
            "file_name",
            "file_size",
            "data_type",
            "experimental_strategy",
            "cases.submitter_id",
            "cases.case_id",
        ]
    )

    results, offset = [], 0
    while True:
        body = {
            "filters": filters,
            "fields": fields,
            "format": "JSON",
            "size": page_size,
            "from": offset,
        }
        # POST (not GET): the case_id filter can make the querystring exceed
        # server URL-length limits (HTTP 414) once we're filtering on ~500+
        # TCGA barcodes at once.
        r = requests.post(f"{GDC_API}/files", json=body, timeout=60)
        r.raise_for_status()
        payload = r.json()["data"]
        hits = payload["hits"]
        results.extend(hits)
        total = payload["pagination"]["total"]
        offset += page_size
        if offset >= total:
            break
    return results


def flatten(hits: list[dict]) -> pd.DataFrame:
    rows = []
    for h in hits:
        case = h["cases"][0]
        rows.append(
            {
                "file_id": h["file_id"],
                "file_name": h["file_name"],
                "file_size": h.get("file_size"),
                "case_id": case["submitter_id"],
                "gdc_case_uuid": case["case_id"],
                "experimental_strategy": h.get("experimental_strategy"),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/raw/mutation_labels.csv",
                     help="restrict to case_ids in this file; omit filter with --all-cases")
    ap.add_argument("--all-cases", action="store_true")
    ap.add_argument("--project", default="TCGA-LUAD")
    ap.add_argument("--out", default="data/raw/gdc_manifest.csv")
    args = ap.parse_args()

    case_ids = None
    if not args.all_cases:
        labels = pd.read_csv(args.labels)
        case_ids = sorted(labels["case_id"].unique().tolist())
        print(f"Restricting to {len(case_ids)} cases from {args.labels}")

    hits = query_slides(args.project, case_ids)
    df = flatten(hits)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Found {len(df)} diagnostic slide files across {df['case_id'].nunique()} cases")
    print(f"Wrote manifest to {out_path}")
    print("\nTo bulk-download with the official GDC Data Transfer Tool:")
    print(f"  gdc-client download -m <(python -c \"import pandas as pd; "
          f"print(pd.read_csv('{out_path}')[['file_id']].rename(columns={{'file_id':'id'}}).to_csv(index=False))\")")


if __name__ == "__main__":
    main()
