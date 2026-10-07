"""Split the project's EGFR label ("any EGFR mutation record") into biologically
motivated variants, to test whether a broad label definition is part of why EGFR
is hard to predict from H&E.

Input : data/raw/egfr_mutation_details.csv   (cBioPortal records, see below)
Output: data/raw/egfr_variant_categories.csv (every distinct variant -> category)
        data/raw/egfr_variants/<name>.csv    (one label file per definition;
                                              patients that must be excluded for
                                              a clean comparison are simply absent)

Categories (assigned from the protein change / position; this is a domain-knowledge
classification written by the analyst, NOT something the repo or cBioPortal provides
-- it should be reviewed by someone with EGFR/lung-pathology expertise):

  classic      L858R, or an exon-19 in-frame deletion/delins starting at codons
               729-761 (E746_A750del and relatives): the two best-known
               TKI-sensitising activating mutations
  activating   classic + other recognised activating hot spots: G719A/C/S,
               E709_T710delinsD, L861Q, S768I, V769*, exon-20 in-frame
               insertions/duplications (codons 761-775), L833F/V
  other        every remaining EGFR record (scattered missense outside the
               hot spots, extracellular-domain changes, nonsense/splice,
               T790M alone, ...): likely passengers or of unclear function

A patient takes the highest category among his/her records
(classic > activating > other).

Also writes data/raw/mutation_labels_10gene_actEGFR.csv (the 10-gene table with EGFR := act_all).

Label definitions written (column name is always EGFR so the existing
cross-validation scripts can read them with --genes EGFR):
  any            positive: any EGFR record (the current definition); negative: none
  act_all        positive: classic or activating; negative: everyone else
                 (including patients that only carry 'other' variants)
  act_clean      positive: classic or activating; negative: NO EGFR record at all
                 ('other'-only patients removed from the analysis)
  classic_clean  positive: classic; negative: NO EGFR record at all
                 (activating-but-not-classic and 'other'-only patients removed)
  other_only     positive: 'other'-only patients; negative: NO EGFR record at all
                 (classic/activating carriers removed) -- a NEGATIVE-CONTROL-like
                 definition: if the image signal comes from true activating
                 mutations this should be close to chance.

Usage:
    python -m deeppatho_gen.data.define_egfr_variants
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RANK = {"none": 0, "other": 1, "activating": 2, "classic": 3}


def categorize(row) -> tuple[str, str]:
    pc, mt = str(row["proteinChange"]), str(row["mutationType"])
    start = row["proteinPosStart"]
    if pc == "L858R":
        return "classic", "L858R (exon 21)"
    if mt == "In_Frame_Del" and pd.notna(start) and 729 <= start <= 761:
        return "classic", "exon-19 in-frame deletion/delins"
    if re.fullmatch(r"G719[ACS]", pc):
        return "activating", "G719X (exon 18)"
    if pc.startswith("E709"):
        return "activating", "E709X (exon 18)"
    if pc == "L861Q":
        return "activating", "L861Q (exon 21)"
    if pc == "S768I":
        return "activating", "S768I (exon 20)"
    if pc.startswith("V769"):
        return "activating", "V769X (exon 20)"
    if mt == "In_Frame_Ins" and pd.notna(start) and 761 <= start <= 775:
        return "activating", "exon-20 insertion/duplication"
    if pc in ("L833F", "L833V"):
        return "activating", "L833X"
    return "other", "not a recognised activating hot spot (passenger/unclear)"


def main():
    d = pd.read_csv("data/raw/egfr_mutation_details.csv")
    d[["category", "rationale"]] = d.apply(lambda r: pd.Series(categorize(r)), axis=1)
    cats = (d.groupby(["proteinChange", "mutationType", "category", "rationale"]).size().rename("records").reset_index()
            .sort_values(["category", "records"], ascending=[True, False]))
    cats.to_csv("data/raw/egfr_variant_categories.csv", index=False)

    d["rank"] = d["category"].map(RANK)
    pat = d.groupby("patientId")["rank"].max()
    cat_of = {p: {v: k for k, v in RANK.items()}[r] for p, r in pat.items()}

    allp = pd.read_csv("data/raw/mutation_labels.csv")[["case_id", "sample_id"]].drop_duplicates("case_id")
    allp["cat"] = allp.case_id.map(cat_of).fillna("none")

    out = Path("data/raw/egfr_variants")
    out.mkdir(parents=True, exist_ok=True)
    spec = {
        "any":           (lambda c: c != "none",                        lambda c: True),
        "act_all":       (lambda c: c in ("classic", "activating"),     lambda c: True),
        "act_clean":     (lambda c: c in ("classic", "activating"),     lambda c: c in ("classic", "activating", "none")),
        "classic_clean": (lambda c: c == "classic",                     lambda c: c in ("classic", "none")),
        "other_only":    (lambda c: c == "other",                       lambda c: c in ("other", "none")),
    }
    summ = []
    for name, (is_pos, keep) in spec.items():
        sub = allp[allp.cat.map(keep)].copy()
        sub["EGFR"] = sub.cat.map(is_pos).astype(int)
        sub[["case_id", "sample_id", "EGFR"]].to_csv(out / f"{name}.csv", index=False)
        summ.append((name, len(sub), int(sub.EGFR.sum())))

    # Stage A (10 genes at once): the same 10-gene label table with ONLY its EGFR column replaced by the
    # "act_all" definition; the original EGFR column is kept as EGFR_any.
    l10 = pd.read_csv("data/raw/mutation_labels_10gene.csv")
    act = pd.read_csv(out / "act_all.csv").set_index("case_id")["EGFR"]
    l10["EGFR_any"] = l10["EGFR"]
    l10["EGFR"] = l10.case_id.map(act).fillna(0).astype(int)
    l10.to_csv("data/raw/mutation_labels_10gene_actEGFR.csv", index=False)

    print("[변이 종류별 기록 수]")
    print(d.groupby("category").size().to_string())
    print("\n[환자 수 (전체 라벨 환자 566명 기준)]")
    print(allp.cat.value_counts().to_string())
    print("\n[분류표: 변이 → 종류]")
    print(cats.to_string(index=False))
    print("\n[라벨 정의별 (분석 환자 수, 양성 수)]")
    for n, tot, pos in summ:
        print(f"  {n:14s} 환자 {tot:3d}명, 양성 {pos:3d}명 ({100 * pos / tot:.1f}%)")


if __name__ == "__main__":
    main()
