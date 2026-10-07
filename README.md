# Evaluation of DeepPATHO Generalizability for Multi-Gene Mutation Prediction in TCGA-LUAD

Can H&E slides of lung adenocarcinoma (TCGA-LUAD) predict gene mutations, **EGFR in particular**? This repository
re-implements the baselines of Coudray et al. 2018 (Inception v3) and Kosaraju et al. 2025 (DeepPATHO, multi-magnification
attention), together with foundation-model features (UNI / UNI2-h, Phikon-v2), and tests them under a leakage-free
protocol. The aim is a **trustworthy evaluation**, not a leaderboard.

**Read [RESULTS.md](RESULTS.md) first.** Short version:

- Patient-level stratified 70/15/15 split (seed 42), 0 patient overlap; shuffled-label controls sit at AUC ≈ 0.5;
  STK11/TP53 positive controls are recovered; the training code reaches slide AUC 0.97 on tumor-vs-normal.
- EGFR is hard here: basic Inception v3 gives test AUC 0.504 with degenerate precision/recall; UNI2-h slide features give
  cross-validated AUC 0.687 (95% CI 0.61-0.78).
- The biggest lever found was the **label definition**: restricting positives to known activating variants (L858R, exon-19
  deletions, ...) raises CV AUC to 0.760, while an `other`-variants control stays at chance. The variant classification
  is analyst-defined and needs expert review.
- Only 9 EGFR-positive test patients, and from-scratch CNN runs vary by up to ±0.14 AUC between reruns. Conclusions rest
  on cross-validation with bootstrap CIs. See "Limitations" in RESULTS.md.

A Korean code walkthrough is in [docs/CODE_TOUR.md](docs/CODE_TOUR.md).

## Layout

| Path | What |
|---|---|
| `src/deeppatho_gen/dataset.py` | patient-level stratified split (`stratified_case_level_split`), patch datasets |
| `src/deeppatho_gen/metrics.py` | AUC, precision/recall/F1, validation-threshold selection, slide-level aggregation |
| `src/deeppatho_gen/data/` | cBioPortal labels, GDC manifest, EGFR variant details and classification |
| `src/deeppatho_gen/preprocessing/` | WSI tiling |
| `src/deeppatho_gen/model/` | DeepPATHO MMA reimplementation, torchvision Inception v3 baseline |
| `src/deeppatho_gen/inception_v3_egfr_train.py` | basic Inception v3 EGFR baseline (`configs/egfr_inception_v3/`) |
| `src/deeppatho_gen/deeppath_style/` | Coudray-style 512px tiles, 10-output Inception from scratch, tumor-vs-normal sanity test |
| `src/deeppatho_gen/slide_level/` | pooled-feature logistic regression, attention MIL, CV, patch-level heads |
| `src/deeppatho_gen/diagnostics/` | split-overlap check and other sanity checks |
| `data/raw/*.csv` | public label tables (cBioPortal) and variant classification outputs |
| `results/` | small result files behind every number in RESULTS.md (`scripts/collect_results.py`) |
| `scripts/` | result collection, run supervisor, legacy full-pipeline shell scripts |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for the exact versions used
pip install -e .                       # or: export PYTHONPATH=src
```

OpenSlide must be installed on the system for tiling. Training was done on a single NVIDIA Blackwell GPU.

## Data and external inputs

These are **not** in the repository:

1. **Slides** (534 diagnostic slides, 471 patients, ~413 GB): download from the GDC with `gdc-client` using the
   manifest created by `python -m deeppatho_gen.data.gdc_manifest` into `data/raw/svs/<file_id>/*.svs`.
2. **Labels**: `python -m deeppatho_gen.data.fetch_mutation_labels` (cBioPortal study `luad_tcga_pan_can_atlas_2018`);
   the resulting tables are tracked in `data/raw/`. EGFR variant detail and categories:
   `python -m deeppatho_gen.data.fetch_egfr_mutation_details`, then `python -m deeppatho_gen.data.define_egfr_variants`.
3. **Tiles / patches**: `python -m deeppatho_gen.preprocessing.tile_wsi` (20x patches), and
   `python -m deeppatho_gen.deeppath_style.tile_512` (512px tiles).
4. **Foundation-model weights**: UNI and UNI2-h are gated on Hugging Face (request access and `huggingface-cli login`);
   Phikon-v2 is public. **UNI2-h slide/patch features (`data/processed/uni_v2_luad_slide_pooled.npz`, 1536-d) were
   extracted outside this repository with TRIDENT**; `slide_level/pool_features.py` only pools them. Re-extracting
   requires access to the gated weights.

## Reproducing the main results

```bash
# 1. Split sanity check (no GPU)
python -m deeppatho_gen.diagnostics.check_egfr_split_overlap

# 2. Basic Inception v3 baseline (GPU, needs 20x patches)
python -m deeppatho_gen.inception_v3_egfr_train --config configs/egfr_inception_v3/train_config.yaml

# 3. Slide-level cross-validation with UNI2-h features (CPU, minutes)
python -m deeppatho_gen.slide_level.cv_baseline --help
python -m deeppatho_gen.slide_level.abmil_cv --help
python -m deeppatho_gen.slide_level.fixed_split_logreg --help

# 4. DeepPATH-style 10-gene Inception from scratch (GPU, days)
scripts/supervise_run.sh configs/deeppath_style/stageA_dx.yaml runs/deeppath_style/stageA_dx runs/deeppath_style/stageA_dx.log

# 5. Collect small result files for the tables
python scripts/collect_results.py
```

Exact options of each script are in its `--help`/config. The commands above are the entry points, not a one-click
pipeline; steps 2 and 4 need the tiles from "Data and external inputs".

## Reproducibility limits (please read)

- GPU training is not bit-reproducible. The from-scratch 10-gene Inception gave TP53 0.685 in one run and 0.547 in a
  rerun with the same settings; treat single-run per-gene numbers as ±0.14.
- UNI2-h features come from an external extraction step with gated weights (see above).
- The tissue-source-site confound check (leave-one-site-out AUC 0.53) was run interactively and has no script.
- The variant classification in `define_egfr_variants.py` is the analyst's reading of the literature, not a validated
  annotation.
- `data/raw/` label files contain only public TCGA case IDs and mutation calls.

## Origins and legacy code

The DeepPATHO MMA model here is a from-scratch PyTorch reconstruction from the paper's methods (the official repo does
not run as published and releases no data pipeline or weights). Earlier first-phase scripts (`train.py`, `evaluate.py`,
`scripts/run_full_experiment*.sh`, `egfr_scratch_train.py`, `roadmap_egfr_train.py`) are kept for provenance; they use an
older non-stratified split and are not used for the headline numbers.

## References

- Coudray et al., *Nat Med* 2018, "Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning".
- Kosaraju et al., *npj Digit Med* 2025, DeepPATHO.
- Chen et al., *Nat Med* 2024 (UNI); Filiot et al. 2024 (Phikon-v2).
