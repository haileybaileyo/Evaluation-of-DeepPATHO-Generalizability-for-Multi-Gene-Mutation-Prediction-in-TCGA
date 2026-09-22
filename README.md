# DeepPATHO-Generalizability

Extends Kosaraju et al. 2025 ("DeepPATHO", npj Digital Medicine) beyond
ALK-only screening: does the same multi-magnification attention (MMA)
architecture generalize to *other* gene mutations in TCGA-LUAD, benchmarked
directly against the numbers Coudray et al. 2018 (Nat Med) published for
plain Inception v3 on the same task?

## Status

Everything in "Phase 1, steps 1-2" below has been run against the live
public APIs and verified to work on this machine (no GPU, no WSI downloads
needed). Everything from "step 3" onward is implemented and unit-tested
with synthetic data, but has **not** been run on real slides yet -- that
needs a GPU and several hundred GB of downloaded TCGA .svs files.

| Step | What | Verified how |
|---|---|---|
| 1. Mutation labels | `data/fetch_mutation_labels.py` | Ran live against cBioPortal; frequencies match paper's Table 4 within ~2pp |
| 2. Slide manifest | `data/gdc_manifest.py` | Ran live against GDC; found 534 diagnostic slides / 471 cases |
| 3. Tiling | `preprocessing/tile_wsi.py` | Code review only -- needs downloaded .svs + openslide |
| 4. Model | `model/{attention,inception_blocks,backbone}.py` | Forward + backward pass verified on random tensors (1.99M params) |
| 5. MIL aggregation | `mil/aggregation.py` | Unit-tested on synthetic patch scores |
| 6. Training loop | `train.py` | Code review only -- needs GPU + tiled data |
| 7. Evaluation | `evaluate.py` | Ran end-to-end on synthetic patch scores; produces a Coudray-comparison table |
| 8. Interpretability | `interpret/patho_cam.py` | Forward + backward (CAM) pass verified on random tensors |

## Why this design

The [official DeepPATHO repo](https://github.com/datax-lab/DeepPATHO) only
publishes the raw Keras layer definitions for the backbone (and even those
have a missing import -- `downscale.py` doesn't exist in the repo, so the
published code doesn't actually run as-is). It has no data pipeline, no MIL
aggregation code (HipoMap/CAMIL are separate publications), no PATHO-CAM,
and no pretrained weights ("we will update repo with the pretrained models
in version 2"). So this project is a from-scratch, PyTorch reconstruction
of the *architecture described in the paper's Methods section* (Eq. 1-7),
not a fork of their code. See the docstring at the top of
`model/attention.py` for exactly which parts are exact reproductions of the
paper's equations vs. our own reasonable-but-undocumented design choices
(the up/downsampling mechanics for synchronizing the two magnifications
aren't fully specified in the paper).

## Target genes (Phase 1)

Chosen to match Coudray et al. 2018 Table 1 exactly, so we can compare
per-gene AUC cell-by-cell: **STK11, EGFR, FAT1, SETBP1, KRAS, TP53**. ALK is
reserved for Phase 2 as a sanity check (Coudray et al. couldn't predict it
with plain Inception v3; if DeepPATHO's attention mechanism is doing
something real, it should do better here, matching Kosaraju et al.'s
headline result). See `configs/genes.yaml`.

## Running it

```bash
pip install -r requirements.txt

# Step 1-2: no GPU, no downloads, just network access. ~2 minutes.
python -m deeppatho_gen.data.fetch_mutation_labels --include-phase2
python -m deeppatho_gen.data.gdc_manifest

# Step 3: needs the actual slides downloaded first, e.g. with the GDC Data
# Transfer Tool (gdc-client) pointed at data/raw/gdc_manifest.csv. This is
# large (534 slides, likely >200GB) -- budget storage/bandwidth accordingly.
python -m deeppatho_gen.preprocessing.tile_wsi

# Step 4-onward: needs a GPU.
python -m deeppatho_gen.train
python -m deeppatho_gen.evaluate --patch-scores runs/.../test_patch_scores.csv
```

## What to decide before running Step 3 for real

1. **Storage & compute** -- 534 diagnostic WSIs at ~0.5-1GB each. Confirm
   where these live before kicking off a bulk download.
2. **Tissue threshold** -- defaults to Kosaraju et al.'s 20% (vs. Coudray et
   al.'s 50%-background-discard rule); both are one CLI flag away, worth
   running a quick sensitivity check on a handful of slides first.
3. **Backbone fidelity** -- `model/inception_blocks.py` implements the
   paper's five-block Inception *pattern* with reasonable channel widths,
   since the exact widths are only in a Supplementary Table we don't have
   access to. If you get access to that table, that's the one place worth
   tightening before drawing strong conclusions from an AUC gap vs. Coudray
   et al.
