# Results

Every number below comes from a file in [`results/`](results/) (small result files copied from the
git-ignored `runs/` folder by `scripts/collect_results.py`); the source file is named next to each table.

**Reading guide.** "Test" is the fixed patient-level split used by all experiments (stratified 70/15/15,
`split_seed` 42): 71 test patients, 86 test slides, 9 EGFR-positive test patients (11 positive slides).
A single split with so few positives is noisy, so cross-validation (CV) over all 462 patients with features is
used for the main conclusions. AUC = slide- or patient-level ROC AUC (patch probabilities are averaged per
slide). "95% CI" = patient (or slide) bootstrap.

## 0. Data and split

| | Patients | EGFR-positive patients | Slides | Patches / tiles |
|---|---|---|---|---|
| TCGA-LUAD DX slides with labels | 471 | 60 (12.7%) | 534 | - |
| train | 329 | 42 | 372 | 110,378 (≤300 per slide) |
| val | 71 | 9 | 76 | 22,451 |
| test | 71 | 9 | 86 | 25,800 |

Check: `python -m deeppatho_gen.diagnostics.check_egfr_split_overlap` (patient overlap 0, slide overlap 0,
21 multi-slide patients each inside one split, rebuilt test slides identical to a stored run).

## 1. Evidence that the setup is sound

| Check | How | Result |
|---|---|---|
| No leakage | patient-level stratified split, overlap check | 0 shared patients or slides |
| No score inflation | labels randomly permuted, same CV | AUC 0.47-0.52 (all definitions and methods) |
| Known signals recovered | STK11 / TP53 with the same pipeline | STK11 0.72 (linear) / 0.75 (attention), TP53 0.72 (CV) |
| Not at the mercy of one test split | 462 patients, 5-fold x 10, bootstrap CI | EGFR 0.687 (0.61-0.78) |
| Training code learns | easy target (tumor vs normal tiles), same code as Stage A | slide AUC 0.967, tile AUC 0.883 (6,000 steps, still improving) |

## 2. Basic Inception v3 baseline (EGFR only)

ImageNet-pretrained torchvision Inception v3, 20x patch only, no MMA, class weight `pos_weight` 5.739,
threshold chosen on validation. Source: `results/egfr_inception_v3/pretrained/test_result.json`.

| Threshold | Value | Precision | Recall | F1 | Predicted positive |
|---|---|---|---|---|---|
| chosen on val (F1-max) | 0.134 | 0.143 | 1.000 | 0.250 | 89.5% |
| fixed 0.5 | 0.500 | 0.107 | 0.273 | 0.154 | 32.6% |
| reference: "everyone positive" | - | 0.128 | 1.000 | 0.227 | 100% |

Test AUC 0.504 (slide) / 0.506 (patch). Train AUC rose from 0.81 to 0.98 while validation AUC stayed at
0.34-0.56 (memorisation); the saved model is epoch 0 (lowest validation loss). When scores carry no signal the
F1-maximising threshold collapses to "predict everyone positive".

## 3. Earlier patch-level experiments on the same split (EGFR)

| Experiment | Test slide AUC | Folder in `results/` |
|---|---|---|
| Inception v3, 20x only (above) | 0.504 | `egfr_inception_v3/pretrained` |
| twin-Inception + MMA from scratch, no pos_weight | 0.415 | `egfr_stratified/mma_from_scratch(_no_posweight)` |
| twin-Inception + MMA from scratch, pos_weight + val threshold | 0.467 | `egfr_stratified/mma_from_scratch_posweight` |
| frozen UNI (ViT-L) + MMA | 0.615 (seed 2: 0.611) | `egfr_uni/frozen`, `egfr_uni_seed1/frozen` |
| frozen Phikon-v2 + MMA | 0.567 | `egfr_phikon/frozen` |
| frozen UNI embeddings + MLP (no MMA), 5 seeds | 0.572-0.618 | `egfr_uni_embedding_ablation` |
| UNI fully fine-tuned | aborted (validation AUC collapsed to chance, catastrophic forgetting) | `egfr_uni_finetune/full/aborted_result.json` |

(`egfr_scratch` and `egfr_roadmap` used an older non-stratified 8:1:1 split and are not comparable.)

## 4. DeepPATH-style Inception, 10 genes at once ("Stage A")

Re-implementation of the method of Coudray et al. 2018: 512x512 tiles at 20x, Inception v3 trained from
scratch, 10 sigmoid outputs, TF-style RMSProp, colour augmentation. Tiles: 326,190 (≤800 per slide). Run twice
with identical settings. Source: `results/deeppath_style/stageA_dx/results.json` (run 1) and
`stageA_dx_actEGFR/results.json` (run 2; only the EGFR label differs, see section 5.3).

| Gene | Positive test slides | Run 1 | Run 2 | Coudray 2018 |
|---|---|---|---|---|
| STK11 | 6 | 0.765 | 0.715 | 0.856 |
| KRAS | 24 | 0.619 | 0.658 | 0.733 |
| KEAP1 | 22 | 0.648 | 0.656 | 0.675 |
| TP53 | 40 | 0.685 | 0.547 | 0.760 |
| LRP1B | 40 | 0.533 | 0.497 | 0.656 |
| FAT4 | 15 | 0.568 | 0.479 | 0.642 |
| EGFR | 11 / 7 | 0.509 | 0.620 | 0.826 |
| FAT1 | 7 | 0.447 | 0.338 | 0.750 |
| NF1 | 11 | 0.446 | 0.304 | 0.640 |
| SETBP1 | 12 | 0.364 | 0.346 | 0.775 |

STK11, KRAS and KEAP1 fall inside the paper's range in both runs; the other genes are unstable or at/below
chance. The genes whose labels did not change moved by up to ±0.14 between runs, which is the noise level of this
training scheme. EGFR precision/recall degenerate in both runs (validation-chosen threshold predicts ~90% positive).

## 5. Why is the EGFR signal weak? The label definition

### 5.1 Splitting EGFR mutations by type

`data/raw/mutation_labels.csv` marks a patient EGFR-positive if cBioPortal lists *any* EGFR mutation record.
`src/deeppatho_gen/data/define_egfr_variants.py` classifies the 86 records (70 patients) by amino-acid change and
position into `classic` (L858R; exon-19 in-frame deletions, codons 729-761), `activating` (G719X, E709X, L861Q,
S768I, V769X, exon-20 insertions at 761-775, L833F/V) and `other` (everything else, probably passengers or of unclear
function). **This classification is the analyst's, based on the literature, and still needs expert review.**
Patients: none 496, classic 46, activating 12, other 12 (of 566).

| Definition | Positive | Negative | Excluded | Patients / positives (566) |
|---|---|---|---|---|
| `any` (original) | any EGFR record | none | - | 566 / 70 |
| `act_all` | classic + activating | everyone else | - | 566 / 58 |
| `act_clean` | classic + activating | none | `other`-only patients | 554 / 58 |
| `classic_clean` | classic | none | activating-only, `other`-only | 542 / 46 |
| `other_only` (control) | `other`-only | none | classic, activating | 508 / 12 |

### 5.2 Cross-validation (UNI2-h slide features, 462 patients with features)

Source: `results/slide_level/R4_variants/`.

| Definition | Patients / pos. | Linear AUC (95% CI) | Shuffled labels | Attention MIL AUC (95% CI) |
|---|---|---|---|---|
| `any` | 462 / 58 | 0.687 (0.61-0.78) | 0.522 | 0.686 (0.63-0.78) |
| **`act_all`** | 462 / 48 | **0.760** (0.70-0.86) | 0.515 | **0.752** (0.70-0.85) |
| `act_clean` | 452 / 48 | 0.771 (0.71-0.87) | 0.505 | not run |
| `classic_clean` | 443 / 39 | 0.747 (0.65-0.86) | 0.490 | 0.750 (0.68-0.86) |
| `other_only` (control) | 414 / 10 | 0.585 (0.36-0.71) | 0.518 | not run |

Restricting to known activating mutations raises AUC from 0.69 to 0.76 with two different learners, while the
`other`-only control is at chance. Most of the gain comes from removing `other` patients from the positives
(`act_clean` ≈ `act_all`). Different definitions are different tasks, so AUCs are not strictly comparable;
all five definitions were tried and all are reported, and `act_all` is the pre-specified headline.

### 5.3 Same label change in the patch-level Inception (Stage A rerun)

EGFR test slide AUC 0.509 → 0.620 (CI 0.42-0.80; 11 → 7 positive slides), but genes whose labels did not change moved
by up to ±0.14 (section 4), so this difference is within noise and is not used as evidence.

## 6. Patch-level versus patient-level training

Same UNI2 features, same labels, same patients, same CV folds; only the unit of training differs
(patch-level: each patch inherits the patient label, patient score = mean patch probability).
Source: `results/slide_level/R6_patch_vs_patient/`.

| Labels | Patch-level linear | Patch-level MLP | Patient-level linear | Patient-level attention |
|---|---|---|---|---|
| `any` | 0.653 (0.59-0.74) | 0.629 | 0.687 | 0.686 |
| `act_all` | 0.727 (0.67-0.83) | 0.705 | 0.760 | 0.752 |

The gap is about 0.03 with overlapping intervals, and the label-definition gain (+0.07) is the same at both levels.
Patch-level training is therefore not the main reason for the weak signal; training Inception from scratch is a
more likely reason for the instability in section 4.

## 7. UNI2 features with the Stage A protocol (fixed split, 10 genes)

Frozen UNI2-h patch features (418,496 patches, ≤800 per slide) with a linear or MLP head, 5 seeds; same genes, split
and evaluation as section 4. Source: `results/slide_level/R5_uni2_patch_heads/results.json`.

| Gene | Stage A (Inception) | UNI2 linear | UNI2 MLP |
|---|---|---|---|
| STK11 | 0.765 | 0.805 | 0.812 |
| TP53 | 0.685 | 0.674 | 0.662 |
| LRP1B | 0.533 | 0.642 | 0.593 |
| FAT1 | 0.447 | 0.653 | 0.659 |
| EGFR | 0.509 | 0.583 | 0.599 |
| SETBP1 | 0.364 | 0.435 | 0.446 |

EGFR-only heads give 0.587 (linear) and 0.573 (MLP), i.e. no benefit from learning the 10 genes jointly. At the
validation-chosen threshold the linear 10-gene head predicts 13% of slides positive with precision 0.27 and recall
0.27 (first non-degenerate patch-level result).

## 8. Slide-level methods: fixed test split versus cross-validation

Sources: `results/slide_level/R0b_fixed_split_logreg/`, `R3_abmil_fixed/`, `R3_abmil_cv/`, `R0_R2_mean/`.

| Evaluation | Linear (mean-pooled UNI2) | Attention MIL |
|---|---|---|
| fixed test, 5 seeds | 0.583 ± 0.058 (slide), 0.620 ± 0.060 (patient) | 0.700 ± 0.028 |
| 5-fold CV x repeats, 462 patients | 0.687 (0.61-0.78) | 0.686 (0.63-0.78) |

The 0.1 advantage of attention MIL on the 9-positive test split does not reproduce in CV. With attention MIL at
threshold 0.5 on the fixed split: precision 0.27-0.32, recall 0.33-0.67, F1 0.30-0.43. In CV (all-EGFR-mutation
labels), flagging the top 20% of patients gives precision 0.29 (0.22-0.41, chance 0.13) and recall 0.47 (0.38-0.62).

## 9. Limitations

- 9 EGFR-positive test patients (6 with the activating definition): single-split numbers are unstable, which is why
  CV and confidence intervals carry the conclusions.
- Variant categories are analyst-defined (needs expert review and cross-checking with OncoKB/CIViC).
- Labels are patient-level, so every patch of a patient inherits it (weak labels); tumor-tile selection was not tested
  because it needs frozen-section slides.
- Smoking history, sex, ethnicity and stage were not controlled. The tissue-source site alone predicts EGFR status
  poorly (leave-one-out AUC 0.53; computed ad hoc, no script).
- Stage A trains from scratch and is not reproducible to the digit (same-seed reruns differ by up to 0.14 AUC).
- Slides are FFPE diagnostic (DX) slides, whereas Coudray et al. used frozen sections.
- UNI2-h features come from outside this repository (see README, "Data and external inputs").
