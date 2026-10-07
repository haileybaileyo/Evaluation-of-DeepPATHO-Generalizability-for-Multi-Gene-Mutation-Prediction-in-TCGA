# DeepPATHO-Generalizability 코드 투어 (처음 보는 사람에게 설명하는 순서)

이 문서는 폴더 안의 코드를 **연구 이야기 → 폴더 지도 → 코드 지도 → 호출 순서 → 파일별 설명** 순서로 정리한 참고서다.
(파일 안의 숫자와 설명은 코드/결과 파일에서 직접 확인한 것만 적었다.)

---

## 1. 연구 이야기 (3분 버전)

**문제.** 폐선암 환자 중 EGFR 같은 유전자에 변이가 있으면 표적치료제가 듣는다. 변이는 유전자 검사로 확인하지만 비싸고 오래 걸린다.
반면 **병리 슬라이드**(조직을 얇게 잘라 염색하고 스캔한 현미경 사진)는 진단 때 이미 만들어져 있다.
→ **"슬라이드 사진만 보고 AI가 EGFR 변이 여부를 예측할 수 있는가?"**

**데이터.** 공개 데이터 TCGA-LUAD. 슬라이드 534장(환자 471명)과 환자별 유전자 변이 정보. EGFR 변이 환자는 60명(12.7%).

**방법.** 슬라이드 한 장이 83억 화소라 통째로 모델에 못 넣는다 → **작은 조각(패치/타일)으로 자른다** → CNN이 조각마다 점수를 낸다 →
**슬라이드별로 평균**을 내서 그 환자의 예측으로 쓴다. 라벨은 환자 단위(변이 있다/없다)인데 모든 조각이 같은 정답을 받는다(약한 라벨).

**이 저장소의 원래 질문.** DeepPATHO(2025)는 "20x와 5x 두 배율을 attention으로 같이 보는 모델(MMA)"로 ALK 변이를 잘 예측했다.
이 모델이 **다른 유전자(EGFR 등)에도 통하는가?** Coudray(2018)의 Inception v3 결과와 비교하려 했다.
(공식 DeepPATHO 코드는 그대로 실행되지 않고 가중치도 없어서, 논문을 보고 **PyTorch로 직접 다시 구현**했다.)

**해 본 것.** 백본(이미지를 특징으로 바꾸는 부분)만 바꿔 가며 같은 MMA 구조를 학습했다.
처음부터 학습한 Inception, 사전학습된 병리 모델 UNI/Phikon(고정), UNI 전체 미세조정 → 대부분 **AUC 0.5~0.62**.
교수님 요청으로 순수 Inception v3를 같은 규칙으로 돌렸더니 **AUC 0.504**(신호 없음).
원인을 찾으려고 슬라이드 단위 방법(특징 평균 + 선형, attention MIL)을 교차검증으로 평가했더니 **AUC 약 0.69**: 신호는 있으나 약하다.
지금은 Coudray 방법(512px 타일, 10개 유전자 동시 학습, 색 증강)을 PyTorch로 충실히 재현해 돌리는 중이다.

**한 줄 결론(현재).** *EGFR 변이는 슬라이드 이미지에서 약하게 예측되며(AUC 약 0.6~0.7), 양성 환자가 60명뿐이라 한 번의 test(양성 9명) 결과는 크게 흔들린다.*

---

## 2. 폴더 지도 (루트: `DeepPATHO-Generalizability/`)

| 폴더/파일 | 역할 |
|---|---|
| `src/deeppatho_gen/` | **모든 코드** (아래 3번) |
| `configs/` | 실험 설정(yaml). 실험마다 폴더 하나 (`egfr_uni/`, `egfr_inception_v3/` …) |
| `data/raw/` | 내려받은 원본: 슬라이드 `svs/`(534장, 약 413GB), 라벨 `mutation_labels*.csv`, GDC 목록 `gdc_manifest*.csv` |
| `data/processed/` | 가공물: 패치 좌표 CSV, UNI 임베딩 캐시, (새로) 슬라이드 풀링 특징, 512px 타일 |
| `runs/` | **실험 결과**: 실험마다 폴더(체크포인트 `.pt`, `history.json`, `test_result.json` …) |
| `scripts/` | 처음 만든 전체 파이프라인 셸 스크립트(`run_full_experiment*.sh`)와 감시 스크립트 |
| `docs/` | 이 문서 (코드 투어, 한국어) |
| `README.md`, `RESULTS.md` | 프로젝트 설명과 결과표 (영어) |
| `results/` | `runs/`에서 모은 작은 결과 파일(json/csv) |

## 3. 코드 지도 (`src/deeppatho_gen/`)

데이터가 흐르는 순서대로 6층으로 나눈다.

### 층 1. 데이터 가져오기 — `data/`
| 파일 | 하는 일 |
|---|---|
| `fetch_mutation_labels.py` | cBioPortal에서 환자별 유전자 변이 0/1 표를 받아 `mutation_labels.csv` 생성 |
| `gdc_manifest.py` | GDC에 질의해 슬라이드 목록(`gdc_manifest.csv`) 생성 |
| `download_slides.py` | 목록대로 슬라이드(.svs)를 병렬·이어받기로 다운로드 |

### 층 2. 전처리 — `preprocessing/`, 필터
| 파일 | 하는 일 |
|---|---|
| `preprocessing/tile_wsi.py` | 슬라이드를 훑어 조직이 있는 칸의 **좌표만** CSV로 저장(256px, 20x/5x 쌍). 이미지는 저장하지 않음 |
| `train_tumor_classifier.py` | 종양/정상 패치 분류기 학습 |
| `filter_tumor_patches.py` | 위 분류기로 패치를 걸러 `patch_coords_tumor_only.csv` 생성 |

### 층 3. 데이터를 모델에 먹이기 — `dataset.py`
| 요소 | 하는 일 |
|---|---|
| `case_level_split` / `stratified_case_level_split` | **환자 단위** train/val/test 분할(후자는 양성 비율 유지) |
| `split_summary` | split별 환자·양성·패치 수 표 |
| `build_patch_index` | 패치 좌표 표 + 라벨 표를 `case_id`로 합침 |
| `PatchPairDataset` | 좌표 한 줄을 받아 **슬라이드에서 20x, 5x 패치를 읽어** 텐서로 반환(증강 포함, 열린 슬라이드는 최대 16개만 캐시) |

### 층 4. 모델 — `model/`
| 파일 | 하는 일 |
|---|---|
| `inception_blocks.py` | **직접 구현한 Inception 탑**(DilatedStem + InceptionA/B/A/B/E). 약 2M 파라미터 |
| `attention.py` | **MMA(두 배율 attention)**: 20x와 5x 특징맵이 서로를 보고 보정. 논문 수식 Eq.1~5의 재구현 |
| `backbone.py` | `DeepPATHOClassifier` = Inception 탑 2개(20x, 5x) + MMA + FC 머리 |
| `uni_encoder.py`, `uni_backbone.py` | 사전학습 UNI(ViT-L)를 백본으로 쓰는 버전(고정 또는 미세조정) |
| `phikon_encoder.py`, `phikon_backbone.py` | Phikon-v2를 백본으로 쓰는 버전 |
| `single_mag_baseline.py` | 20x만 쓰는 단일 탑(MMA 없음), TP53 진단용 |
| `inception_v3_baseline.py` (새) | torchvision Inception v3(사전학습), 20x만, MMA 없음 |

**모든 MMA 모델의 공통 구조**: `20x 패치 → 백본 → ┐ MMA → 풀링 → FC → logit(유전자 수만큼)` / `5x 패치 → 백본 → ┘`

### 층 5. 학습 스크립트 — 실험마다 하나 (`python -m deeppatho_gen.<이름> --config …`)
| 스크립트 | 실험 |
|---|---|
| `train.py` | 첫 실험: **6개 유전자 동시**, 처음부터 학습, 환자 무작위 split. `run_epoch`, `get_device`는 다른 스크립트가 가져다 씀 |
| `egfr_scratch_train.py` | EGFR 단일, 처음부터, 8:1:1 (ALK 논문 절차를 그대로) |
| `roadmap_egfr_train.py` | EGFR, Phikon 고정 + MMA, 8:1:1 |
| **`egfr_stratified_train.py`** (A) | EGFR, 처음부터 + MMA, **층화 70/15/15**, pos_weight, val 임계값 |
| **`uni_egfr_train.py`** (B) | EGFR, **UNI 고정** + MMA (`freeze_backbone: false`면 미세조정=D, 중단됨) |
| `phikon_stratified_train.py` (C) | EGFR, Phikon 고정 + MMA, 층화 70/15/15 |
| `cache_uni_embeddings.py` + `uni_embedding_ablation.py` | UNI 임베딩을 미리 저장하고 **MMA 없이 MLP만** 학습(몇 초, 시드 5번) |
| `inception_v3_egfr_train.py` (새) | 순수 Inception v3 기본 실험 (교수님 요청) |

### 층 6. 평가 — `metrics.py`, `mil/`, `evaluate.py`, `interpret/`
| 파일 | 하는 일 |
|---|---|
| `metrics.py` | `binary_metrics`(Precision/Recall/F1/AUC/PR-AUC/혼동행렬), `find_best_threshold`(val에서 F1 최대), 패치·슬라이드 단위 지표 |
| `mil/aggregation.py` | 패치 확률 → 슬라이드 점수: 평균 / 양성 타일 비율 / 최댓값, Youden·FDR 임계값 |
| `evaluate.py` | 유전자별 슬라이드 AUC와 부트스트랩 95% 구간 (Coudray 표 형식) |
| `interpret/patho_cam.py` | PATHO-CAM(모델이 어디를 봤는지 히트맵) |

### 보조 층 — `diagnostics/` (TP53 과적합 원인 진단, A~E)
| 파일 | 질문 |
|---|---|
| `audit_split.py` (A) | split이 정말 환자 단위인가? (assert로 검증) |
| `overfit_test.py` (B, C) | 작은 데이터를 외울 수 있나(정상 라벨) / 라벨을 섞어도 외우나(대조) |
| `fixed_validation.py` (D) | 실제 split에서 일반화하나. `cap_patches_per_slide`가 여기 있음 |
| `mma_vs_baseline.py` (E) | MMA가 단일 탑보다 나은가 |

### 이번에 새로 추가한 것
| 위치 | 내용 |
|---|---|
| `slide_level/` | 슬라이드 단위 분석: `pool_features.py`(UNI 특징 평균), `cv_baseline.py`(교차검증+대조), `fixed_split_logreg.py`(고정 split 평가), `abmil_cv.py`(attention MIL) |
| `deeppath_style/` | Coudray 방식 충실 재현: `tile_512.py`(512px 타일 저장), `train_multilabel.py`(10유전자 동시 Inception v3, TF식 RMSProp 등) |

---

## 4. 한 실험이 돌아가는 호출 순서 (예: Inception v3 기본 실험)

```
inception_v3_egfr_train.py · main()
 ├ yaml 설정 읽기, 시드 고정
 ├ build_index(cfg)                [egfr_stratified_train.py]
 │    ├ 패치 좌표 + 라벨 → case_id로 merge
 │    ├ stratified_case_level_split [dataset.py]  → 환자별 train/val/test
 │    └ cap_patches_per_slide       [diagnostics/fixed_validation.py] → 슬라이드당 300개
 ├ PatchPairDataset(train/val/test)  [dataset.py]  → DataLoader
 ├ InceptionV3Classifier             [model/inception_v3_baseline.py]
 ├ BCEWithLogitsLoss(pos_weight = 음성수/양성수)
 ├ epoch 반복: run_epoch(학습) → run_epoch(검증)  → val loss가 최저면 best_model.pt 저장, 8번 개선 없으면 중단
 └ best 모델로: val에서 임계값(find_best_threshold) → test 평가(binary_metrics) → test_result.json
```

## 5. 연구가 걸어온 순서 (git 기록과 파일에서 확인)

1. **스캐폴드**: 6개 유전자(STK11, EGFR, FAT1, SETBP1, KRAS, TP53) 동시 예측, 처음부터 학습하는 DeepPATHO 재구현
2. 24시간 안에 전체 파이프라인(다운로드, 타일링, 학습)을 돌리는 스크립트와 이어하기·감시 기능
3. **종양/정상 필터와 증강 추가**: 초기 학습이 증강 없이 바로 과적합해서
4. TP53에서 과적합이 심해 **진단 A~E** 수행 → 파이프라인은 외울 수 있고, 일반화가 어려움을 확인
5. **EGFR 하나로 집중**: Phikon 고정 → 처음부터 → **층화 split(70/15/15)**, pos_weight, val 임계값 도입
6. **UNI 고정(B)**, Phikon 고정(C), UNI 미세조정(D, 성능 붕괴로 중단), **임베딩 ablation**(MMA 없이도 비슷)
7. (이번 작업) 순수 Inception v3 기본 실험 → 슬라이드 단위 분석(교차검증) → DeepPATH 방식 재현

## 6. 코드에서 확인된, 알아 두면 좋은 점
- `configs/train_config.yaml`에 `weight_decay: 0.85`가 있지만, `train.py`의 optimizer 생성에는 이 값이 **전달되지 않는다**(`Adam(lr=…)`만).
- `gdc_manifest.py` 설명 주석은 "frozen-section"이라 쓰였지만 실제 요청은 **Diagnostic Slide(DX)**다.
- `uni_egfr_train.py`(B)는 임계값 0.5 고정이고 val 임계값 코드가 없다. A, C, Inception v3, 임베딩 ablation은 있다.
- `tile_wsi.py`는 슬라이드의 `objective-power`만 보고 배율을 정해 약 6%(32장)가 배율이 어긋난다. `tile_512.py`는 mpp를 우선한다.
- README의 Status 표는 오래됐다.

## 7. 강의 순서 (이 순서로 하나씩 코드를 읽는다)

| 강의 | 내용 | 읽을 파일 |
|---|---|---|
| L1 | 데이터가 만들어지는 과정 | `data/*`, `mutation_labels.csv`, `gdc_manifest.csv` |
| L2 | 슬라이드를 모델 입력으로 | `tile_wsi.py`, `dataset.py` (split, PatchPairDataset) |
| L3 | 모델: Inception 탑과 MMA | `inception_blocks.py`, `attention.py`, `backbone.py` |
| L4 | 사전학습 백본 (UNI, Phikon) | `uni_*.py`, `phikon_*.py` |
| L5 | 학습 루프 | `train.py`, `egfr_stratified_train.py`, `uni_egfr_train.py` |
| L6 | 평가와 지표 | `metrics.py`, `mil/aggregation.py`, `evaluate.py` |
| L7 | 진단과 ablation | `diagnostics/*`, `uni_embedding_ablation.py` |
| L8 | 이번에 추가한 것과 전체 결론 | `slide_level/*`, `deeppath_style/*`, `inception_v3_*` |
