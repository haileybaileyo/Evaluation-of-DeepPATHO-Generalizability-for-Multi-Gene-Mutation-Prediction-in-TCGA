#!/bin/bash
# v3: adds tumor/normal-tile filtering before mutation-model training, and
# uses the now-augmented PatchPairDataset. Reads the persisted original
# deadline like v2, and retries the final training stage on failure.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

LOG_DIR=runs/full_experiment/logs
DEADLINE_FILE=runs/full_experiment/deadline.epoch
mkdir -p "$LOG_DIR"

if [ -f "$DEADLINE_FILE" ]; then
    DEADLINE_EPOCH=$(cat "$DEADLINE_FILE")
else
    TOTAL_BUDGET_HOURS="${1:-24}"
    DEADLINE_EPOCH=$(python3 -c "import time; print(time.time() + $TOTAL_BUDGET_HOURS*3600)")
    echo "$DEADLINE_EPOCH" > "$DEADLINE_FILE"
fi

log() { echo "=== [$(date)] $1 ===" | tee -a "$LOG_DIR/00_orchestrator.log"; }
remaining_hours() { python3 -c "import time; print(max(0.0, ($DEADLINE_EPOCH - time.time())/3600))"; }

log "(v3) Using deadline: $(date -d "@${DEADLINE_EPOCH%.*}" 2>/dev/null || date), $(remaining_hours)h remaining"

RESULT_FILE=runs/phase1_gene_mutation/slide_auc_table.csv
if [ -f "$RESULT_FILE" ]; then
    log "Result already exists at $RESULT_FILE -- nothing to do."
    exit 0
fi

log "(v3) Stage A: downloading normal-tissue slides (resumable)"
python3 -u -m deeppatho_gen.data.download_slides \
    --manifest data/raw/gdc_manifest_normal.csv --out-dir data/raw/svs_normal --workers 8 \
    >> "$LOG_DIR/0a_download_normal.log" 2>&1
log "(v3) Stage A done (exit=$?)"

REMAIN=$(remaining_hours)
TILE_BUDGET=$(python3 -c "print(max(0.25, $REMAIN * 0.15))")
log "(v3) Stage B: tiling normal slides (${REMAIN}h left, budget ${TILE_BUDGET}h)"
python3 -u -m deeppatho_gen.preprocessing.tile_wsi \
    --manifest data/raw/gdc_manifest_normal.csv --slide-dir data/raw/svs_normal \
    --out data/processed/patch_coords_normal.csv --min-tissue-fraction 0.20 \
    --max-patches-per-slide 600 --resume --workers 12 --time-budget-hours "$TILE_BUDGET" \
    >> "$LOG_DIR/0b_tile_normal.log" 2>&1
log "(v3) Stage B done (exit=$?)"

TUMOR_MODEL=runs/tumor_classifier/best_model.pt
if [ ! -f "$TUMOR_MODEL" ]; then
    REMAIN=$(remaining_hours)
    TUMOR_BUDGET=$(python3 -c "print(max(0.25, min(3.0, $REMAIN * 0.25)))")
    log "(v3) Stage C: training tumor/normal classifier (${REMAIN}h left, budget ${TUMOR_BUDGET}h)"
    python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/tumor_classifier_config.yaml'))
cfg['optim']['max_hours'] = $TUMOR_BUDGET
yaml.safe_dump(cfg, open('configs/tumor_classifier_config_runtime.yaml', 'w'))
"
    python3 -u -m deeppatho_gen.train_tumor_classifier \
        --config configs/tumor_classifier_config_runtime.yaml \
        >> "$LOG_DIR/0c_tumor_classifier.log" 2>&1
    log "(v3) Stage C done (exit=$?)"
else
    log "(v3) Stage C skipped: $TUMOR_MODEL already exists"
fi

FILTERED_COORDS=data/processed/patch_coords_tumor_only.csv
if [ -f "$TUMOR_MODEL" ] && [ ! -f "$FILTERED_COORDS" ]; then
    log "(v3) Stage D: filtering mutation-training patches by tumor probability"
    python3 -u -m deeppatho_gen.filter_tumor_patches \
        --patch-coords data/processed/patch_coords.csv \
        --tumor-model "$TUMOR_MODEL" \
        --tumor-config configs/tumor_classifier_config_runtime.yaml \
        --out "$FILTERED_COORDS" --threshold 0.5 \
        >> "$LOG_DIR/0d_filter.log" 2>&1
    log "(v3) Stage D done (exit=$?)"
elif [ -f "$FILTERED_COORDS" ]; then
    log "(v3) Stage D skipped: $FILTERED_COORDS already exists"
else
    log "(v3) Stage D SKIPPED -- no tumor model available, cannot filter. Aborting."
    exit 1
fi

TEST_SCORES=runs/phase1_gene_mutation/test_patch_scores.csv
MAX_RETRIES=10
attempt=0
while [ ! -f "$TEST_SCORES" ] && [ "$attempt" -lt "$MAX_RETRIES" ]; do
    REMAIN=$(remaining_hours)
    if python3 -c "exit(0 if $REMAIN > 0.1 else 1)"; then :; else
        log "(v3) No time left before deadline; stopping without a trained mutation model."
        break
    fi
    TRAIN_BUDGET=$(python3 -c "print(max(0.1, $REMAIN - 0.5))")
    attempt=$((attempt + 1))
    log "(v3) Stage E: mutation-model training attempt $attempt/$MAX_RETRIES (${REMAIN}h left, budget ${TRAIN_BUDGET}h)"
    python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/train_config.yaml'))
cfg['optim']['max_hours'] = $TRAIN_BUDGET
cfg['data']['index_csv'] = '$FILTERED_COORDS'
yaml.safe_dump(cfg, open('configs/train_config_runtime.yaml', 'w'))
"
    python3 -u -m deeppatho_gen.train \
        --config configs/train_config_runtime.yaml --genes-config configs/genes.yaml \
        >> "$LOG_DIR/03_train.log" 2>&1
    exit_code=$?
    log "(v3) Stage E attempt $attempt exited with code $exit_code"
    if [ -f "$TEST_SCORES" ]; then
        break
    fi
    sleep 5
done

if [ -f "$TEST_SCORES" ]; then
    log "(v3) Stage F: evaluation"
    python3 -u -m deeppatho_gen.evaluate \
        --patch-scores "$TEST_SCORES" \
        --labels data/raw/mutation_labels.csv \
        --genes-config configs/genes.yaml \
        --out "$RESULT_FILE" \
        >> "$LOG_DIR/04_evaluate.log" 2>&1
    log "(v3) Stage F done (exit=$?) -- results at $RESULT_FILE"
else
    log "(v3) No test_patch_scores.csv produced after $attempt attempt(s); cannot evaluate."
fi
