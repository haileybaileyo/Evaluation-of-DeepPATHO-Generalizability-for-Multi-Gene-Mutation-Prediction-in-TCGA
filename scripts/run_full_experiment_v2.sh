#!/bin/bash
# Full-scale experiment, deadline-driven, restart-safe (v2).
#
# Differences from v1 (kept as a separate file deliberately -- v1 may still
# be running when this is created, and editing a live-executing bash script
# in place risks corrupting its read position):
#   - reads the ORIGINAL deadline from runs/full_experiment/deadline.epoch
#     if present, instead of computing a fresh "now + N hours" each launch,
#     so a restart doesn't silently extend the total time budget
#   - retries stage 3 (training) automatically if it dies for any reason
#     (crash, OOM, kill) as long as time remains -- train.py resumes from
#     its own checkpoint, so a retry picks up where it left off
#   - stage 4 (evaluate) only runs if training actually produced
#     test_patch_scores.csv; the presence of the final slide_auc_table.csv
#     is what watchdog.sh treats as "genuinely done"
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

log "(v2) Using deadline: $(date -d "@${DEADLINE_EPOCH%.*}" 2>/dev/null || date), $(remaining_hours)h remaining"

RESULT_FILE=runs/phase1_gene_mutation/slide_auc_table.csv
if [ -f "$RESULT_FILE" ]; then
    log "Result already exists at $RESULT_FILE -- nothing to do."
    exit 0
fi

log "(v2) Stage 1/4: downloading all slides (resumable)"
python3 -u -m deeppatho_gen.data.download_slides \
    --manifest data/raw/gdc_manifest.csv --out-dir data/raw/svs --workers 8 \
    >> "$LOG_DIR/01_download.log" 2>&1
log "(v2) Stage 1/4 done (exit=$?)"

REMAIN=$(remaining_hours)
TILE_BUDGET=$(python3 -c "print(max(0.25, $REMAIN * 0.35))")
log "(v2) Stage 2/4: tiling (${REMAIN}h left, budget ${TILE_BUDGET}h)"
python3 -u -m deeppatho_gen.preprocessing.tile_wsi \
    --manifest data/raw/gdc_manifest.csv --slide-dir data/raw/svs \
    --out data/processed/patch_coords.csv --min-tissue-fraction 0.20 \
    --max-patches-per-slide 600 --resume --workers 12 \
    --time-budget-hours "$TILE_BUDGET" \
    >> "$LOG_DIR/02_tiling.log" 2>&1
log "(v2) Stage 2/4 done (exit=$?)"

TEST_SCORES=runs/phase1_gene_mutation/test_patch_scores.csv
MAX_RETRIES=10
attempt=0
while [ ! -f "$TEST_SCORES" ] && [ "$attempt" -lt "$MAX_RETRIES" ]; do
    REMAIN=$(remaining_hours)
    if python3 -c "exit(0 if $REMAIN > 0.1 else 1)"; then :; else
        log "(v2) No time left before deadline; stopping without a trained model."
        break
    fi
    TRAIN_BUDGET=$(python3 -c "print(max(0.1, $REMAIN - 0.75))")
    attempt=$((attempt + 1))
    log "(v2) Stage 3/4: training attempt $attempt/$MAX_RETRIES (${REMAIN}h left, budget ${TRAIN_BUDGET}h)"
    python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/train_config.yaml'))
cfg['optim']['max_hours'] = $TRAIN_BUDGET
yaml.safe_dump(cfg, open('configs/train_config_runtime.yaml', 'w'))
"
    python3 -u -m deeppatho_gen.train \
        --config configs/train_config_runtime.yaml --genes-config configs/genes.yaml \
        >> "$LOG_DIR/03_train.log" 2>&1
    exit_code=$?
    log "(v2) Stage 3/4 attempt $attempt exited with code $exit_code"
    if [ -f "$TEST_SCORES" ]; then
        break
    fi
    sleep 5
done

if [ -f "$TEST_SCORES" ]; then
    log "(v2) Stage 4/4: evaluation"
    python3 -u -m deeppatho_gen.evaluate \
        --patch-scores "$TEST_SCORES" \
        --labels data/raw/mutation_labels.csv \
        --genes-config configs/genes.yaml \
        --out "$RESULT_FILE" \
        >> "$LOG_DIR/04_evaluate.log" 2>&1
    log "(v2) Stage 4/4 done (exit=$?) -- results at $RESULT_FILE"
else
    log "(v2) No test_patch_scores.csv produced after $attempt attempt(s); cannot evaluate."
fi
