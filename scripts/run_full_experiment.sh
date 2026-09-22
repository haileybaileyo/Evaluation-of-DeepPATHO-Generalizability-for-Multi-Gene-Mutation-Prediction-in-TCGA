#!/bin/bash
# Full-scale experiment, deadline-driven: download all 534 TCGA-LUAD slides,
# tile them, train DeepPATHOClassifier on the 6 phase-1 genes, then
# auto-evaluate. A hard wall-clock deadline (default 24h from launch) is
# split across the tiling and training stages so a result is GUARANTEED to
# come out by the deadline, even if tiling or training would otherwise take
# longer -- see tile_wsi.py's --time-budget-hours and train.py's max_hours.
#
# Survives the launching terminal/SSH session disconnecting: no stdin read,
# meant to be started via setsid so it gets its own session. Each stage logs
# to its own file under runs/full_experiment/logs/.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

TOTAL_BUDGET_HOURS="${1:-24}"
LOG_DIR=runs/full_experiment/logs
mkdir -p "$LOG_DIR"

DEADLINE_EPOCH=$(python3 -c "import time; print(time.time() + $TOTAL_BUDGET_HOURS*3600)")
log() { echo "=== [$(date)] $1 ===" | tee -a "$LOG_DIR/00_orchestrator.log"; }
remaining_hours() { python3 -c "import time; print(max(0.0, ($DEADLINE_EPOCH - time.time())/3600))"; }

log "Deadline set: ${TOTAL_BUDGET_HOURS}h from now ($(date -d "@${DEADLINE_EPOCH%.*}" 2>/dev/null || date))"

log "Stage 1/4: downloading all slides (resumable -- already-complete files are skipped)"
python3 -u -m deeppatho_gen.data.download_slides \
    --manifest data/raw/gdc_manifest.csv \
    --out-dir data/raw/svs \
    --workers 8 \
    >> "$LOG_DIR/01_download.log" 2>&1
log "Stage 1/4 done (exit=$?)"

REMAIN=$(remaining_hours)
TILE_BUDGET=$(python3 -c "print(max(0.5, $REMAIN * 0.35))")
log "Stage 2/4: tiling (${REMAIN}h left total, capping this stage at ${TILE_BUDGET}h)"
python3 -u -m deeppatho_gen.preprocessing.tile_wsi \
    --manifest data/raw/gdc_manifest.csv \
    --slide-dir data/raw/svs \
    --out data/processed/patch_coords.csv \
    --min-tissue-fraction 0.20 \
    --max-patches-per-slide 600 \
    --resume \
    --workers 12 \
    --time-budget-hours "$TILE_BUDGET" \
    >> "$LOG_DIR/02_tiling.log" 2>&1
log "Stage 2/4 done (exit=$?)"

REMAIN=$(remaining_hours)
TRAIN_BUDGET=$(python3 -c "print(max(0.25, $REMAIN - 0.75))")   # leave 45min buffer for eval
log "Stage 3/4: training (${REMAIN}h left total, capping this stage at ${TRAIN_BUDGET}h)"
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/train_config.yaml'))
cfg['optim']['max_hours'] = $TRAIN_BUDGET
yaml.safe_dump(cfg, open('configs/train_config_runtime.yaml', 'w'))
"
python3 -u -m deeppatho_gen.train \
    --config configs/train_config_runtime.yaml \
    --genes-config configs/genes.yaml \
    >> "$LOG_DIR/03_train.log" 2>&1
log "Stage 3/4 done (exit=$?)"

log "Stage 4/4: evaluation"
python3 -u -m deeppatho_gen.evaluate \
    --patch-scores runs/phase1_gene_mutation/test_patch_scores.csv \
    --labels data/raw/mutation_labels.csv \
    --genes-config configs/genes.yaml \
    --out runs/phase1_gene_mutation/slide_auc_table.csv \
    >> "$LOG_DIR/04_evaluate.log" 2>&1
log "Stage 4/4 done (exit=$?)"

log "FULL EXPERIMENT COMPLETE -- results at runs/phase1_gene_mutation/slide_auc_table.csv"
