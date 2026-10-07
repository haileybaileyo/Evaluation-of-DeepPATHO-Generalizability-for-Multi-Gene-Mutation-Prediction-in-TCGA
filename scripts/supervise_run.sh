#!/bin/sh
# usage: scripts/supervise_run.sh <config.yaml> <out_dir> <log_file>
# Runs deeppath_style.train_multilabel and restarts it from its last checkpoint if it exits
# without results.json (not after a divergence, max 8 attempts).
CFG=$1; OUT=$2; LOG=$3
cd "$(dirname "$0")/.." || exit 1
n=0
while [ $n -lt 8 ]; do
  n=$((n+1)); echo "=== attempt $n $(date)" >> "$LOG"
  PYTHONPATH=src python -u -W ignore -m deeppatho_gen.deeppath_style.train_multilabel --config "$CFG" >> "$LOG" 2>&1
  rc=$?
  [ -f "$OUT/results.json" ] && { echo "=== finished OK $(date)" >> "$LOG"; exit 0; }
  tail -n 20 "$LOG" | grep -q "loss diverged" && { echo "=== diverged; not restarting" >> "$LOG"; exit 2; }
  echo "=== exited rc=$rc without results.json; restarting in 60s" >> "$LOG"; sleep 60
done
echo "=== gave up after $n attempts" >> "$LOG"; exit 3
