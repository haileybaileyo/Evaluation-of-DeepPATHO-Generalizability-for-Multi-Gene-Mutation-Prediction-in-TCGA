#!/bin/bash
# Watches the full-experiment pipeline and relaunches it (via the
# restart-safe v2 orchestrator) if it ever stops before producing a result.
# Does NOT touch the currently-running v1 orchestrator process or its
# script file -- only observes process liveness and the filesystem.
#
# "Done" = runs/phase1_gene_mutation/slide_auc_table.csv exists (evaluate.py's
# final output). Nothing else counts, so a crash anywhere in stages 1-4
# gets noticed and retried.
set -u
cd "$(dirname "$0")/.."

LOG_DIR=runs/full_experiment/logs
WATCHDOG_LOG="$LOG_DIR/05_watchdog.log"
RESULT_FILE=runs/phase1_gene_mutation/slide_auc_table.csv
DEADLINE_FILE=runs/full_experiment/deadline.epoch
CHECK_INTERVAL=45

mkdir -p "$LOG_DIR"
log() { echo "[$(date)] $1" | tee -a "$WATCHDOG_LOG"; }

log "Watchdog started (pid $$), checking every ${CHECK_INTERVAL}s"

is_pipeline_alive() {
    pgrep -f "run_full_experiment.*\.sh|deeppatho_gen\.data\.download_slides|deeppatho_gen\.preprocessing\.tile_wsi|deeppatho_gen\.train|deeppatho_gen\.evaluate" > /dev/null 2>&1
}

while true; do
    if [ -f "$RESULT_FILE" ]; then
        log "Result found at $RESULT_FILE -- experiment complete. Watchdog exiting."
        exit 0
    fi

    if [ -f "$DEADLINE_FILE" ]; then
        REMAIN=$(python3 -c "import time; print(max(0.0, (float(open('$DEADLINE_FILE').read()) - time.time())/3600))" 2>/dev/null || echo "999")
        if python3 -c "exit(0 if $REMAIN <= 0 else 1)" 2>/dev/null; then
            log "Deadline has passed with no result yet. Watchdog stopping (won't restart past the deadline)."
            exit 1
        fi
    fi

    if is_pipeline_alive; then
        : # still running, nothing to do
    else
        log "No pipeline process is running and no result exists -- restarting via v2 orchestrator."
        setsid nohup ./scripts/run_full_experiment_v2.sh > "$LOG_DIR/nohup_wrapper_v2.log" 2>&1 < /dev/null &
        disown
        log "Relaunched (new orchestrator pid group started)."
        sleep 10  # give it a moment to actually start before we check again
    fi

    sleep "$CHECK_INTERVAL"
done
