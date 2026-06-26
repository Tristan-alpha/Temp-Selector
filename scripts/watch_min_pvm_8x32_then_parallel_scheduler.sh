#!/usr/bin/env bash
set -euo pipefail

RUN_STAMP=${RUN_STAMP:-20260624_121747}
RUN_NAME=${RUN_NAME:-min_pvm_8x32_${RUN_STAMP}_parallel_watcher}
DATA_DIR=${DATA_DIR:-datasets/min_pvm_ppo_500_seed42_8x32_${RUN_STAMP}}
LOG_ROOT=${LOG_ROOT:-tmux_logs/min_pvm_8x32_${RUN_STAMP}}
TRAIN_SHARD_DIR=${TRAIN_SHARD_DIR:-${DATA_DIR}/shards}
TRAIN_PART1=${TRAIN_PART1:-${TRAIN_SHARD_DIR}/prefix_continuations_train.part1.jsonl}
POLL_SECONDS=${POLL_SECONDS:-10}
PIPELINE_PID=${PIPELINE_PID:-}
PIPELINE_PATTERN=${PIPELINE_PATTERN:-[b]ash scripts/run_min_pvm_8x32_pipeline.sh}
SCHEDULER_SCRIPT=${SCHEDULER_SCRIPT:-scripts/run_min_pvm_8x32_parallel_scheduler.sh}

mkdir -p "$LOG_ROOT"
WATCH_LOG="$LOG_ROOT/${RUN_NAME}.log"

log() {
    echo "[$(date --iso-8601=seconds)] $*" | tee -a "$WATCH_LOG"
}

find_pipeline_pid() {
    if [[ -n "$PIPELINE_PID" ]] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
        echo "$PIPELINE_PID"
        return 0
    fi
    pgrep -f "$PIPELINE_PATTERN" | head -n 1 || true
}

train_part1_running() {
    pgrep -f "extend_prefix_continuations.py .*prefix_continuations_train.part1.jsonl" >/dev/null
}

validate_train_part1() {
    python - "$TRAIN_PART1" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    raise SystemExit(1)
rows = 0
totals = Counter()
errors = []
expected_seeds = set(range(32))
with path.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        totals[str(row.get("n_total"))] += 1
        grouped = defaultdict(set)
        for item in row.get("continuations", []):
            grouped[int(item["temperature_index"])].add(int(item["seed_index"]))
        for temp_idx in range(8):
            if grouped.get(temp_idx, set()) != expected_seeds:
                errors.append(f"row {idx} temp {temp_idx}")
                break
if rows != 2097 or set(totals) != {"256"} or errors:
    raise SystemExit(f"rows={rows} totals={dict(totals)} errors={errors[:3]}")
print(f"ok rows={rows} totals={dict(totals)}")
PY
}

log "watching train part1: ${TRAIN_PART1}"
log "poll_seconds=${POLL_SECONDS}"

while true; do
    if validate_train_part1 >/tmp/min_pvm_8x32_part1_check.$$ 2>&1; then
        if train_part1_running; then
            log "train part1 output validates, but generation process is still alive; waiting"
        else
            log "train part1 complete: $(cat /tmp/min_pvm_8x32_part1_check.$$)"
            rm -f /tmp/min_pvm_8x32_part1_check.$$
            break
        fi
    else
        msg=$(tail -n 1 /tmp/min_pvm_8x32_part1_check.$$ 2>/dev/null || true)
        log "train part1 not ready yet: ${msg}"
    fi
    rm -f /tmp/min_pvm_8x32_part1_check.$$
    sleep "$POLL_SECONDS"
done

old_pid=$(find_pipeline_pid)
if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    log "terminating old serial pipeline parent after train part1 completion: pid=${old_pid}"
    kill "$old_pid" || true
    sleep 3
    if kill -0 "$old_pid" 2>/dev/null; then
        log "old pipeline parent still alive; sending SIGINT: pid=${old_pid}"
        kill -INT "$old_pid" || true
        sleep 2
    fi
else
    log "old serial pipeline parent is not alive; continuing"
fi

if train_part1_running; then
    log "refusing to start scheduler because train part1 is still running"
    exit 1
fi

log "starting parallel scheduler: ${SCHEDULER_SCRIPT}"
exec bash "$SCHEDULER_SCRIPT"
