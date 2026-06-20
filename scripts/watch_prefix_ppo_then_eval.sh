#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/training/min_pvm_ppo_500_seed42_ppo_success_runtime.yaml}
GPU_DEVICES=${GPU_DEVICES:-4,5}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
SEED=${SEED:-42}
RUN_STAMP=${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}
RUN_NAME=${RUN_NAME:-prefix_ppo_auto_eval_${RUN_STAMP}}
LOG_ROOT=${LOG_ROOT:-tmux_logs/prefix_ppo_auto_eval_${RUN_STAMP}}
RESULT_ROOT=${RESULT_ROOT:-results/min_pvm_ppo_500_seed42_20260618}
OUTPUT=${OUTPUT:-${RESULT_ROOT}/final_eval_seed${SEED}.json}

# Optional guards for attaching this script to an already-running training job.
WAIT_PID=${WAIT_PID:-}
STATUS_LOG=${STATUS_LOG:-}
CHECKPOINT=${CHECKPOINT:-}
POLL_SECONDS=${POLL_SECONDS:-60}
STATUS_GRACE_SECONDS=${STATUS_GRACE_SECONDS:-120}

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
MASTER_LOG="${LOG_ROOT}/${RUN_NAME}.watch_then_eval.log"
EVAL_LOG="${LOG_ROOT}/${RUN_NAME}.prefix_eval.log"

log() {
    echo "[$(date --iso-8601=seconds)] $*" | tee -a "$MASTER_LOG"
}

pid_is_running() {
    local pid="$1"
    kill -0 "$pid" 2>/dev/null || return 1
    local stat
    stat=$(ps -p "$pid" -o stat= 2>/dev/null || true)
    [[ "$stat" != Z* ]]
}

latest_status_from_log() {
    local path="$1"
    awk -F= '/^status=/{status=$2} END{print status}' "$path" 2>/dev/null || true
}

wait_for_training() {
    if [[ -z "$WAIT_PID" ]]; then
        log "wait_pid=none; starting evaluation immediately"
        return 0
    fi

    log "waiting_for_pid=${WAIT_PID} poll_seconds=${POLL_SECONDS}"
    while pid_is_running "$WAIT_PID"; do
        sleep "$POLL_SECONDS"
    done
    log "pid_finished=${WAIT_PID}"

    if [[ -z "$STATUS_LOG" ]]; then
        log "status_log=none; cannot verify training exit status"
        return 0
    fi

    log "checking_status_log=${STATUS_LOG}"
    local deadline=$((SECONDS + STATUS_GRACE_SECONDS))
    local status=""
    while (( SECONDS < deadline )); do
        if [[ -f "$STATUS_LOG" ]]; then
            status=$(latest_status_from_log "$STATUS_LOG")
            [[ -n "$status" ]] && break
        fi
        sleep 2
    done

    if [[ -z "$status" ]]; then
        log "error=no_status_found status_log=${STATUS_LOG}"
        return 2
    fi
    if [[ "$status" != "0" && "$status" != "complete" ]]; then
        log "error=training_failed status=${status}"
        return "$status"
    fi
    log "training_status=${status}"
}

wait_for_training

if [[ -n "$CHECKPOINT" ]]; then
    if [[ ! -s "$CHECKPOINT" ]]; then
        log "error=checkpoint_missing checkpoint=${CHECKPOINT}"
        exit 3
    fi
    log "checkpoint_ready=${CHECKPOINT}"
fi

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

log "eval_start config=${CONFIG} seed=${SEED} gpu_devices=${GPU_DEVICES} parallel_size=${PARALLEL_SIZE}"
log "eval_output=${OUTPUT}"
log "eval_log=${EVAL_LOG}"

set +e
CUDA_VISIBLE_DEVICES="$GPU_DEVICES" python -m ppo.prefix_eval \
    --config "$CONFIG" \
    --seed "$SEED" \
    --parallel-size "$PARALLEL_SIZE" \
    --output "$OUTPUT" \
    --run-name "$RUN_NAME" \
    --log-dir "$LOG_ROOT" \
    > "$EVAL_LOG" 2>&1
STATUS=$?
set -e

log "eval_finished status=${STATUS}"
exit "$STATUS"
