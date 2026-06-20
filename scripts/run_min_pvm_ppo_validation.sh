#!/usr/bin/env bash
set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

CONFIG=${CONFIG:-configs/training/min_pvm_ppo_500_seed42.yaml}
GPU_DEVICES=${GPU_DEVICES:-0,1}
LABEL_PARALLEL_SIZE=${LABEL_PARALLEL_SIZE:-2}
TRAIN_PARALLEL_SIZE=${TRAIN_PARALLEL_SIZE:-1}
RUN_STAMP=${RUN_STAMP:-$(date +"%Y%m%d_%H%M%S")}
RUN_NAME=${RUN_NAME:-min_pvm_ppo_500_seed42_${RUN_STAMP}}
LOG_ROOT=${LOG_ROOT:-tmux_logs/min_pvm_ppo_500_seed42_20260618}
RESULT_ROOT=${RESULT_ROOT:-results/min_pvm_ppo_500_seed42_20260618}

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"

MASTER_LOG="$LOG_ROOT/${RUN_NAME}.master.log"

run_stage() {
    local name="$1"
    shift
    local stage_log="$LOG_ROOT/${RUN_NAME}.${name}.log"
    {
        echo "===== ${name} START $(date --iso-8601=seconds) ====="
        echo "command: $*"
    } | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$@" 2>&1 | tee "$stage_log"
    {
        echo "===== ${name} END $(date --iso-8601=seconds) ====="
        echo "stage_log=${stage_log}"
    } | tee -a "$MASTER_LOG"
}

run_stage_non_blocking() {
    local name="$1"
    shift
    if run_stage "$name" "$@"; then
        return 0
    fi
    local status=$?
    {
        echo "===== ${name} NON_BLOCKING_FAILURE status=${status} $(date --iso-8601=seconds) ====="
        echo "continuing_after=${name}"
    } | tee -a "$MASTER_LOG"
    return 0
}

{
    echo "run_name=${RUN_NAME}"
    echo "config=${CONFIG}"
    echo "gpu_devices=${GPU_DEVICES}"
    echo "label_parallel_size=${LABEL_PARALLEL_SIZE}"
    echo "train_parallel_size=${TRAIN_PARALLEL_SIZE}"
    echo "log_root=${LOG_ROOT}"
    echo "result_root=${RESULT_ROOT}"
    echo "pid=$$"
    echo "started_at=$(date --iso-8601=seconds)"
} | tee "$MASTER_LOG"

run_stage train_continuations \
    python scripts/build_prefix_continuations.py \
    --config "$CONFIG" \
    --split train \
    --parallel-size "$LABEL_PARALLEL_SIZE"

run_stage val_continuations \
    python scripts/build_prefix_continuations.py \
    --config "$CONFIG" \
    --split val \
    --parallel-size "$LABEL_PARALLEL_SIZE"

run_stage prefix_value_training \
    python -m mil.value_training \
    --config "$CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_prefix_value" \
    --log-dir "$LOG_ROOT"

run_stage_non_blocking pvm_gate \
    python scripts/check_pvm_evidence_gate.py \
    --config "$CONFIG" \
    --output "$RESULT_ROOT/pvm_gate_seed42.json"

run_stage full_ppo_training \
    python -m ppo.prefix_training \
    --config "$CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_full_ppo" \
    --log-dir "$LOG_ROOT"

run_stage final_eval \
    python -m ppo.prefix_eval \
    --config "$CONFIG" \
    --seed 42 \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --output "$RESULT_ROOT/final_eval_seed42.json" \
    --run-name "${RUN_NAME}_final_eval" \
    --log-dir "$LOG_ROOT"

{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "status=complete"
} | tee -a "$MASTER_LOG"
