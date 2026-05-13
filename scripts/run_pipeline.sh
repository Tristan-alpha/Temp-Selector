#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PIPELINE_ENV_FILE:-}" && -f "${PIPELINE_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${PIPELINE_ENV_FILE}"
fi

# ===================================================================
# Environment variables (all overridable)
# ===================================================================
CONFIG=${CONFIG:-configs/base.yaml}
DATASET_CONFIG=${DATASET_CONFIG:-configs/dataset.yaml}
RAW_INPUT=${RAW_INPUT:-data/prompts.jsonl}
ALL_DATASET=${ALL_DATASET:-datasets/all.jsonl}
TRAIN_DATASET=${TRAIN_DATASET:-datasets/train.jsonl}
VAL_DATASET=${VAL_DATASET:-datasets/val.jsonl}
TEST_DATASET=${TEST_DATASET:-datasets/test.jsonl}
MIL_CKPT=${MIL_CKPT:-checkpoints/mil_ckpt.pt}
PPO_CKPT=${PPO_CKPT:-checkpoints/ppo_ckpt.pt}
BACKEND=${BACKEND:-vllm}
LOG_DIR=${LOG_DIR:-logs}
RUN_NAME=${RUN_NAME:-exp_$(date +"%Y%m%d_%H%M%S")}
VAL_RATIO=${VAL_RATIO:-0.1}
TEST_RATIO=${TEST_RATIO:-0.1}

GPU_DEVICES=${GPU_DEVICES:-}
ENABLE_STARTUP_SELF_CHECK=${ENABLE_STARTUP_SELF_CHECK:-1}
ENABLE_ONLINE_EVAL=${ENABLE_ONLINE_EVAL:-1}

# Default: training + evaluation only.  Data prep (build, split, eval_ds)
# are one-time operations — run them manually before experiments.
STAGES=${STAGES:-mil,eval,ppo,eval_ol}

# ===================================================================
# Helpers
# ===================================================================

stage_enabled() {
    local name="$1"
    [[ ",${STAGES}," == *",${name},"* ]]
}

run_stage() {
    local stage_name="$1"; shift
    if [[ -n "$GPU_DEVICES" ]]; then
        echo "[$stage_name] CUDA_VISIBLE_DEVICES=$GPU_DEVICES"
        CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$@"
    else
        echo "[$stage_name] CUDA_VISIBLE_DEVICES=<inherit>"
        "$@"
    fi
}

run_startup_preflight() {
    echo "[SELF-CHECK] preflight_start"

    if [[ -n "$GPU_DEVICES" ]]; then
        if ! [[ "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            echo "[ERROR] GPU_DEVICES must be comma-separated numeric ids, got: '$GPU_DEVICES'" >&2; exit 1
        fi
        if command -v nvidia-smi >/dev/null 2>&1; then
            local total; total=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
            if [[ "$total" -eq 0 ]]; then
                echo "[ERROR] GPU_DEVICES is set but nvidia-smi reports 0 visible GPUs." >&2; exit 1
            fi
            IFS=',' read -r -a ids <<< "$GPU_DEVICES"
            for id in "${ids[@]}"; do
                if [[ "$id" -ge "$total" ]]; then
                    echo "[ERROR] GPU id '$id' out of range. nvidia-smi reports $total GPU(s)." >&2; exit 1
                fi
            done
            echo "[SELF-CHECK] nvidia_smi_gpus=$total requested='$GPU_DEVICES'"
        else
            echo "[WARN] nvidia-smi not found; skipping GPU index validation."
        fi
    fi
    echo "[SELF-CHECK] preflight_ok"
}

# ===================================================================
# Banner
# ===================================================================

if [[ ! -f "$CONFIG" ]]; then echo "[ERROR] CONFIG not found: $CONFIG" >&2; exit 1; fi

mkdir -p "$LOG_DIR"
echo "run_name=$RUN_NAME"
echo "log_file=$LOG_DIR/$RUN_NAME.log"
echo "config=$CONFIG"
echo "dataset_config=$DATASET_CONFIG"
echo "stages=$STAGES"
echo "backend=$BACKEND"
echo "gpu_devices=${GPU_DEVICES:-<inherit>}"
echo "raw_input=$RAW_INPUT"
echo "all_dataset=$ALL_DATASET"
echo "train_dataset=$TRAIN_DATASET"
echo "val_dataset=$VAL_DATASET"
echo "test_dataset=$TEST_DATASET"
echo "mil_ckpt=$MIL_CKPT"
echo "ppo_ckpt=$PPO_CKPT"

# ===================================================================
# Stages
# ===================================================================

if [[ "$ENABLE_STARTUP_SELF_CHECK" == "1" ]]; then
    run_startup_preflight
fi

if stage_enabled "build"; then
    run_stage "build_dataset" \
        python scripts/build_dataset.py --config "$DATASET_CONFIG" --backend "$BACKEND" \
            --run-name "$RUN_NAME" --log-dir "$LOG_DIR"
else
    echo "[skip] build_dataset"
fi

if stage_enabled "split"; then
    run_stage "split_jsonl" \
        python scripts/split_jsonl.py --config "$CONFIG" \
            --val-ratio "$VAL_RATIO" --test-ratio "$TEST_RATIO" --group-by sample_prefix --seed 42
else
    echo "[skip] split_jsonl"
fi

if stage_enabled "eval_ds"; then
    run_stage "dataset_eval" \
        python -m features.dataset_eval --config "$CONFIG" --data "$TEST_DATASET" \
            --run-name "$RUN_NAME" --log-dir "$LOG_DIR"
else
    echo "[skip] dataset_eval"
fi

if stage_enabled "mil"; then
    run_stage "train_mil" \
        python -m mil.training --config "$CONFIG" --run-name "$RUN_NAME" --log-dir "$LOG_DIR"
else
    echo "[skip] train_mil"
fi

if stage_enabled "eval"; then
    run_stage "evaluate_mil" \
        python -m mil.eval --config "$CONFIG" --data "$TEST_DATASET" \
            --run-name "$RUN_NAME" --log-dir "$LOG_DIR"
else
    echo "[skip] evaluate_mil"
fi

if stage_enabled "ppo"; then
    run_stage "train_ppo" \
        python -m ppo.training --config "$CONFIG" \
            --run-name "$RUN_NAME" --log-dir "$LOG_DIR"
else
    echo "[skip] train_ppo"
fi

if stage_enabled "eval_ol"; then
    run_stage "online_evaluate" \
        python -m ppo.eval --config "$CONFIG" \
            --run-name "$RUN_NAME" --log-dir "$LOG_DIR"
else
    echo "[skip] online_evaluate"
fi

echo "pipeline_done run_name=$RUN_NAME"
