#!/usr/bin/env bash
set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

GPU_DEVICES=${GPU_DEVICES:-}
LEGACY_CONFIG=${LEGACY_CONFIG:-configs/training/legacy_concat_500.yaml}
FULL_CONFIG=${FULL_CONFIG:-configs/training/full_prefix_value_500.yaml}
STAGES=${STAGES:-validate,continuations,value,value_eval,full_ppo,evaluate,compare}
RUN_LEGACY=${RUN_LEGACY:-0}
HISTORICAL_LEGACY_ACCURACY=${HISTORICAL_LEGACY_ACCURACY:-0.535}
RUN_NAME=${RUN_NAME:-legacy_full_$(date +"%Y%m%d_%H%M%S")}
LOG_DIR=${LOG_DIR:-logs}

run() {
    if [[ -n "$GPU_DEVICES" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$@"
    else
        "$@"
    fi
}

enabled() {
    [[ ",${STAGES}," == *",$1,"* ]]
}

mkdir -p results "$LOG_DIR"

if enabled validate; then
    python scripts/validate_500_split.py
fi

if enabled legacy; then
    run python -m mil.training --config "$LEGACY_CONFIG" --run-name "${RUN_NAME}_legacy_mil" --log-dir "$LOG_DIR"
    run python -m ppo.training --config "$LEGACY_CONFIG" --run-name "${RUN_NAME}_legacy_ppo" --log-dir "$LOG_DIR"
fi

if enabled continuations; then
    run python scripts/build_prefix_continuations.py --config "$FULL_CONFIG" --split train
    run python scripts/build_prefix_continuations.py --config "$FULL_CONFIG" --split val
fi

if enabled value; then
    run python -m mil.value_training --config "$FULL_CONFIG" --run-name "${RUN_NAME}_value" --log-dir "$LOG_DIR"
fi

if enabled value_eval; then
    run python -m mil.value_eval --config "$FULL_CONFIG" --output results/prefix_value_metrics.json
fi

if enabled full_ppo; then
    run python -m ppo.prefix_training --config "$FULL_CONFIG" --run-name "${RUN_NAME}_full_ppo" --log-dir "$LOG_DIR"
fi

if enabled evaluate; then
    for seed in 42 43 44; do
        if [[ "$RUN_LEGACY" == "1" ]]; then
            run python -m ppo.eval --config "$LEGACY_CONFIG" --seed "$seed" \
                --ppo-only \
                --output "results/legacy_seed${seed}.json" \
                --run-name "${RUN_NAME}_legacy_eval_${seed}" --log-dir "$LOG_DIR"
        fi
        run python -m ppo.prefix_eval --config "$FULL_CONFIG" --seed "$seed" \
            --output "results/full_seed${seed}.json" \
            --run-name "${RUN_NAME}_full_eval_${seed}" --log-dir "$LOG_DIR"
    done
fi

if enabled compare; then
    if [[ "$RUN_LEGACY" == "1" ]]; then
        python scripts/compare_legacy_full.py \
            --legacy results/legacy_seed42.json results/legacy_seed43.json results/legacy_seed44.json \
            --full results/full_seed42.json results/full_seed43.json results/full_seed44.json \
            --value-metrics results/prefix_value_metrics.json \
            --output results/legacy_vs_full.json
    else
        python scripts/compare_legacy_full.py \
            --historical-legacy-accuracy "$HISTORICAL_LEGACY_ACCURACY" \
            --full results/full_seed42.json results/full_seed43.json results/full_seed44.json \
            --value-metrics results/prefix_value_metrics.json \
            --output results/historical_legacy_vs_full.json
    fi
fi

echo "comparison_pipeline_done run_name=$RUN_NAME"
