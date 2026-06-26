#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

RUN_STAMP=${RUN_STAMP:-20260624_121747}
RUN_NAME=${RUN_NAME:-min_pvm_8x32_${RUN_STAMP}_parallel}

Q_CONFIG=${Q_CONFIG:-configs/training/min_pvm_q_500_seed42_8x32_${RUN_STAMP}.yaml}
VALUE_CONFIG=${VALUE_CONFIG:-configs/training/min_pvm_ppo_500_seed42_8x32_${RUN_STAMP}.yaml}
SOURCE_DIR=${SOURCE_DIR:-datasets/min_pvm_ppo_500_seed42_20260618}
DATA_DIR=${DATA_DIR:-datasets/min_pvm_ppo_500_seed42_8x32_${RUN_STAMP}}
LOG_ROOT=${LOG_ROOT:-tmux_logs/min_pvm_8x32_${RUN_STAMP}}
RESULT_ROOT=${RESULT_ROOT:-results/min_pvm_8x32_${RUN_STAMP}}

TRAIN_SHARD_DIR=${TRAIN_SHARD_DIR:-${DATA_DIR}/shards}
VAL_SHARD_DIR=${VAL_SHARD_DIR:-${DATA_DIR}/shards_parallel}

TARGET_SEEDS_PER_TEMPERATURE=${TARGET_SEEDS_PER_TEMPERATURE:-32}
APPEND_SEED_OFFSET=${APPEND_SEED_OFFSET:-10000000}
RECORDS_PER_BATCH=${RECORDS_PER_BATCH:-4}
GPU_WAIT_ENABLED=${GPU_WAIT_ENABLED:-1}
GPU_WAIT_POLL_SECONDS=${GPU_WAIT_POLL_SECONDS:-60}
GPU_IDLE_MAX_MEMORY_MIB=${GPU_IDLE_MAX_MEMORY_MIB:-1024}
GPU_IDLE_MAX_UTILIZATION=${GPU_IDLE_MAX_UTILIZATION:-20}

VAL_GPU_DEVICES=${VAL_GPU_DEVICES:-1,2,3,4}
VAL_PARALLEL_SIZE=${VAL_PARALLEL_SIZE:-1}
VAL_RECORD_END=${VAL_RECORD_END:-520}

Q_TRAIN_GPU_DEVICES=${Q_TRAIN_GPU_DEVICES:-1}
VALUE_TRAIN_GPU_DEVICES=${VALUE_TRAIN_GPU_DEVICES:-3}
Q_EVAL_GPU_DEVICES=${Q_EVAL_GPU_DEVICES:-4}
PPO_GPU_DEVICES=${PPO_GPU_DEVICES:-2}
PPO_EVAL_GPU_DEVICES=${PPO_EVAL_GPU_DEVICES:-2}
TRAIN_PARALLEL_SIZE=${TRAIN_PARALLEL_SIZE:-1}
Q_EVAL_GPU_MEMORY_UTILIZATION=${Q_EVAL_GPU_MEMORY_UTILIZATION:-0.75}
PPO_EVAL_GPU_MEMORY_UTILIZATION=${PPO_EVAL_GPU_MEMORY_UTILIZATION:-0.75}
POLL_SECONDS=${POLL_SECONDS:-15}

mkdir -p "$DATA_DIR" "$LOG_ROOT" "$RESULT_ROOT" "$VAL_SHARD_DIR"
MASTER_LOG="$LOG_ROOT/${RUN_NAME}.master.log"

log_master() {
    echo "$*" | tee -a "$MASTER_LOG"
}

wait_for_devices() {
    local devices="$1"
    if [[ "${GPU_WAIT_ENABLED}" != "1" ]]; then
        return 0
    fi
    python - "$devices" "$GPU_IDLE_MAX_MEMORY_MIB" "$GPU_IDLE_MAX_UTILIZATION" <<'PY'
import subprocess
import sys

devices = [int(part) for part in sys.argv[1].split(",") if part.strip()]
max_mem = int(sys.argv[2])
max_util = int(sys.argv[3])
query = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True)
state = {}
for line in query.splitlines():
    idx, mem, util = [part.strip() for part in line.split(",")]
    state[int(idx)] = (int(mem), int(util))
busy = [
    f"gpu{idx}:mem={state.get(idx, (10**9, 100))[0]}MiB,util={state.get(idx, (10**9, 100))[1]}%"
    for idx in devices
    if state.get(idx, (10**9, 100))[0] > max_mem or state.get(idx, (10**9, 100))[1] > max_util
]
if busy:
    print("; ".join(busy))
    raise SystemExit(1)
PY
}

wait_until_devices_available() {
    local devices="$1"
    local name="$2"
    while true; do
        local status=0
        local details
        details=$(wait_for_devices "$devices" 2>&1) || status=$?
        if [[ "$status" -eq 0 ]]; then
            return 0
        fi
        log_master "===== ${name} WAIT_FOR_GPU $(date --iso-8601=seconds) cuda_visible_devices=${devices} ${details} ====="
        sleep "$GPU_WAIT_POLL_SECONDS"
    done
}

run_stage() {
    local name="$1"
    shift
    local stage_log="$LOG_ROOT/${RUN_NAME}.${name}.log"
    {
        echo "===== ${name} START $(date --iso-8601=seconds) ====="
        echo "command: $*"
    } | tee -a "$MASTER_LOG"
    "$@" 2>&1 | tee "$stage_log"
    {
        echo "===== ${name} END $(date --iso-8601=seconds) ====="
        echo "stage_log=${stage_log}"
    } | tee -a "$MASTER_LOG"
}

run_stage_gpu() {
    local name="$1"
    local devices="$2"
    shift 2
    local stage_log="$LOG_ROOT/${RUN_NAME}.${name}.log"
    {
        echo "===== ${name} START $(date --iso-8601=seconds) ====="
        echo "cuda_visible_devices=${devices}"
        echo "command: $*"
    } | tee -a "$MASTER_LOG"
    wait_until_devices_available "$devices" "$name"
    CUDA_VISIBLE_DEVICES="$devices" "$@" 2>&1 | tee "$stage_log"
    {
        echo "===== ${name} END $(date --iso-8601=seconds) ====="
        echo "stage_log=${stage_log}"
    } | tee -a "$MASTER_LOG"
}

start_stage_gpu() {
    local name="$1"
    local devices="$2"
    shift 2
    local stage_log="$LOG_ROOT/${RUN_NAME}.${name}.log"
    local status_file="$LOG_ROOT/${RUN_NAME}.${name}.status"
    rm -f "$status_file" "${status_file}.tmp"
    wait_until_devices_available "$devices" "$name"
    (
        set +e
        echo "===== ${name} START $(date --iso-8601=seconds) ====="
        echo "cuda_visible_devices=${devices}"
        echo "command: $*"
        CUDA_VISIBLE_DEVICES="$devices" "$@"
        status=$?
        echo "status=${status}" > "${status_file}.tmp"
        mv "${status_file}.tmp" "$status_file"
        echo "===== ${name} END $(date --iso-8601=seconds) status=${status} ====="
        exit "$status"
    ) > "$stage_log" 2>&1 &
    STAGE_PID=$!
    {
        echo "===== ${name} ASYNC $(date --iso-8601=seconds) ====="
        echo "pid=${STAGE_PID}"
        echo "cuda_visible_devices=${devices}"
        echo "stage_log=${stage_log}"
        echo "status_file=${status_file}"
    } | tee -a "$MASTER_LOG"
}

wait_for_stage() {
    local name="$1"
    local pid="$2"
    local status_file="$LOG_ROOT/${RUN_NAME}.${name}.status"
    while [[ ! -f "$status_file" ]]; do
        sleep "$POLL_SECONDS"
    done
    set +e
    wait "$pid"
    local wait_status=$?
    set -e
    local status
    status=$(sed -n 's/^status=//p' "$status_file" | tail -n 1)
    if [[ -z "$status" ]]; then
        status="$wait_status"
    fi
    log_master "===== ${name} WAIT $(date --iso-8601=seconds) status=${status} ====="
    return "$status"
}

{
    echo "run_name=${RUN_NAME}"
    echo "run_stamp=${RUN_STAMP}"
    echo "q_config=${Q_CONFIG}"
    echo "value_config=${VALUE_CONFIG}"
    echo "source_dir=${SOURCE_DIR}"
    echo "data_dir=${DATA_DIR}"
    echo "train_shard_dir=${TRAIN_SHARD_DIR}"
    echo "val_shard_dir=${VAL_SHARD_DIR}"
    echo "log_root=${LOG_ROOT}"
    echo "result_root=${RESULT_ROOT}"
    echo "val_gpu_devices=${VAL_GPU_DEVICES}"
    echo "q_train_gpu_devices=${Q_TRAIN_GPU_DEVICES}"
    echo "value_train_gpu_devices=${VALUE_TRAIN_GPU_DEVICES}"
    echo "q_eval_gpu_devices=${Q_EVAL_GPU_DEVICES}"
    echo "ppo_gpu_devices=${PPO_GPU_DEVICES}"
    echo "ppo_eval_gpu_devices=${PPO_EVAL_GPU_DEVICES}"
    echo "pid=$$"
    echo "started_at=$(date --iso-8601=seconds)"
} | tee "$MASTER_LOG"

run_stage preflight_gpu \
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

run_stage preflight_data python - "$Q_CONFIG" "$VALUE_CONFIG" "$SOURCE_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

q_config, value_config, source_dir = sys.argv[1:4]

with open(value_config, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def check_existing(path):
    rows = load_jsonl(path)
    totals = Counter(int(row.get("n_total", -1)) for row in rows)
    if set(totals) != {32}:
        raise SystemExit(f"{path} expected n_total=32, got {dict(totals)}")
    return len(rows), dict(totals)

print("existing_train", check_existing(Path(source_dir) / "prefix_continuations_train.jsonl"))
print("existing_val", check_existing(Path(source_dir) / "prefix_continuations_val.jsonl"))

def check_cache(dataset_path, cache_path):
    rows = load_jsonl(dataset_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    row_ids = [str(row.get("sample_id", "")) for row in rows]
    cache_ids = [str(row.get("sample_id", "")) for row in cache]
    if row_ids != cache_ids:
        raise SystemExit(f"cache id mismatch: {cache_path}")
    print("cache_ok", cache_path, len(cache))

check_cache(cfg["paths"]["train_dataset"], cfg["paths"]["train_feature_cache"])
check_cache(cfg["paths"]["val_dataset"], cfg["paths"]["val_feature_cache"])
PY

run_stage validate_train_shards python - "$TRAIN_SHARD_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

shard_dir = Path(sys.argv[1])
expected = {
    "prefix_continuations_train.part0.jsonl": 2097,
    "prefix_continuations_train.part1.jsonl": 2097,
}
for name, expected_rows in expected.items():
    path = shard_dir / name
    if not path.exists():
        raise SystemExit(f"missing train shard: {path}")
    totals = Counter()
    rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            totals[str(row.get("n_total"))] += 1
    if rows != expected_rows or set(totals) != {"256"}:
        raise SystemExit(f"{path} rows={rows} totals={dict(totals)}")
    print(path, "rows", rows, "totals", dict(totals))
PY

if [[ ! -s "$DATA_DIR/prefix_continuations_train.jsonl" ]]; then
    run_stage merge_train_continuations python - "$DATA_DIR" train "$RESULT_ROOT/train_merge_validation.json" \
        "$TRAIN_SHARD_DIR/prefix_continuations_train.part0.jsonl" \
        "$TRAIN_SHARD_DIR/prefix_continuations_train.part1.jsonl" <<'PY'
import json
import sys
from pathlib import Path

data_dir, split, validation_path, *shard_args = sys.argv[1:]
rows = []
seen = set()
for shard_arg in shard_args:
    shard = Path(shard_arg)
    with shard.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("source_sample_id"), row.get("prefix_token_end"))
            if key in seen:
                raise SystemExit(f"duplicate merged key: {key}")
            seen.add(key)
            rows.append(row)
output = Path(data_dir) / f"prefix_continuations_{split}.jsonl"
with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
meta = {
    "mode": "merge_shards",
    "split": split,
    "output_path": str(output),
    "shards": shard_args,
    "n_prefixes": len(rows),
    "n_continuations": sum(len(row.get("continuations", [])) for row in rows),
}
output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
Path(validation_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY
else
    log_master "merge_train_continuations SKIP existing ${DATA_DIR}/prefix_continuations_train.jsonl"
fi

IFS=',' read -r -a VAL_GPU_ARRAY <<< "$VAL_GPU_DEVICES"
VAL_SHARD_COUNT=${#VAL_GPU_ARRAY[@]}
if [[ "$VAL_SHARD_COUNT" -lt 1 ]]; then
    echo "VAL_GPU_DEVICES must contain at least one GPU" >&2
    exit 1
fi

VAL_SHARDS=()
VAL_PIDS=()
for ((idx = 0; idx < VAL_SHARD_COUNT; idx++)); do
    start=$((idx * VAL_RECORD_END / VAL_SHARD_COUNT))
    end=$(((idx + 1) * VAL_RECORD_END / VAL_SHARD_COUNT))
    output="$VAL_SHARD_DIR/prefix_continuations_val.part${idx}.jsonl"
    VAL_SHARDS+=("$output")
    if [[ -s "$output" ]]; then
        log_master "val_continuation_extension_part${idx} SKIP existing ${output}"
        continue
    fi
    start_stage_gpu "val_continuation_extension_part${idx}" "${VAL_GPU_ARRAY[$idx]}" \
        python scripts/extend_prefix_continuations.py \
        --config "$VALUE_CONFIG" \
        --split val \
        --existing "$SOURCE_DIR/prefix_continuations_val.jsonl" \
        --output "$output" \
        --target-seeds-per-temperature "$TARGET_SEEDS_PER_TEMPERATURE" \
        --append-seed-offset "$APPEND_SEED_OFFSET" \
        --parallel-size "$VAL_PARALLEL_SIZE" \
        --record-start "$start" \
        --record-end "$end" \
        --records-per-batch "$RECORDS_PER_BATCH" \
        --resume \
        --save-generated-text
    VAL_PIDS+=("${STAGE_PID}:val_continuation_extension_part${idx}")
done

val_status=0
for item in "${VAL_PIDS[@]}"; do
    pid="${item%%:*}"
    name="${item#*:}"
    wait_for_stage "$name" "$pid" || val_status=$?
done
if [[ "$val_status" -ne 0 ]]; then
    echo "val continuation extension failed with status ${val_status}" >&2
    exit "$val_status"
fi

if [[ ! -s "$DATA_DIR/prefix_continuations_val.jsonl" ]]; then
    run_stage merge_val_continuations python - "$DATA_DIR" val "$RESULT_ROOT/val_merge_validation.json" "${VAL_SHARDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

data_dir, split, validation_path, *shard_args = sys.argv[1:]
rows = []
seen = set()
for shard_arg in shard_args:
    shard = Path(shard_arg)
    with shard.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("source_sample_id"), row.get("prefix_token_end"))
            if key in seen:
                raise SystemExit(f"duplicate merged key: {key}")
            seen.add(key)
            rows.append(row)
output = Path(data_dir) / f"prefix_continuations_{split}.jsonl"
with output.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
meta = {
    "mode": "merge_shards",
    "split": split,
    "output_path": str(output),
    "shards": shard_args,
    "n_prefixes": len(rows),
    "n_continuations": sum(len(row.get("continuations", [])) for row in rows),
}
output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
Path(validation_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY
else
    log_master "merge_val_continuations SKIP existing ${DATA_DIR}/prefix_continuations_val.jsonl"
fi

run_stage validate_continuations python - "$SOURCE_DIR" "$DATA_DIR" "$RESULT_ROOT/continuation_validation.json" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

source_dir, data_dir, output_path = sys.argv[1:4]
temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
expected_seeds = set(range(32))
expected_total = len(temperatures) * len(expected_seeds)

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def key(item):
    return (
        int(item["temperature_index"]),
        int(item["seed_index"]),
        int(item["generation_seed"]),
        bool(item["correct"]),
        item.get("extracted_answer"),
        int(item.get("generated_tokens", -1)),
        item.get("finish_reason"),
    )

def stat_for(record, temp):
    stats = record.get("per_temperature_stats", {})
    return stats.get(str(temp)) or stats.get(f"{temp:.1f}") or stats.get(temp)

def check_split(split):
    old = load(Path(source_dir) / f"prefix_continuations_{split}.jsonl")
    new = load(Path(data_dir) / f"prefix_continuations_{split}.jsonl")
    errors = []
    if len(old) != len(new):
        errors.append(f"{split}: record count {len(new)} != old {len(old)}")
    n_total_distribution = Counter()
    for idx, (old_record, new_record) in enumerate(zip(old, new)):
        n_total = int(new_record.get("n_total", -1))
        n_total_distribution[str(n_total)] += 1
        if n_total != expected_total:
            errors.append(f"{split}:{idx} n_total={n_total}")
        grouped = defaultdict(set)
        correct_by_temp = Counter()
        total_by_temp = Counter()
        for item in new_record.get("continuations", []):
            temp_idx = int(item["temperature_index"])
            seed_index = int(item["seed_index"])
            grouped[temp_idx].add(seed_index)
            total_by_temp[temp_idx] += 1
            if bool(item.get("correct")):
                correct_by_temp[temp_idx] += 1
            if bool(new_record.get("generation", {}).get("save_generated_text")) and seed_index >= 4:
                if "generated_text" not in item or "full_response_text" not in item:
                    errors.append(f"{split}:{idx} seed_index={seed_index} missing generated text")
        for temp_idx, temp in enumerate(temperatures):
            if grouped.get(temp_idx, set()) != expected_seeds:
                errors.append(f"{split}:{idx} temp_idx={temp_idx} seed coverage failed")
            stat = stat_for(new_record, temp)
            if not stat:
                errors.append(f"{split}:{idx} missing per_temperature_stats for {temp}")
                continue
            if int(stat.get("n_total", -1)) != 32:
                errors.append(f"{split}:{idx} {temp} stat n_total={stat.get('n_total')}")
            if int(stat.get("n_correct", -1)) != int(correct_by_temp[temp_idx]):
                errors.append(f"{split}:{idx} {temp} stat n_correct mismatch")
        old_items = {
            (int(item["temperature_index"]), int(item["seed_index"])): key(item)
            for item in old_record.get("continuations", [])
        }
        new_items = {
            (int(item["temperature_index"]), int(item["seed_index"])): key(item)
            for item in new_record.get("continuations", [])
            if int(item["seed_index"]) < 4
        }
        if old_items != new_items:
            errors.append(f"{split}:{idx} old seed continuations changed")
        if int(new_record.get("n_correct", -1)) != sum(bool(item.get("correct")) for item in new_record.get("continuations", [])):
            errors.append(f"{split}:{idx} n_correct mismatch")
    return {
        "split": split,
        "passed": not errors,
        "n_records": len(new),
        "n_total_distribution": dict(sorted(n_total_distribution.items())),
        "errors": errors[:20],
        "n_errors": len(errors),
    }

result = {
    "expected_n_total": expected_total,
    "train": check_split("train"),
    "val": check_split("val"),
}
result["passed"] = result["train"]["passed"] and result["val"]["passed"]
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
PY

start_stage_gpu prefix_q_training "$Q_TRAIN_GPU_DEVICES" \
    python -m mil.value_training \
    --config "$Q_CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_prefix_q" \
    --log-dir "$LOG_ROOT"
q_train_pid=$STAGE_PID

start_stage_gpu prefix_value_training "$VALUE_TRAIN_GPU_DEVICES" \
    python -m mil.value_training \
    --config "$VALUE_CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_prefix_value" \
    --log-dir "$LOG_ROOT"
value_train_pid=$STAGE_PID

q_train_done=0
value_train_done=0
q_eval_started=0
ppo_started=0
q_eval_pid=
ppo_pid=
failure=0

while [[ "$q_train_done" -eq 0 || "$value_train_done" -eq 0 ]]; do
    if [[ "$q_train_done" -eq 0 && -f "$LOG_ROOT/${RUN_NAME}.prefix_q_training.status" ]]; then
        if wait_for_stage prefix_q_training "$q_train_pid"; then
            run_stage check_q_checkpoint python - "$Q_CONFIG" "$RESULT_ROOT/q_checkpoint_validation.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
import yaml

config_path, output_path = sys.argv[1:3]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
ckpt = torch.load(cfg["paths"]["prefix_value_ckpt"], map_location="cpu", weights_only=False)
state = ckpt.get("prefix_value", {})
metrics = ckpt.get("validation_metrics", {})
dist = {str(k): v for k, v in (metrics.get("n_total_distribution") or {}).items()}
result = {
    "checkpoint": cfg["paths"]["prefix_value_ckpt"],
    "has_q_head": any(key.startswith("q_head.") for key in state),
    "n_total_distribution": dist,
}
result["passed"] = result["has_q_head"] and set(dist) == {"256"}
Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
PY
            start_stage_gpu q_selector_seed42_eval "$Q_EVAL_GPU_DEVICES" \
                python scripts/eval_q_selector.py \
                --config "$Q_CONFIG" \
                --seed 42 \
                --parallel-size 1 \
                --gpu-memory-utilization "$Q_EVAL_GPU_MEMORY_UTILIZATION" \
                --output "$RESULT_ROOT/q_selector_seed42.json"
            q_eval_pid=$STAGE_PID
            q_eval_started=1
        else
            failure=1
        fi
        q_train_done=1
    fi
    if [[ "$value_train_done" -eq 0 && -f "$LOG_ROOT/${RUN_NAME}.prefix_value_training.status" ]]; then
        if wait_for_stage prefix_value_training "$value_train_pid"; then
            run_stage check_value_checkpoint python - "$VALUE_CONFIG" "$RESULT_ROOT/value_checkpoint_validation.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
import yaml

config_path, output_path = sys.argv[1:3]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
ckpt = torch.load(cfg["paths"]["prefix_value_ckpt"], map_location="cpu", weights_only=False)
state = ckpt.get("prefix_value", {})
metrics = ckpt.get("validation_metrics", {})
dist = {str(k): v for k, v in (metrics.get("n_total_distribution") or {}).items()}
result = {
    "checkpoint": cfg["paths"]["prefix_value_ckpt"],
    "has_q_head": any(key.startswith("q_head.") for key in state),
    "n_total_distribution": dist,
}
result["passed"] = (not result["has_q_head"]) and set(dist) == {"256"}
Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise SystemExit(1)
PY
            start_stage_gpu full_ppo_training "$PPO_GPU_DEVICES" \
                python -m ppo.prefix_training \
                --config "$VALUE_CONFIG" \
                --parallel-size "$TRAIN_PARALLEL_SIZE" \
                --run-name "${RUN_NAME}_full_ppo" \
                --log-dir "$LOG_ROOT"
            ppo_pid=$STAGE_PID
            ppo_started=1
        else
            failure=1
        fi
        value_train_done=1
    fi
    if [[ "$q_train_done" -eq 0 || "$value_train_done" -eq 0 ]]; then
        sleep "$POLL_SECONDS"
    fi
done

if [[ "$ppo_started" -eq 1 ]]; then
    if wait_for_stage full_ppo_training "$ppo_pid"; then
        run_stage_gpu value_ppo_seed42_eval "$PPO_EVAL_GPU_DEVICES" \
            python scripts/eval_prefix_ppo_online.py \
            --config "$VALUE_CONFIG" \
            --seed 42 \
            --parallel-size 1 \
            --gpu-memory-utilization "$PPO_EVAL_GPU_MEMORY_UTILIZATION" \
            --output "$RESULT_ROOT/value_ppo_seed42.json"
    else
        failure=1
    fi
fi

if [[ "$q_eval_started" -eq 1 ]]; then
    wait_for_stage q_selector_seed42_eval "$q_eval_pid" || failure=1
fi

if [[ "$failure" -ne 0 ]]; then
    log_master "status=failed"
    exit 1
fi

{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "status=complete"
} | tee -a "$MASTER_LOG"
