#!/usr/bin/env bash
set -euo pipefail

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

RUN_STAMP=${RUN_STAMP:-20260624_121747}
RUN_NAME=${RUN_NAME:-min_pvm_8x32_${RUN_STAMP}}

Q_CONFIG=${Q_CONFIG:-configs/training/min_pvm_q_500_seed42_8x32_${RUN_STAMP}.yaml}
VALUE_CONFIG=${VALUE_CONFIG:-configs/training/min_pvm_ppo_500_seed42_8x32_${RUN_STAMP}.yaml}
SOURCE_DIR=${SOURCE_DIR:-datasets/min_pvm_ppo_500_seed42_20260618}
DATA_DIR=${DATA_DIR:-datasets/min_pvm_ppo_500_seed42_8x32_${RUN_STAMP}}
LOG_ROOT=${LOG_ROOT:-tmux_logs/min_pvm_8x32_${RUN_STAMP}}
RESULT_ROOT=${RESULT_ROOT:-results/min_pvm_8x32_${RUN_STAMP}}

LABEL_GPU_A=${LABEL_GPU_A:-1}
LABEL_GPU_B=${LABEL_GPU_B:-2}
LABEL_PARALLEL_SIZE=${LABEL_PARALLEL_SIZE:-1}
RECORDS_PER_BATCH=${RECORDS_PER_BATCH:-4}
TARGET_SEEDS_PER_TEMPERATURE=${TARGET_SEEDS_PER_TEMPERATURE:-32}
APPEND_SEED_OFFSET=${APPEND_SEED_OFFSET:-10000000}
TRAIN_RECORD_MID=${TRAIN_RECORD_MID:-2097}
TRAIN_RECORD_END=${TRAIN_RECORD_END:-4194}
VAL_RECORD_MID=${VAL_RECORD_MID:-260}
VAL_RECORD_END=${VAL_RECORD_END:-520}

Q_TRAIN_GPU_DEVICES=${Q_TRAIN_GPU_DEVICES:-1}
Q_EVAL_GPU_DEVICES=${Q_EVAL_GPU_DEVICES:-1}
Q_EVAL_GPU_MEMORY_UTILIZATION=${Q_EVAL_GPU_MEMORY_UTILIZATION:-0.75}

VALUE_TRAIN_GPU_DEVICES=${VALUE_TRAIN_GPU_DEVICES:-2}
PPO_GPU_DEVICES=${PPO_GPU_DEVICES:-2,3}
PPO_EVAL_GPU_DEVICES=${PPO_EVAL_GPU_DEVICES:-2,3}
TRAIN_PARALLEL_SIZE=${TRAIN_PARALLEL_SIZE:-1}
PPO_EVAL_GPU_MEMORY_UTILIZATION=${PPO_EVAL_GPU_MEMORY_UTILIZATION:-0.75}

mkdir -p "$DATA_DIR" "$LOG_ROOT" "$RESULT_ROOT"
MASTER_LOG="$LOG_ROOT/${RUN_NAME}.master.log"

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
    CUDA_VISIBLE_DEVICES="$devices" "$@" 2>&1 | tee "$stage_log"
    {
        echo "===== ${name} END $(date --iso-8601=seconds) ====="
        echo "stage_log=${stage_log}"
    } | tee -a "$MASTER_LOG"
}

run_extension_shards() {
    local split="$1"
    local existing="$2"
    local record_mid="$3"
    local record_end="$4"
    local shard_dir="$DATA_DIR/shards"
    mkdir -p "$shard_dir"
    local shard0="$shard_dir/prefix_continuations_${split}.part0.jsonl"
    local shard1="$shard_dir/prefix_continuations_${split}.part1.jsonl"
    local log0="$LOG_ROOT/${RUN_NAME}.${split}_continuation_extension.part0.log"
    local log1="$LOG_ROOT/${RUN_NAME}.${split}_continuation_extension.part1.log"
    {
        echo "===== ${split}_continuation_extension_shards START $(date --iso-8601=seconds) ====="
        echo "gpu_a=${LABEL_GPU_A} records=0:${record_mid} output=${shard0}"
        echo "gpu_b=${LABEL_GPU_B} records=${record_mid}:${record_end} output=${shard1}"
    } | tee -a "$MASTER_LOG"
    CUDA_VISIBLE_DEVICES="$LABEL_GPU_A" python scripts/extend_prefix_continuations.py \
        --config "$VALUE_CONFIG" \
        --split "$split" \
        --existing "$existing" \
        --output "$shard0" \
        --target-seeds-per-temperature "$TARGET_SEEDS_PER_TEMPERATURE" \
        --append-seed-offset "$APPEND_SEED_OFFSET" \
        --parallel-size "$LABEL_PARALLEL_SIZE" \
        --record-start 0 \
        --record-end "$record_mid" \
        --records-per-batch "$RECORDS_PER_BATCH" \
        --resume \
        --save-generated-text > "$log0" 2>&1 &
    local pid0=$!
    CUDA_VISIBLE_DEVICES="$LABEL_GPU_B" python scripts/extend_prefix_continuations.py \
        --config "$VALUE_CONFIG" \
        --split "$split" \
        --existing "$existing" \
        --output "$shard1" \
        --target-seeds-per-temperature "$TARGET_SEEDS_PER_TEMPERATURE" \
        --append-seed-offset "$APPEND_SEED_OFFSET" \
        --parallel-size "$LABEL_PARALLEL_SIZE" \
        --record-start "$record_mid" \
        --record-end "$record_end" \
        --records-per-batch "$RECORDS_PER_BATCH" \
        --resume \
        --save-generated-text > "$log1" 2>&1 &
    local pid1=$!
    local status0=0
    local status1=0
    wait "$pid0" || status0=$?
    wait "$pid1" || status1=$?
    {
        echo "part0_log=${log0} status=${status0}"
        echo "part1_log=${log1} status=${status1}"
        echo "===== ${split}_continuation_extension_shards END $(date --iso-8601=seconds) ====="
    } | tee -a "$MASTER_LOG"
    if [[ "$status0" -ne 0 || "$status1" -ne 0 ]]; then
        return 1
    fi
}

{
    echo "run_name=${RUN_NAME}"
    echo "run_stamp=${RUN_STAMP}"
    echo "q_config=${Q_CONFIG}"
    echo "value_config=${VALUE_CONFIG}"
    echo "source_dir=${SOURCE_DIR}"
    echo "data_dir=${DATA_DIR}"
    echo "log_root=${LOG_ROOT}"
    echo "result_root=${RESULT_ROOT}"
    echo "label_gpu_a=${LABEL_GPU_A}"
    echo "label_gpu_b=${LABEL_GPU_B}"
    echo "label_parallel_size=${LABEL_PARALLEL_SIZE}"
    echo "train_record_mid=${TRAIN_RECORD_MID}"
    echo "train_record_end=${TRAIN_RECORD_END}"
    echo "val_record_mid=${VAL_RECORD_MID}"
    echo "val_record_end=${VAL_RECORD_END}"
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

train_existing = Path(source_dir) / "prefix_continuations_train.jsonl"
val_existing = Path(source_dir) / "prefix_continuations_val.jsonl"
print("existing_train", check_existing(train_existing))
print("existing_val", check_existing(val_existing))

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

run_extension_shards \
    train \
    "$SOURCE_DIR/prefix_continuations_train.jsonl" \
    "$TRAIN_RECORD_MID" \
    "$TRAIN_RECORD_END"

if [[ "${HANDOFF_AFTER_TRAIN_SHARDS:-1}" == "1" ]]; then
    {
        echo "handoff_after_train_shards=1"
        echo "handoff_script=scripts/run_min_pvm_8x32_parallel_scheduler.sh"
        echo "handoff_at=$(date --iso-8601=seconds)"
        echo "status=handoff_after_train_shards"
    } | tee -a "$MASTER_LOG"
    exit 0
fi

run_stage merge_train_continuations python - "$DATA_DIR" train "$RESULT_ROOT/train_merge_validation.json" <<'PY'
import json
import sys
from pathlib import Path

data_dir, split, validation_path = sys.argv[1:4]
shard_dir = Path(data_dir) / "shards"
shards = [
    shard_dir / f"prefix_continuations_{split}.part0.jsonl",
    shard_dir / f"prefix_continuations_{split}.part1.jsonl",
]
rows = []
seen = set()
for shard in shards:
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
    "shards": [str(path) for path in shards],
    "n_prefixes": len(rows),
    "n_continuations": sum(len(row.get("continuations", [])) for row in rows),
}
meta_path = output.with_suffix(".meta.json")
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
Path(validation_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

run_extension_shards \
    val \
    "$SOURCE_DIR/prefix_continuations_val.jsonl" \
    "$VAL_RECORD_MID" \
    "$VAL_RECORD_END"

run_stage merge_val_continuations python - "$DATA_DIR" val "$RESULT_ROOT/val_merge_validation.json" <<'PY'
import json
import sys
from pathlib import Path

data_dir, split, validation_path = sys.argv[1:4]
shard_dir = Path(data_dir) / "shards"
shards = [
    shard_dir / f"prefix_continuations_{split}.part0.jsonl",
    shard_dir / f"prefix_continuations_{split}.part1.jsonl",
]
rows = []
seen = set()
for shard in shards:
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
    "shards": [str(path) for path in shards],
    "n_prefixes": len(rows),
    "n_continuations": sum(len(row.get("continuations", [])) for row in rows),
}
meta_path = output.with_suffix(".meta.json")
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
Path(validation_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

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
        for item in new_record.get("continuations", []):
            seed_index = int(item["seed_index"])
            grouped[int(item["temperature_index"])].add(seed_index)
            if bool(new_record.get("generation", {}).get("save_generated_text")) and seed_index >= 4:
                if "generated_text" not in item or "full_response_text" not in item:
                    errors.append(f"{split}:{idx} seed_index={seed_index} missing generated text")
        for temp_idx in range(len(temperatures)):
            if grouped.get(temp_idx, set()) != expected_seeds:
                errors.append(f"{split}:{idx} temp_idx={temp_idx} seed coverage failed")
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

run_stage_gpu prefix_q_training "$Q_TRAIN_GPU_DEVICES" \
    python -m mil.value_training \
    --config "$Q_CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_prefix_q" \
    --log-dir "$LOG_ROOT"

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
result = {
    "checkpoint": cfg["paths"]["prefix_value_ckpt"],
    "has_q_head": any(key.startswith("q_head.") for key in state),
    "n_total_distribution": metrics.get("n_total_distribution"),
}
result["passed"] = result["has_q_head"] and result["n_total_distribution"] == {"256": metrics.get("n_total_distribution", {}).get("256")}
Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["has_q_head"] or set((metrics.get("n_total_distribution") or {}).keys()) != {"256"}:
    raise SystemExit(1)
PY

run_stage_gpu q_selector_seed42_eval "$Q_EVAL_GPU_DEVICES" \
    python scripts/eval_q_selector.py \
    --config "$Q_CONFIG" \
    --seed 42 \
    --parallel-size 1 \
    --gpu-memory-utilization "$Q_EVAL_GPU_MEMORY_UTILIZATION" \
    --output "$RESULT_ROOT/q_selector_seed42.json"

run_stage_gpu prefix_value_training "$VALUE_TRAIN_GPU_DEVICES" \
    python -m mil.value_training \
    --config "$VALUE_CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_prefix_value" \
    --log-dir "$LOG_ROOT"

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
result = {
    "checkpoint": cfg["paths"]["prefix_value_ckpt"],
    "has_q_head": any(key.startswith("q_head.") for key in state),
    "n_total_distribution": metrics.get("n_total_distribution"),
}
Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if result["has_q_head"] or set((metrics.get("n_total_distribution") or {}).keys()) != {"256"}:
    raise SystemExit(1)
PY

run_stage_gpu full_ppo_training "$PPO_GPU_DEVICES" \
    python -m ppo.prefix_training \
    --config "$VALUE_CONFIG" \
    --parallel-size "$TRAIN_PARALLEL_SIZE" \
    --run-name "${RUN_NAME}_full_ppo" \
    --log-dir "$LOG_ROOT"

run_stage_gpu value_ppo_seed42_eval "$PPO_EVAL_GPU_DEVICES" \
    python scripts/eval_prefix_ppo_online.py \
    --config "$VALUE_CONFIG" \
    --seed 42 \
    --parallel-size 1 \
    --gpu-memory-utilization "$PPO_EVAL_GPU_MEMORY_UTILIZATION" \
    --output "$RESULT_ROOT/value_ppo_seed42.json"

{
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "status=complete"
} | tee -a "$MASTER_LOG"
