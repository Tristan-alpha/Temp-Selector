#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TF_MIL_ROOT="$(cd -- "${EXPERIMENT_ROOT}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_ROOT}/outputs/segment_latent_paths}"
MAX_PREFIXES="${MAX_PREFIXES:-0}"
SEEDS_PER_TEMPERATURE="${SEEDS_PER_TEMPERATURE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
PREFIX_BATCH_SIZE="${PREFIX_BATCH_SIZE:-16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-64}"
PARALLEL_SIZE="${PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
VLLM_MICRO_BATCH_SIZE="${VLLM_MICRO_BATCH_SIZE:-64}"

mkdir -p "${OUTPUT_DIR}"

common_args=(
  --max-prefixes "${MAX_PREFIXES}"
  --seeds-per-temperature "${SEEDS_PER_TEMPERATURE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --prefix-batch-size "${PREFIX_BATCH_SIZE}"
  --score-batch-size "${SCORE_BATCH_SIZE}"
  --parallel-size "${PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --vllm-micro-batch-size "${VLLM_MICRO_BATCH_SIZE}"
  --output "${OUTPUT_DIR}/segment_candidate_records.jsonl"
)

cd "${TF_MIL_ROOT}"

python "${EXPERIMENT_ROOT}/scripts/generate_segment_candidates.py" \
  --stage generate \
  "${common_args[@]}" \
  --manifest "${OUTPUT_DIR}/generation_manifest.json"

python "${EXPERIMENT_ROOT}/scripts/generate_segment_candidates.py" \
  --stage score \
  "${common_args[@]}" \
  --manifest "${OUTPUT_DIR}/scoring_manifest.json"

python "${EXPERIMENT_ROOT}/scripts/analyze_segment_latent_paths.py" \
  --candidates "${OUTPUT_DIR}/segment_candidate_records.jsonl" \
  --output-dir "${OUTPUT_DIR}"

