#!/usr/bin/env bash
# Run all 4 greedy benchmark conditions and compare results.
#
# Usage:
#   CONFIG=configs/training/base.yaml DATA=datasets/test.jsonl bash scripts/run_bench.sh
#
# Environment variables:
#   CONFIG        YAML config path (required)
#   DATA          JSONL dataset path (required)
#   SEED          Random seed (default: 42)
#   MAX_SAMPLES   Limit prompts for quick test (default: 0 = all)
#   PARALLEL_SIZE Tensor parallelism size (default: auto-detect)
#   OUTDIR        Output directory (default: results/bench_greedy)

set -euo pipefail

CONFIG="${CONFIG:?must set CONFIG}"
DATA="${DATA:?must set DATA}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PARALLEL="${PARALLEL_SIZE:-}"
OUTDIR="${OUTDIR:-results/bench_greedy}"

mkdir -p "$OUTDIR"

BASELINE_OUT="$OUTDIR/baseline.json"
HIDDEN_OUT="$OUTDIR/hidden_states.json"
SEGMENT_OUT="$OUTDIR/segment.json"
SEGMENT_TOK_OUT="$OUTDIR/segment_token_ids.json"
SEGMENT_FIXED_OUT="$OUTDIR/segment_fixed.json"
COMPARE_OUT="$OUTDIR/comparison.json"

PARALLEL_ARG=""
if [ -n "$PARALLEL" ]; then
    PARALLEL_ARG="--parallel-size $PARALLEL"
fi

MAX_ARG=""
if [ "$MAX_SAMPLES" -gt 0 ]; then
    MAX_ARG="--max-samples $MAX_SAMPLES"
fi

echo "============================================"
echo "  GREEDY BENCHMARK SUITE"
echo "  Config:  $CONFIG"
echo "  Data:    $DATA"
echo "  Seed:    $SEED"
echo "  Max:     ${MAX_SAMPLES:-all}"
echo "  Outdir:  $OUTDIR"
echo "============================================"

# ── Condition 1: Baseline ──
echo ""
echo "[1/4] Running BASELINE (nothing enabled)..."
python scripts/bench_baseline.py \
    --config "$CONFIG" --data "$DATA" --seed "$SEED" \
    $MAX_ARG $PARALLEL_ARG --output "$BASELINE_OUT"

# ── Condition 2: Hidden states only ──
echo ""
echo "[2/4] Running HIDDEN STATES ONLY (speculative decode ON)..."
python scripts/bench_hidden_only.py \
    --config "$CONFIG" --data "$DATA" --seed "$SEED" \
    $MAX_ARG $PARALLEL_ARG --output "$HIDDEN_OUT"

# ── Condition 3: Segment only (text concat) ──
echo ""
echo "[3/4] Running SEGMENT ONLY (text concatenation)..."
python scripts/bench_segment_only.py \
    --config "$CONFIG" --data "$DATA" --seed "$SEED" \
    $MAX_ARG $PARALLEL_ARG --output "$SEGMENT_OUT"

# ── Condition 4: Segment by token IDs ──
echo ""
echo "[4/5] Running SEGMENT BY TOKEN IDs..."
python scripts/bench_segment_token_ids.py \
    --config "$CONFIG" --data "$DATA" --seed "$SEED" \
    $MAX_ARG $PARALLEL_ARG --output "$SEGMENT_TOK_OUT"

# ── Condition 5: Segment by token IDs + block alignment + fixed batch ──
echo ""
echo "[5/5] Running SEGMENT FIXED (block alignment + fixed batch size)..."
python scripts/bench_segment_fixed.py \
    --config "$CONFIG" --data "$DATA" --seed "$SEED" \
    $MAX_ARG $PARALLEL_ARG --output "$SEGMENT_FIXED_OUT"

# ── Compare ──
echo ""
echo "============================================"
echo "  COMPARISON"
echo "============================================"
python scripts/compare_benches.py \
    --baseline "$BASELINE_OUT" \
    --hidden "$HIDDEN_OUT" \
    --segment "$SEGMENT_OUT" \
    --segment-token "$SEGMENT_TOK_OUT" \
    --segment-fixed "$SEGMENT_FIXED_OUT" \
    --output "$COMPARE_OUT"

echo ""
echo "All results in: $OUTDIR/"
