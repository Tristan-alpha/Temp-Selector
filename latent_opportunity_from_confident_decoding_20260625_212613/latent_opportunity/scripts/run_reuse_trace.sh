#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/data/nas_hdd/dazhou/Confident-Decoding}"
CONFIG="${CONFIG:-$ROOT/latent_opportunity/configs/latent_opportunity_reuse_trace.yaml}"
RUN_DIR="${RUN_DIR:-$ROOT/latent_opportunity/runs/reuse_trace_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-latent_opp_reuse_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/latent_opportunity/tmux_logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${SESSION}.log}"
LIMIT_PREFIXES="${LIMIT_PREFIXES:-0}"
PREFIX_SCORE_SOURCE="${PREFIX_SCORE_SOURCE:-config}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${LATENT_OPP_CUDA_VISIBLE_DEVICES:-0}}"

if [[ ",$CUDA_VISIBLE_DEVICES," =~ ,(1|2|3|4), ]]; then
  echo "refusing to run with forbidden CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; do not use GPUs 1,2,3,4" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [[ "${LATENT_OPPORTUNITY_IN_TMUX:-0}" != "1" ]]; then
  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && LATENT_OPPORTUNITY_IN_TMUX=1 CONFIG='$CONFIG' RUN_DIR='$RUN_DIR' LIMIT_PREFIXES='$LIMIT_PREFIXES' PREFIX_SCORE_SOURCE='$PREFIX_SCORE_SOURCE' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' bash latent_opportunity/scripts/run_reuse_trace.sh > '$LOG_FILE' 2>&1"
  sleep 1
  PANE_PID="$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)"
  echo "started latent-opportunity reuse-trace run"
  echo "tmux_session=$SESSION"
  echo "pane_pid=$PANE_PID"
  echo "run_dir=$RUN_DIR"
  echo "log_file=$LOG_FILE"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "progress_cmd=tail -f '$LOG_FILE'"
  echo "reattach_cmd=tmux attach -t '$SESSION'"
  exit 0
fi

cd "$ROOT"
export PYTHONUNBUFFERED=1

CMD=(
  python latent_opportunity/scripts/analyze_reuse_trace.py
  --config "$CONFIG"
  --run-dir "$RUN_DIR"
  --prefix-score-source "$PREFIX_SCORE_SOURCE"
)
if [[ "$LIMIT_PREFIXES" != "0" ]]; then
  CMD+=(--limit-prefixes "$LIMIT_PREFIXES")
fi

echo "started_at=$(date --iso-8601=seconds)"
echo "root=$ROOT"
echo "config=$CONFIG"
echo "run_dir=$RUN_DIR"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "command=${CMD[*]}"
"${CMD[@]}"
echo "finished_at=$(date --iso-8601=seconds)"
