#!/usr/bin/env bash
# Stage 4: offline evaluation + online PPO evaluation
# Usage: GPU_DEVICES=6,7 bash scripts/stage4_eval.sh
STAGES=eval,eval_ol exec bash "$(dirname "$0")/run_pipeline.sh"
