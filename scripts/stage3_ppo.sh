#!/usr/bin/env bash
# Stage 3: online PPO training (vLLM generation + policy update)
# Usage: GPU_DEVICES=6,7 bash scripts/stage3_ppo.sh
STAGES=ppo exec bash "$(dirname "$0")/run_pipeline.sh"
