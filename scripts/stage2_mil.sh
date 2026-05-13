#!/usr/bin/env bash
# Stage 2: MIL model training
# Usage: GPU_DEVICES=6,7 bash scripts/stage2_mil.sh
STAGES=mil exec bash "$(dirname "$0")/run_pipeline.sh"
