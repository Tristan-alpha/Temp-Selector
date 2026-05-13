#!/usr/bin/env bash
# Stage 1: feature extraction + dataset construction + train/eval split
# Usage: GPU_DEVICES=6,7 RAW_INPUT=data/math-small.jsonl bash scripts/stage1_build.sh
STAGES=build,split exec bash "$(dirname "$0")/run_pipeline.sh"
