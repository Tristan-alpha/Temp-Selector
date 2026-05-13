## Why

`build`, `split`, `eval_ds` are one-time data preparation steps that should not be re-run on every experiment. Keeping them in the pipeline forces every run to redo vLLM generation (expensive) and re-split data (unnecessary since all configs now share the same dataset). Moving them out makes the pipeline lean, lets the user control vLLM generation parameters independently, and saves `dataset_eval` results as persistent JSON files.

## What Changes

- Move `features/build_dataset.py` → `scripts/build_dataset.py`
- Remove `build`, `split`, `eval_ds` from `run_pipeline.sh` default STAGES
- `dataset_eval` (`features/dataset_eval.py`) saves results to `datasets/eval_stats.json` by default (in addition to logging)
- Pipeline defaults to `STAGES=mil,eval,ppo,eval_ol`
- Document one-time data prep commands in README/PIPELINE

## Capabilities

### New Capabilities

- `data-prep-scripts`: Three standalone scripts for one-time data preparation
- `lean-pipeline`: Pipeline only contains iterative training/evaluation stages
- `eval-output-file`: `dataset_eval` writes results to a persistent JSON file

## Impact

- **Code**: `features/build_dataset.py` moved; `features/dataset_eval.py` adds JSON output; `run_pipeline.sh` simplified
- **No config changes**
- **Docs**: README, PIPELINE.md, CLAUDE.md
