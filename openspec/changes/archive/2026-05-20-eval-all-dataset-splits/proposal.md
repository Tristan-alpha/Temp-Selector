## Why

`features/dataset_eval.py` currently only analyzes the test split. Running it three times with different `--data` overrides to check train/val/test is tedious. The config already has all three paths — `main()` should just iterate over them.

## What Changes

- `features/dataset_eval.py` `main()`: Loop over `train_dataset`, `val_dataset`, `test_dataset` from config, call `evaluate_dataset()` for each, print a combined summary. The `--data` flag still overrides to a single split for ad-hoc use.
- Output: `eval_stats.json` becomes `eval_stats_train.json`, `eval_stats_val.json`, `eval_stats_test.json` (one per split).

## Capabilities

### Modified Capabilities

- `data-label-schema`: `evaluate_dataset` usage now covers all three splits. No schema change to the function itself.

## Impact

- `features/dataset_eval.py`: `main()` only — loop over splits
