## 1. Update main() to iterate over all splits

- [x] 1.1 Change `main()` default behavior: loop over `train_dataset`, `val_dataset`, `test_dataset` from config, call `evaluate_dataset()` for each, print per-split summary
- [x] 1.2 Keep `--data` override: when provided, run single-split mode (backward compatible)
- [x] 1.3 Write one output JSON per split: `eval_stats_train.json`, `eval_stats_val.json`, `eval_stats_test.json`

## 2. Verification

- [x] 2.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 2.2 Run `python -m compileall -q features/dataset_eval.py`
- [x] 2.3 Run `python features/dataset_eval.py --config configs/training/base.yaml --output /tmp/eval_stats.json` — verify it runs without error (skip if no dataset files exist)
