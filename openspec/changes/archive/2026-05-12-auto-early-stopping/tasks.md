## 1. Train/val/test split

- [x] 1.1 Update `scripts/split_jsonl.py`: three-way split with `--val-ratio` and `--test-ratio`, produce train/val/test outputs
- [x] 1.2 Update `scripts/run_pipeline.sh`: `split` stage invokes three-way split, sets `VAL_DATASET` and `TEST_DATASET` env vars; `eval_ds` stage uses `--data "$TEST_DATASET"`; `mil.eval` uses `--data "$TEST_DATASET"`

## 2. Config paths and keys

- [x] 2.1 Update `configs/base.yaml`: replace `eval_dataset` → `val_dataset` + `test_dataset`; replace `epochs` → `max_epochs: 50` + `mil.training.early_stop_patience: 5`; replace `iterations` → `max_iterations: 200` + `ppo.training.early_stop_patience: 10`
- [x] 2.2 Update all 6 ablation configs with same changes

## 3. MIL early stopping

- [x] 3.1 In `mil/training.py`, add validation DataLoader from `paths.val_dataset`
- [x] 3.2 Add `compute_bag_accuracy()` helper and run after each epoch
- [x] 3.3 Add patience tracking; save checkpoint only when bag_accuracy improves
- [x] 3.4 Read `max_epochs` and `early_stop_patience` from config

## 4. PPO early stopping

- [x] 4.1 In `ppo/training.py`, read `max_iterations` and `early_stop_patience` from config
- [x] 4.2 Add patience tracking based on `val_value`; save checkpoint only when val_value improves

## 5. Data consumers use correct split

- [x] 5.1 Update `mil/eval.py`: read `paths.test_dataset` for final evaluation
- [x] 5.2 Update `ppo/training.py` / `ppo/eval.py`: best-fixed temp `load_temperature_labels` reads `paths.val_dataset`
- [x] 5.3 Update `features/dataset_eval.py` CLI: `--data` defaults to `paths.test_dataset`

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 6.2 Run `python -m compileall -q` on modified files
- [x] 6.3 Verify all 7 configs parse

## 7. Documentation

- [x] 7.1 Update PIPELINE.md: split section, training sections, env vars table
- [x] 7.2 Update mil/DESIGN.md training loop section
- [x] 7.3 Update ppo/DESIGN.md training loop section
- [x] 7.4 Update README.md directory structure and commands
