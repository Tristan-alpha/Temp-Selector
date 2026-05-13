## Why

Two problems: (1) MIL and PPO training use hardcoded magic numbers (`epochs: 15`, `iterations: 80`) with no data-driven stopping, and (2) the current train/eval split conflates validation (used for early stopping and best-temperature selection) with testing (used for final evaluation metrics), producing biased final metrics. Adding early stopping requires a proper train/val/test split first.

## What Changes

- **BREAKING**: Split `datasets/eval.jsonl` into `datasets/val.jsonl` and `datasets/test.jsonl`. `scripts/split_jsonl.py` now does train/val/test three-way split (80/10/10)
- **BREAKING**: Rename config paths: `paths.eval_dataset` → `paths.val_dataset` + `paths.test_dataset`
- **BREAKING**: Replace `mil.training.epochs` with `mil.training.max_epochs` + `mil.training.early_stop_patience`
- **BREAKING**: Replace `ppo.training.iterations` with `ppo.training.max_iterations` + `ppo.training.early_stop_patience`
- Add validation step inside MIL training loop on `val.jsonl`, early stop and save best checkpoint
- Add early stop logic to PPO training loop on `val_value`, save best checkpoint
- Best-fixed temperature selection uses `val.jsonl`; final evaluation uses `test.jsonl`
- Update all 7 config files

## Capabilities

### New Capabilities

- `train-val-test-split`: Three-way group-aware data split replacing two-way train/eval
- `mil-early-stop`: Validation-based early stopping for MIL training on `val.jsonl`
- `ppo-early-stop`: Validation-based early stopping for PPO training on `val_value`

## Impact

- **Config**: All 7 config files — path renames + key renames
- **Code**: `scripts/split_jsonl.py` (three-way split), `mil/training.py` (+validation), `ppo/training.py` (+patience), `mil/eval.py` (+test dataset), `ppo/eval.py` (+val dataset), `features/dataset_eval.py` (+CLI dataset selection)
- **Data**: `datasets/eval.jsonl` no longer generated; replaced by `datasets/val.jsonl` + `datasets/test.jsonl`
- **Docs**: PIPELINE.md, mil/DESIGN.md, ppo/DESIGN.md, README.md
