## ADDED Requirements

### Requirement: Three-way group-aware split
`scripts/split_jsonl.py` SHALL accept `--val-ratio` and `--test-ratio` and produce three output files: train, val, test. All splits SHALL be group-aware (same prompt's variants go to the same split).

#### Scenario: 80/10/10 split
- **WHEN** split_jsonl is invoked with `--val-ratio 0.1 --test-ratio 0.1`
- **THEN** ~80% of groups go to train, ~10% to val, ~10% to test

#### Scenario: Backward compatible defaults
- **WHEN** split_jsonl is invoked with `--config configs/base.yaml`
- **THEN** val-ratio defaults to 0.1 and test-ratio defaults to 0.1

### Requirement: Config paths reflect three-way split
`paths` SHALL contain `val_dataset` and `test_dataset`. The old `eval_dataset` key SHALL NOT exist.

#### Scenario: val and test paths present
- **WHEN** loading config
- **THEN** `paths.val_dataset` and `paths.test_dataset` exist

#### Scenario: eval_dataset absent
- **WHEN** loading config
- **THEN** `paths.get("eval_dataset")` returns None

### Requirement: Consumers use correct split
- `mil/training.py` early stopping SHALL read `paths.val_dataset`
- `mil/eval.py` final evaluation SHALL read `paths.test_dataset`
- `ppo/training.py` / `ppo/eval.py` best-fixed temperature selection SHALL read `paths.val_dataset`
- `features/dataset_eval.py` SHALL support `--data` override for both val and test
- `run_pipeline.sh` SHALL pass `--data "$VAL_DATASET"` or `--data "$TEST_DATASET"` as appropriate

#### Scenario: No leakage between val and test
- **WHEN** training runs
- **THEN** the val dataset is never used for final evaluation and the test dataset is never used for early stopping
