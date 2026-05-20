## ADDED Requirements

### Requirement: dataset_eval.py evaluates all splits

`features/dataset_eval.py` `main()` SHALL evaluate `train_dataset`, `val_dataset`, and `test_dataset` from config by default. The `--data` flag SHALL override to single-split mode.

#### Scenario: Default multi-split evaluation

- **WHEN** `python features/dataset_eval.py --config configs/training/base.yaml` is run without `--data`
- **THEN** all three splits SHALL be analyzed in order
- **AND** per-split results SHALL be printed and saved to `eval_stats_{split}.json`
