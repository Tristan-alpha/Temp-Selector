## ADDED Requirements

### Requirement: build_dataset is a standalone script
`features/build_dataset.py` SHALL be moved to `scripts/build_dataset.py`. The old module `features.build_dataset` SHALL NOT exist.

#### Scenario: Script runs independently
- **WHEN** `python scripts/build_dataset.py --config configs/base.yaml --output datasets/all.jsonl` is executed
- **THEN** the dataset is generated without depending on run_pipeline.sh

### Requirement: split_jsonl is standalone
`scripts/split_jsonl.py` SHALL remain in `scripts/` and SHALL NOT be invoked by run_pipeline.sh by default.

### Requirement: dataset_eval is standalone
`features/dataset_eval.py` SHALL remain importable as `python -m features.dataset_eval`.

## MODIFIED Requirements
<!-- None -->
