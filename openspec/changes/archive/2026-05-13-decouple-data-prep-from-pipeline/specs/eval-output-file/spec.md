## ADDED Requirements

### Requirement: dataset_eval saves results to a JSON file
`features/dataset_eval.py` main() SHALL write the evaluation result dict to `datasets/eval_stats.json` (or a `--output` CLI override) in addition to logging.

#### Scenario: Default output path
- **WHEN** `python -m features.dataset_eval --config configs/base.yaml` is executed
- **THEN** `datasets/eval_stats.json` is created containing the full result dict

#### Scenario: Custom output path
- **WHEN** `python -m features.dataset_eval --config configs/base.yaml --output datasets/my_stats.json` is executed
- **THEN** `datasets/my_stats.json` is created

#### Scenario: Pipeline still works
- **WHEN** `eval_ds` stage in `run_pipeline.sh` is explicitly invoked
- **THEN** `--output "$EVAL_STATS_FILE"` is passed, defaulting to `datasets/eval_stats.json`
