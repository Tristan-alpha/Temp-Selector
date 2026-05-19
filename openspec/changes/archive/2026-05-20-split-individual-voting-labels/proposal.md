## Why

MIL is currently trained on majority-voting labels: a response is labeled "correct" if the majority of votes for that question are correct, even if the individual response itself is wrong. This gives the MIL model incorrect supervision — error segments inside individually-wrong-but-majority-correct responses are labeled as "no error." Splitting labels by their actual semantics (individual correctness vs. majority-vote result) fixes this and removes the pervasive "flipped label" confusion throughout the codebase.

## What Changes

- **BREAKING**: Replace `label` field in JSONL with `individual_label` (0=correct, 1=error — per-response correctness) and `voting_label` (0=correct, 1=error — majority vote result)
- MIL training, eval, and collate_fn read `individual_label` instead of `label`
- PPO temperature bias (`load_temperature_labels`) reads `voting_label` instead of `label`
- MIL code uses an explicit `POSITIVE_BAG_VALUE = 1` constant instead of hardcoded `> 0.5` checks, making the positive=error convention clear at the code level
- Remove all "flipped label convention" warnings from CLAUDE.md, DESIGN.md, PIPELINE.md, and README.md
- `features/dataset_eval.py`: `evaluate_dataset` reports both individual and voting accuracy
- Existing `metadata.individual_correct` bool is promoted to the top-level `individual_label` field

## Capabilities

### New Capabilities

- `data-label-schema`: Defines the dataset label field contract — `individual_label` for per-response correctness, `voting_label` for majority-vote result. All pipeline stages read the appropriate field.

### Modified Capabilities

- `mil-online-hidden-extract`: MIL training, eval, and bag dataset now consume `individual_label` (per-response correctness) instead of majority-voting `label`. Positive bag class is an explicit named constant rather than a hardcoded threshold.
- `ppo-online-generation`: `load_temperature_labels` now reads `voting_label` instead of `label` for per-temperature accuracy statistics.
- `collate-feature-extraction`: `make_collate_fn` reads `individual_label` field when constructing the `y` label tensor.

## Impact

- `scripts/build_dataset.py`: Write both `individual_label` and `voting_label`; stop writing `label`
- `mil/utils.py`: collate_fn field name change; default value stays 0
- `mil/training.py`: pos_weight computation, BCE loss branching, validation — all use `individual_label` + named constant
- `mil/eval.py`: Bag metrics, per-segment analysis, dynamic head distribution — all use `individual_label`
- `features/dataset_eval.py`: `evaluate_dataset` reports dual accuracy; `load_temperature_labels` flips `voting_label` (1=correct for PPO consumers)
- `ppo/training.py`, `ppo/eval.py`: No code change (they don't read `label` directly; `load_temperature_labels` handles the field internally)
- Tests: `test_mil_training.py`, `test_metrics.py` — update label field references
- Docs: CLAUDE.md, README.md, PIPELINE.md, mil/DESIGN.md, ppo/DESIGN.md — remove "flipped" warnings
