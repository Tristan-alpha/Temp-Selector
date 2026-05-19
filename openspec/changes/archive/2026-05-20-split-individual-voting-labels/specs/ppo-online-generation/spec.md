## ADDED Requirements

### Requirement: load_temperature_labels uses voting_label

`load_temperature_labels` in `features/dataset_eval.py` SHALL read the `voting_label` field (not `label` or `individual_label`) from dataset rows. It SHALL continue to flip the value (returning `1=correct, 0=error`) for PPO consumers that expect this encoding.

#### Scenario: load_temperature_labels reads voting_label

- **WHEN** `load_temperature_labels` processes a JSONL dataset
- **THEN** it SHALL extract `int(row["voting_label"])` for per-temperature label groups
- **AND** it SHALL return `1 - voting_label` so that `1.0` means correct (matching PPO's `ep_correct` convention)

#### Scenario: PPO pi head init uses voting accuracy

- **WHEN** `train_ppo` initializes the policy head's temperature bias
- **THEN** it SHALL use per-temperature accuracy derived from `voting_label` (majority-vote correctness)
- **AND** the result SHALL be the same as before (same data, different field name)
