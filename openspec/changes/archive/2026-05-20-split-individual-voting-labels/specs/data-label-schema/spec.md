## ADDED Requirements

### Requirement: Dataset rows SHALL include individual_label field

Every JSONL row SHALL contain a top-level `individual_label` field (int, 0 or 1). `individual_label=0` means the individual response is correct. `individual_label=1` means the individual response contains at least one error. This field SHALL be used by MIL training and evaluation as the bag label.

#### Scenario: individual_label for correct response

- **WHEN** a single response's extracted answer matches the gold answer
- **THEN** `individual_label` SHALL be `0`

#### Scenario: individual_label for incorrect response

- **WHEN** a single response's extracted answer does not match the gold answer
- **THEN** `individual_label` SHALL be `1`

### Requirement: Dataset rows SHALL include voting_label field

Every JSONL row SHALL contain a top-level `voting_label` field (int, 0 or 1). `voting_label=0` means the majority vote for the question is correct. `voting_label=1` means the majority vote is wrong. This field SHALL be used for per-temperature accuracy statistics (e.g., PPO temperature bias initialization).

#### Scenario: voting_label for majority-correct question

- **WHEN** majority voting across all votes for a question is correct
- **THEN** `voting_label` SHALL be `0` for all rows belonging to that question

#### Scenario: voting_label for majority-wrong question

- **WHEN** majority voting across all votes for a question is wrong
- **THEN** `voting_label` SHALL be `1` for all rows belonging to that question

### Requirement: individual_label and voting_label MAY differ

For a given row, `individual_label` and `voting_label` MAY have different values. A row in a majority-correct question (`voting_label=0`) may have `individual_label=1` if its specific response is wrong. A row in a majority-wrong question (`voting_label=1`) may have `individual_label=0` if its specific response is correct.

#### Scenario: Individually wrong response in majority-correct question

- **WHEN** a question has 5 correct votes and 3 wrong votes (majority correct)
- **THEN** the 3 wrong votes SHALL have `individual_label=1` and `voting_label=0`

### Requirement: build_dataset.py SHALL write both label fields

`scripts/build_dataset.py` SHALL write `individual_label` and `voting_label` for every sample row. The `individual_label` SHALL be computed from the individual response's answer correctness. The `voting_label` SHALL be computed from the majority-vote result for the question. The old `label` field SHALL NOT be written.

#### Scenario: Sample row contains both fields

- **WHEN** `build_dataset.py` generates a sample row
- **THEN** the row dict SHALL contain keys `individual_label` and `voting_label`
- **AND** the row dict SHALL NOT contain key `label`
