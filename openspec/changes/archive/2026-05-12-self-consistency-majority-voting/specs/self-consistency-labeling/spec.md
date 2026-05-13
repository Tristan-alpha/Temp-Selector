## ADDED Requirements

### Requirement: Stage 1 labels use self-consistency
In `features/build_dataset.py`, the majority label for a (prompt, temperature) group SHALL be determined by: extracting the answer from each of the `num_votes` responses, finding the modal answer (plurality via `collections.Counter.most_common(1)`), and comparing the modal answer to the gold answer via `verify_answer_by_value()`. The label is 0 if the modal answer matches gold, 1 otherwise.

#### Scenario: Mode matches gold
- **WHEN** 4 votes extract answers ["3", "3", "5", "3"] for gold "3"
- **THEN** mode="3", matches gold → label=0 (correct)

#### Scenario: Mode does not match gold
- **WHEN** 4 votes extract answers ["7", "7", "3", "7"] for gold "3"
- **THEN** mode="7", does not match gold → label=1 (error)

#### Scenario: Tie — first most-common wins
- **WHEN** 4 votes extract answers ["3", "3", "5", "5"] for gold "3"
- **THEN** mode is either "3" or "5" (Counter.most_common(1) returns first encountered), label depends on which wins

### Requirement: PPO terminal reward uses self-consistency
In `ppo/training.py`, the terminal reward SHALL be determined by the same self-consistency logic: extract answers from all V completions, find the modal answer, compare to gold. The reward is +1 for correct, -1 for incorrect.

#### Scenario: PPO reward matches new label logic
- **WHEN** a PPO episode ends and majority voting is performed
- **THEN** the correctness check uses `extract_answer` + modal comparison, NOT per-vote `verify_answer` counting

### Requirement: Online evaluation accuracy uses self-consistency
In `ppo/eval.py`, the `OnlineResult` accuracy for each strategy SHALL be computed by the same self-consistency logic.

#### Scenario: Online eval matches new label logic
- **WHEN** running online evaluation with `_evaluate_strategy_batch`
- **THEN** the correctness check for each prompt uses extract-answers + modal comparison

### Requirement: individual_correct remains unchanged
`BagSample.metadata["individual_correct"]` SHALL continue to be computed via per-vote `verify_answer()` against gold, unchanged from current behavior.

#### Scenario: individual_correct still per-vote
- **WHEN** reading BagSample construction in build_dataset.py
- **THEN** `metadata["individual_correct"]` is set via `verify_answer(prediction=ex["response"], gold=gold_answer)` for each individual vote

### Requirement: Dataset statistics remain unchanged
`features/dataset_eval.py` SHALL continue to use per-vote `individual_correct` values for its auxiliary statistics (`individual_accuracy`, `per_temperature_breakdown`), unchanged from current behavior.

#### Scenario: dataset_eval reads historical individual_correct
- **WHEN** `evaluate_dataset` processes a JSONL file
- **THEN** it reads `metadata["individual_correct"]` from each row without any logic change
