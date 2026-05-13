## ADDED Requirements

### Requirement: PPO training stops early on val_value plateau
`ppo/training.py` SHALL track the best `val_value` (validation MSE). If `val_value` has not improved (decreased) for `early_stop_patience` consecutive iterations, training stops. The checkpoint from the iteration with the lowest `val_value` SHALL be saved.

#### Scenario: Early stop triggers
- **WHEN** `val_value` has not decreased for `early_stop_patience` consecutive iterations
- **THEN** training terminates and the best-iteration checkpoint is retained

#### Scenario: Continues while improving
- **WHEN** a new minimum `val_value` is reached
- **THEN** the patience counter resets to 0

### Requirement: PPO config uses max_iterations and early_stop_patience
`ppo.training` SHALL contain `max_iterations` (upper bound, default 200) and `early_stop_patience` (default 10). The old `iterations` key SHALL NOT exist.

#### Scenario: Config keys present
- **WHEN** loading `configs/base.yaml`
- **THEN** `ppo.training.max_iterations` is 200 and `ppo.training.early_stop_patience` is 10

#### Scenario: Old key absent
- **WHEN** loading `configs/base.yaml`
- **THEN** `ppo.training.get("iterations")` returns None

### Requirement: PPO saves best checkpoint
The `ppo/training.py` loop SHALL save a checkpoint each time `val_value` reaches a new minimum, overwriting the previous best. The final saved checkpoint is the best-iteration checkpoint.

#### Scenario: Best checkpoint retained
- **WHEN** training completes (either early stop or max_iterations reached)
- **THEN** the checkpoint on disk is from the iteration with the lowest `val_value`
