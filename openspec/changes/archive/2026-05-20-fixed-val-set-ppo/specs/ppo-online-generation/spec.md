## MODIFIED Requirements

### Requirement: PPO validation uses fixed held-out set

`train_ppo` SHALL load a fixed validation prompt set from `paths.val_dataset` at training start. After each PPO iteration, it SHALL run a val rollout on this fixed set to compute `val_value` for early stopping. It SHALL NOT use an 80/20 split of training data for validation.

#### Scenario: Fixed val set stable across iterations

- **WHEN** PPO training runs for multiple iterations
- **THEN** the same set of validation prompts SHALL be used for `val_value` computation
- **AND** `val_value` SHALL only vary due to policy changes, not dataset sampling noise
