## MODIFIED Requirements

### Requirement: PPO intermediate reward is attention-based credit assignment

`ppo/training.py` SHALL NOT use `mil_model` output `inst_logit` for intermediate-step shaping rewards. Instead, it SHALL call `mil_model` once per chain on the accumulated full bag of segment observations during PPO batch construction. For **incorrect** chains (`terminal_reward < 0`), it SHALL distribute the terminal reward across all decision steps proportional to L1-normalized MIL attention weights. For **correct** chains (`terminal_reward > 0`), it SHALL distribute the terminal reward uniformly across all steps. When MIL attention weights are unavailable (`mil_model is None`), both SHALL use uniform distribution. The `shaping_coef` hyperparameter SHALL NOT exist. The `mil_model` SHALL be called with `torch.no_grad()`.

#### Scenario: MIL is only called during batch construction

- **WHEN** PPO batch construction runs after a rollout
- **THEN** `mil_model` SHALL be called within the batch construction loop (not the rollout loop)
- **AND** each chain's `mil_model` call SHALL receive all accumulated segment observations as a single bag
- **AND** the call SHALL be wrapped in `torch.no_grad()`

#### Scenario: Incorrect chain uses attention weights

- **WHEN** `terminal_reward = -1.0` and MIL `attn_weights = [0.5, 0.3, 0.2]` for a chain with 3 steps
- **THEN** rewards SHALL be `[-0.5, -0.3, -0.2]`
- **AND** the sum SHALL equal `-1.0`

#### Scenario: Correct chain uses uniform weights

- **WHEN** `terminal_reward = +1.0` and a chain has 4 steps
- **THEN** each step SHALL receive `+1.0 / 4 = 0.25`
- **AND** the sum SHALL equal `+1.0`

#### Scenario: MIL unavailable, both uniform

- **WHEN** `mil_model is None` and a chain has 4 steps with `terminal_reward = -1.0`
- **THEN** each step SHALL receive `-1.0 / 4 = -0.25`
- **AND** the sum SHALL equal `-1.0`
