## MODIFIED Requirements

### Requirement: PPO intermediate reward is attention-based credit assignment

`ppo/training.py` SHALL NOT use `mil_model` output `inst_logit` for intermediate-step shaping rewards. Instead, it SHALL call `mil_model` once per chain on the accumulated full bag of segment observations during PPO batch construction. When MIL attention weights are available, it SHALL distribute the terminal reward (±1) across ALL decision steps proportional to the L1-normalized attention weights: `reward[t] = terminal_reward × attn_weights[t] / attn_weights.sum()`. When MIL attention weights are unavailable (`mil_model is None`), it SHALL distribute the terminal reward uniformly: `reward[t] = terminal_reward / n_steps`. The `shaping_coef` hyperparameter SHALL be removed from config and training code. The `mil_model` SHALL be called with `torch.no_grad()`.

#### Scenario: MIL is only called during batch construction

- **WHEN** PPO batch construction runs after a rollout
- **THEN** `mil_model` SHALL be called within the batch construction loop (not the rollout loop)
- **AND** each chain's `mil_model` call SHALL receive all accumulated segment observations as a single bag
- **AND** the call SHALL be wrapped in `torch.no_grad()`

#### Scenario: Terminal reward distributed via attention weights

- **WHEN** `mil_model is not None` and a chain has 3 decision steps with attention weights [0.5, 0.3, 0.2] and terminal_reward = 1.0
- **THEN** rewards SHALL be [0.5, 0.3, 0.2]
- **AND** the sum SHALL equal 1.0

#### Scenario: Terminal reward distributed uniformly when no MIL

- **WHEN** `mil_model is None` and a chain has 4 decision steps with terminal_reward = -1.0
- **THEN** each step SHALL receive `-1.0 / 4 = -0.25`
- **AND** the sum SHALL equal -1.0

#### Scenario: Final step receives weighted reward, not full terminal

- **WHEN** a chain has 3 decision steps with attention weights and terminal_reward = 1.0
- **THEN** the final step SHALL receive `1.0 × attn_weights[2] / attn_weights.sum()`, not `1.0`
