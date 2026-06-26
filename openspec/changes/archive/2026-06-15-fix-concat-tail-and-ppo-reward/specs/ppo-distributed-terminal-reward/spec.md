## ADDED Requirements

### Requirement: PPO distributes terminal reward via MIL attention weights

When a MIL model is loaded, PPO training SHALL distribute the terminal reward (±1) across all decision steps proportional to the L1-normalized MIL attention weights. When no MIL model is loaded, PPO SHALL distribute the terminal reward uniformly across all steps. The `shaping_coef` hyperparameter SHALL be removed from config and training code.

#### Scenario: MIL attention available, terminal reward distribution

- **WHEN** `mil_model is not None` and `attn_weights` has shape `[K]` for a chain with K decision steps
- **THEN** each step `t` SHALL receive `reward[t] = terminal_reward × attn_weights[t] / attn_weights.sum()`
- **AND** the sum of all rewards for that chain SHALL equal `terminal_reward`

#### Scenario: MIL model not loaded

- **WHEN** `mil_model is None`
- **THEN** each step `t` SHALL receive `reward[t] = terminal_reward / n_steps` where `n_steps` is the number of decision steps

#### Scenario: Shaping coefficient removed

- **WHEN** a training config is loaded for PPO training
- **THEN** `shaping_coef` SHALL NOT be read from `ppo.training`
- **AND** the reward construction SHALL NOT reference `shaping_coef`

### Requirement: PPO reward applies to all steps

The terminal reward distribution SHALL apply to ALL decision steps, including the final step. No step SHALL receive the full unweighted `terminal_reward`.

#### Scenario: Final step receives weighted reward

- **WHEN** a chain has 3 decision steps and terminal_reward = 1.0 with attention weights [0.5, 0.3, 0.2]
- **THEN** the final step (t=3) SHALL receive `1.0 × 0.2 = 0.2`, not `1.0`
