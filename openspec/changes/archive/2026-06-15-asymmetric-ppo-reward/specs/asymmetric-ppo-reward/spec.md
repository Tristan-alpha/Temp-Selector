## ADDED Requirements

### Requirement: PPO reward is asymmetric between correct and incorrect chains

When MIL attention weights are available, PPO training SHALL use attention-weighted distribution for incorrect chains and uniform distribution for correct chains. When MIL is unavailable, both SHALL use uniform distribution.

#### Scenario: Incorrect chain uses attention weights

- **WHEN** `terminal_reward = -1.0` and MIL `attn_weights = [0.5, 0.3, 0.2]` for a chain with 3 steps
- **THEN** `weights = attn_weights / attn_weights.sum()` (L1-normalized)
- **AND** `reward[t] = -1.0 × weights[t]`
- **AND** the sum SHALL equal `-1.0`

#### Scenario: Correct chain uses uniform weights

- **WHEN** `terminal_reward = +1.0` and a chain has 4 steps
- **THEN** `reward[t] = +1.0 / 4 = 0.25` for all steps
- **AND** the sum SHALL equal `+1.0`

#### Scenario: MIL unavailable, both use uniform

- **WHEN** `mil_model is None`, regardless of `terminal_reward` sign
- **THEN** `reward[t] = terminal_reward / n_steps` for all steps
