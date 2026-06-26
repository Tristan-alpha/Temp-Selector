## ADDED Requirements

### Requirement: PPO intermediate reward is attention-based credit assignment

`ppo/training.py` SHALL NOT use `mil_model` output `inst_logit` for intermediate-step shaping rewards. Instead, it SHALL call `mil_model` once per chain on the accumulated full bag of segment observations during PPO batch construction, and SHALL compute intermediate rewards as `shaping_coef × terminal_reward × attention_weight[t]`. The `mil_model` SHALL be called with `torch.no_grad()`.

#### Scenario: MIL is only called during batch construction

- **WHEN** PPO batch construction runs after a rollout
- **THEN** `mil_model` SHALL be called within the batch construction loop (not the rollout loop)
- **AND** each chain's `mil_model` call SHALL receive all accumulated segment observations as a single bag
- **AND** the call SHALL be wrapped in `torch.no_grad()`
