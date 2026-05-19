## MODIFIED Requirements

### Requirement: PPO training uses independent chains

`train_ppo` SHALL treat each of the V generation chains per prompt as an independent episode. Each chain SHALL receive its own temperature from the policy via `sample_action`. Each chain SHALL independently accumulate text, observations, actions, logprobs, and values. The terminal majority-vote reward (±1) for a prompt SHALL be applied to every chain's terminal step. The PPO batch SHALL include trajectories from all chains.

#### Scenario: Independent chain generation

- **WHEN** a prompt has V=8 active chains and a policy decision is needed
- **THEN** each chain SHALL independently call `sample_action(policy(segment_obs[i][v]))` to get its own temperature
- **AND** each chain SHALL generate independently via `generate_with_features`

#### Scenario: Shared reward across chains

- **WHEN** all chains for a prompt have terminated
- **THEN** `self_consistency_correct` SHALL compute majority-vote correctness
- **AND** the same ±1 reward SHALL be applied to every chain's terminal step for that prompt
