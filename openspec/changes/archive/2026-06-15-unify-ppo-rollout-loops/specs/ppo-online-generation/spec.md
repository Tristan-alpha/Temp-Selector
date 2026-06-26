## MODIFIED Requirements

### Requirement: PPO validation uses fixed held-out set

`train_ppo` SHALL load a fixed validation prompt set from `paths.val_dataset` at training start. After each PPO iteration, it SHALL run a val rollout on this fixed set to compute `val_acc` for early stopping. It SHALL NOT use an 80/20 split of training data for validation. The validation rollout SHALL use the same `generate_with_features` parameters as the training rollout (`return_logprobs=True, return_hidden=hs_needed`). The validation rollout SHALL construct segment observations via `build_segment_obs_from_lp` after each round, and SHALL select temperatures via the policy's argmax (not hardcoded T=0.7 after the first segment). It SHALL call `_decide_temperature` and `_process_generated_features` shared with the training rollout.

#### Scenario: Fixed val set stable across iterations

- **WHEN** PPO training runs for multiple iterations
- **THEN** the same set of validation prompts SHALL be used for `val_acc` computation
- **AND** `val_acc` SHALL only vary due to policy changes, not dataset sampling noise

#### Scenario: Validation rollout uses policy for temperature after first round

- **WHEN** the validation rollout runs round 2+ for an active chain
- **THEN** it SHALL call `_decide_temperature(deterministic=True)` with the segment observation from the previous round
- **AND** the policy's argmax SHALL determine the temperature (not a hardcoded T=0.7)

#### Scenario: Validation rollout builds segment observations

- **WHEN** a chain in the validation rollout has not terminated after generation
- **THEN** it SHALL call `_process_generated_features` to build the next round's segment observation via `build_segment_obs_from_lp`
- **AND** it SHALL store the resulting observation for the next round's temperature decision
