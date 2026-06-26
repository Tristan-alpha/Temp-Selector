## ADDED Requirements

### Requirement: Shared temperature decision function

`ppo/training.py` SHALL provide a `_decide_temperature` function that encapsulates temperature selection for a single chain. It SHALL accept `segment_obs` (or `None` for the first segment), `policy`, `temp_bins`, `device`, and `deterministic: bool`. When `deterministic=True`, it SHALL select the temperature via `argmax` of policy logits. When `deterministic=False`, it SHALL select via `sample_action`. Both training and validation rollouts SHALL use this function.

#### Scenario: First segment uses default temperature

- **WHEN** `_decide_temperature` is called with `segment_obs=None`
- **THEN** it SHALL return `temp=0.7`, `action=torch.tensor(0)`, `logp=torch.tensor(0.0)`, `value=torch.tensor(0.0)`

#### Scenario: Deterministic mode uses argmax

- **WHEN** `_decide_temperature` is called with `deterministic=True` and a valid `segment_obs`
- **THEN** it SHALL compute `logits, value = policy(segment_obs[-1:])`
- **AND** it SHALL select `action = logits.argmax(dim=-1)` without sampling
- **AND** it SHALL return `temp_bins[action]` as the temperature

#### Scenario: Stochastic mode uses sample_action

- **WHEN** `_decide_temperature` is called with `deterministic=False` and a valid `segment_obs`
- **THEN** it SHALL compute `logits, value = policy(segment_obs[-1:])`
- **AND** it SHALL select `action, logp = sample_action(logits.squeeze(0))`
- **AND** it SHALL return `temp_bins[action.item()]` as the temperature, along with `action.cpu()`, `logp.cpu()`, `value.squeeze(0).cpu()`

### Requirement: Shared feature processing function

`ppo/training.py` SHALL provide a `_process_generated_features` function that processes the output of `generate_with_features` for a single chain. It SHALL detect chain termination (EOS, stop, or empty tokens). For non-terminated chains, it SHALL call `build_segment_obs_from_lp` to construct the observation for the next round and return it. Both training and validation rollouts SHALL use this function.

#### Scenario: Chain terminates on EOS

- **WHEN** `_process_generated_features` is called and `tokenizer.eos_token_id in new_tokens`
- **THEN** it SHALL return `(text_delta, is_done=True, next_segment_obs=None)`

#### Scenario: Chain terminates on stop reason

- **WHEN** `_process_generated_features` is called and `finish_reason == 'stop'`
- **THEN** it SHALL return `(text_delta, is_done=True, next_segment_obs=None)`

#### Scenario: Chain terminates on empty tokens

- **WHEN** `_process_generated_features` is called and `new_tokens` is empty
- **THEN** it SHALL return `(text_delta, is_done=True, next_segment_obs=None)`

#### Scenario: Chain continues — builds segment observation

- **WHEN** `_process_generated_features` is called and the chain has not terminated
- **THEN** it SHALL call `build_segment_obs_from_lp(feats["logprobs"], feats["tokens"], feats["text"], segment_size, instance_dim, device=device, extra_parts=extra, segment_mode=segment_mode, include_topk=(not hs_needed), pooling_mode=pooling_mode)`
- **AND** it SHALL return `(text_delta, is_done=False, next_segment_obs=obs.cpu())`

### Requirement: Validation rollout uses policy for temperature decisions

The PPO validation rollout SHALL NOT hardcode T=0.7 for all segments. Instead, it SHALL use `_decide_temperature` with `deterministic=True` so that the policy's argmax temperature is selected for every round after the first.

#### Scenario: Validation uses policy after first segment

- **WHEN** the validation rollout reaches round 2+ for a chain
- **THEN** `_decide_temperature` SHALL be called with `deterministic=True` and the segment observation from the previous round
- **AND** the policy's argmax logit SHALL determine the temperature

### Requirement: Validation rollout uses correct generate_with_features flags

The PPO validation rollout SHALL pass `return_logprobs=True` and `return_hidden=hs_needed` to `generate_with_features`, matching the training rollout exactly.

#### Scenario: Validation generate_with_features flags match training

- **WHEN** the validation rollout calls `generate_with_features`
- **THEN** `return_logprobs` SHALL be `True`
- **AND** `return_hidden` SHALL be `hs_needed` (derived from `feature_mode == "hidden_states"`)
