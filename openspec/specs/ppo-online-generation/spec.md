## ADDED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method that generates text and returns per-token logprob and hidden state tensors. The method SHALL use a two-pass approach: Pass 1 generates tokens, Pass 2 extracts features via `extract_from_ids` on the full prompt+generated sequence. Speculative decode SHALL always be configured. The method SHALL accept `prompts`, `temperatures`, `segment_size`, `top_k`, `return_logprobs`, `return_hidden`, and `device`. It SHALL return a list of dicts with keys `token_ids`, `tokens`, `text`, `logprobs`, `hidden_states`, `finish_reason`.

#### Scenario: Generation with two-pass feature extraction

- **WHEN** `generate_with_features(prompts=["..."]*2, temperatures=[0.7, 0.3], segment_size=512, top_k=4096, return_logprobs=True)` is called
- **THEN** the first vLLM call SHALL generate `segment_size` tokens per prompt
- **AND** the second vLLM call SHALL extract features by passing full pre-tokenized sequences to `extract_from_ids`
- **AND** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`, computed from the full prefill hidden states

#### Scenario: PPO eval uses runner

- **WHEN** `OnlineTemperatureEvaluator` is constructed
- **THEN** it SHALL create a `VLLMFeatureExporter` with `reserve_training_gpu=True`

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

### Requirement: Shared feature construction helper

`features/segmenter.py` SHALL provide a `build_segment_obs_from_lp` helper that converts `generate_with_features` output into a segment observation vector. The helper SHALL accept a `segment_mode` parameter. Both `ppo/training.py` and `ppo/eval.py` SHALL use this helper and pass the configured `segment_mode`.

#### Scenario: Training and eval use the same helper

- **WHEN** constructing segment observations from `generate_with_features` output
- **THEN** both PPO training and PPO eval SHALL call `build_segment_obs_from_lp`

### Requirement: PPO uses reserve_training_gpu

`train_ppo` and `OnlineTemperatureEvaluator` SHALL construct `VLLMFeatureExporter` with `reserve_training_gpu=True`.

### Requirement: No manual TP computation

`train_ppo` and `OnlineTemperatureEvaluator` SHALL pass `parallel_size` directly to `VLLMFeatureExporter` without pre-resolving it.

### Requirement: PPO eval uses runner for prompt rendering

`OnlineTemperatureEvaluator` SHALL use `runner.build_math_messages()` and `runner.render_messages()` instead of its own `_render_prompt` method.

### Requirement: load_temperature_labels uses voting_label

`load_temperature_labels` in `features/dataset_eval.py` SHALL read the `voting_label` field (not `label` or `individual_label`) from dataset rows. It SHALL continue to flip the value (returning `1=correct, 0=error`) for PPO consumers that expect this encoding.

#### Scenario: load_temperature_labels reads voting_label

- **WHEN** `load_temperature_labels` processes a JSONL dataset
- **THEN** it SHALL extract `int(row["voting_label"])` for per-temperature label groups
- **AND** it SHALL return `1 - voting_label` so that `1.0` means correct (matching PPO's `ep_correct` convention)

#### Scenario: PPO pi head init uses voting accuracy

- **WHEN** `train_ppo` initializes the policy head's temperature bias
- **THEN** it SHALL use per-temperature accuracy derived from `voting_label` (majority-vote correctness)
- **AND** the result SHALL be the same as before (same data, different field name)

### Requirement: PPO supports concat pooling mode

`ppo/training.py` and `ppo/eval.py` SHALL read `data.segment_pooling` from config. When `segment_pooling == "concat"`, `obs_dim` SHALL be `instance_dim * segment_size` and `build_segment_obs_from_lp` SHALL receive `pooling_mode="concat"`.

#### Scenario: PPO training with concat pooling

- **WHEN** `data.segment_pooling` is `"concat"` in config
- **THEN** `train_ppo` SHALL set `obs_dim = instance_dim * segment_size`
- **AND** `build_segment_obs_from_lp` SHALL be called with `pooling_mode="concat"`

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

## REMOVED Requirements

### Requirement: generate_raw

**Reason**: Replaced by `generate_with_features`.

**Migration**: Use `generate_with_features(prompts, temperatures, segment_size, top_k, return_hidden=...)`.

### Requirement: _extract_segment_obs

**Reason**: Manual logprob parsing replaced by inline tensor computation in `generate_with_features`.

**Migration**: Use `generate_with_features(...)["logprobs"]` directly.
