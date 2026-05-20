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

`train_ppo` SHALL load a fixed validation prompt set from `paths.val_dataset` at training start. After each PPO iteration, it SHALL run a val rollout on this fixed set to compute `val_value` for early stopping. It SHALL NOT use an 80/20 split of training data for validation.

#### Scenario: Fixed val set stable across iterations

- **WHEN** PPO training runs for multiple iterations
- **THEN** the same set of validation prompts SHALL be used for `val_value` computation
- **AND** `val_value` SHALL only vary due to policy changes, not dataset sampling noise

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

## REMOVED Requirements

### Requirement: generate_raw

**Reason**: Replaced by `generate_with_features`.

**Migration**: Use `generate_with_features(prompts, temperatures, segment_size, top_k, return_hidden=...)`.

### Requirement: _extract_segment_obs

**Reason**: Manual logprob parsing replaced by inline tensor computation in `generate_with_features`.

**Migration**: Use `generate_with_features(...)["logprobs"]` directly.
