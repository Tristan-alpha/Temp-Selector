## ADDED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method that generates text and returns pre-computed per-token logprob and hidden state tensors. Speculative decode SHALL always be configured. The method SHALL accept `prompts`, `temperatures`, `segment_size`, `top_k`, `return_hidden`, and `n`. It SHALL return a list of dicts with keys `token_ids`, `tokens`, `text`, `all_texts`, `logprobs`, `hidden_states`, `finish_reason`.

#### Scenario: Generation with logprobs

- **WHEN** `generate_with_features(prompts=["..."]*2, temperatures=[0.7, 0.3], segment_size=512, top_k=4096)` is called
- **THEN** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`

#### Scenario: PPO eval uses runner

- **WHEN** `OnlineTemperatureEvaluator` is constructed
- **THEN** it SHALL create a `VLLMFeatureExporter` with `reserve_training_gpu=True`

### Requirement: Shared feature construction helper

`features/segmenter.py` SHALL provide a `build_segment_obs_from_lp` helper that converts `generate_with_features` output into a segment observation vector. Both `ppo/training.py` and `ppo/eval.py` SHALL use this helper.

#### Scenario: Training and eval use the same helper

- **WHEN** constructing segment observations from `generate_with_features` output
- **THEN** both PPO training and PPO eval SHALL call `build_segment_obs_from_lp`

### Requirement: PPO uses reserve_training_gpu

`train_ppo` and `OnlineTemperatureEvaluator` SHALL construct `VLLMFeatureExporter` with `reserve_training_gpu=True`.

### Requirement: No manual TP computation

`train_ppo` and `OnlineTemperatureEvaluator` SHALL pass `parallel_size` directly to `VLLMFeatureExporter` without pre-resolving it.

### Requirement: PPO eval uses runner for prompt rendering

`OnlineTemperatureEvaluator` SHALL use `runner.build_math_messages()` and `runner.render_messages()` instead of its own `_render_prompt` method.

## REMOVED Requirements

### Requirement: generate_raw

**Reason**: Replaced by `generate_with_features`.

**Migration**: Use `generate_with_features(prompts, temperatures, segment_size, top_k, return_hidden=...)`.

### Requirement: _extract_segment_obs

**Reason**: Manual logprob parsing replaced by inline tensor computation in `generate_with_features`.

**Migration**: Use `generate_with_features(...)["logprobs"]` directly.
