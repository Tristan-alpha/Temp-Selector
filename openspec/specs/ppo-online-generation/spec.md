## ADDED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method that generates text and returns pre-computed per-token logprob and hidden state tensors. The method SHALL accept `prompts: List[str]`, `temperatures: List[float]`, `segment_size: int`, `top_k: int`, and `return_hidden: bool`. It SHALL return a list of dicts with keys `token_ids`, `tokens`, `text`, `logprobs`, `hidden_states`, `finish_reason`.

#### Scenario: Generation with logprobs

- **WHEN** `generate_with_features(prompts=["..."]*2, temperatures=[0.7, 0.3], segment_size=512, top_k=4096)` is called
- **THEN** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`

#### Scenario: Generation with hidden states

- **WHEN** `generate_with_features(..., return_hidden=True)` is called and `feature_mode` is `"hidden_states"` or `"all"`
- **THEN** each returned dict SHALL contain `hidden_states` as a `torch.Tensor` of shape `[n_tokens, hidden_dim]`

### Requirement: PPO uses reserve_training_gpu

`train_ppo` SHALL construct `VLLMFeatureExporter` with `reserve_training_gpu=True`, eliminating GPU overlap between vLLM and training.

#### Scenario: Training GPU isolated

- **WHEN** `train_ppo` is called on a machine with N >= 2 GPUs
- **THEN** vLLM SHALL use N-1 GPUs and training SHALL use the last GPU exclusively

### Requirement: No manual TP computation

`train_ppo` SHALL pass `parallel_size` directly to `VLLMFeatureExporter` without pre-resolving it. All TP resolution SHALL happen inside `_resolve_parallel_size`.

## REMOVED Requirements

### Requirement: generate_raw

**Reason**: Replaced by `generate_with_features`, which provides pre-computed tensors instead of raw vLLM objects.

**Migration**: Use `generate_with_features(prompts, temperatures, segment_size, top_k, return_hidden=...)`.

### Requirement: _extract_segment_obs

**Reason**: Manual logprob parsing replaced by inline tensor computation in `generate_with_features`.

**Migration**: Use `generate_with_features(...)["logprobs"]` directly.
