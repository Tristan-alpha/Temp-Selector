## ADDED Requirements

### Requirement: Eval respects feature_mode config

`OnlineTemperatureEvaluator` SHALL read `inference.feature_mode` from config and pass matching arguments to `generate_with_features` and `build_segment_obs_from_lp`.

When `feature_mode == "hidden_states"`:
- `generate_with_features` SHALL be called with `return_hidden=True`
- `build_segment_obs_from_lp` SHALL be called with `include_topk=False` and `extra_parts=[f["hidden_states"]]` (when hidden states are not None)

When `feature_mode == "topk_logprobs"`:
- `generate_with_features` SHALL be called without `return_hidden` (defaults to False)
- `build_segment_obs_from_lp` SHALL be called with `include_topk=True` and no `extra_parts`

#### Scenario: topk_logprobs mode (default) produces logprob-based features

- **WHEN** config has `feature_mode: topk_logprobs` (or omits it)
- **AND** `_evaluate_strategy_batch` generates segments
- **THEN** `generate_with_features` SHALL NOT pass `return_hidden=True`
- **AND** `build_segment_obs_from_lp` SHALL receive `include_topk=True` with no `extra_parts`
- **AND** the feature vector SHALL consist of `[sampled_logprob, entropy, topk_logprobs]` truncated to `obs_dim`

#### Scenario: hidden_states mode produces hidden-state-based features

- **WHEN** config has `feature_mode: hidden_states`
- **AND** `_evaluate_strategy_batch` generates segments
- **THEN** `generate_with_features` SHALL pass `return_hidden=True`
- **AND** `build_segment_obs_from_lp` SHALL receive `include_topk=False` and `extra_parts=[hidden_states]`
- **AND** the feature vector SHALL consist of `[sampled_logprob, entropy, hidden_states]` truncated to `obs_dim`

#### Scenario: Feature construction matches training for all modes

- **WHEN** the same config is used for training and eval
- **THEN** the arguments passed to `build_segment_obs_from_lp` in eval SHALL match those in training
- **AND** the same `feature_mode` value SHALL produce identical feature vectors from the same logprobs + hidden states
