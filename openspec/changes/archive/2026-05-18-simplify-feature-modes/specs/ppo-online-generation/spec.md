## MODIFIED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method. The method SHALL NOT accept a `feature_mode` parameter — speculative decode is always configured. PPO training and eval SHALL check `feature_mode` against the two valid values only (`topk_logprobs`, `hidden_states`).

#### Scenario: No feature_mode parameter

- **WHEN** `VLLMFeatureExporter` is constructed
- **THEN** speculative decode SHALL always be configured regardless of `feature_mode`
