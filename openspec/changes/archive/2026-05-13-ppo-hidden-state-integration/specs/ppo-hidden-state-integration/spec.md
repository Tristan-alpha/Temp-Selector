## ADDED Requirements

### Requirement: PPO extracts per-segment hidden states
`_extract_segment_obs` in `ppo/training.py` SHALL, when `feature_mode` is `hidden_states` or `all`, call `VLLMHiddenStateExtractor.extract()` with the accumulated `prompt + previous_segments` as prompt and the new segment text as response. The returned hidden states SHALL be mean-pooled to produce the segment observation vector.

#### Scenario: hidden_states mode uses prefill
- **WHEN** `feature_mode` is `hidden_states` and a new segment is generated
- **THEN** the extractor pre-fills `prompt + seg₀ + ... + seg_k` and returns hidden states for `seg_k`'s token positions

#### Scenario: all mode includes hidden states
- **WHEN** `feature_mode` is `all`
- **THEN** segment observation includes both logprob/topk_logits features AND mean-pooled hidden states

### Requirement: Per-segment prefix accumulation
`train_ppo` SHALL maintain an accumulated text prefix per episode. Each round the new segment text is appended. The prefix SHALL be passed to the extractor as the prompt argument.

#### Scenario: Prefix grows across segments
- **WHEN** three segments have been generated: "seg₀", "seg₁", "seg₂"
- **THEN** the extractor receives `prompt + "seg₀"` for round 1, `prompt + "seg₀seg₁"` for round 2, etc.

### Requirement: No new config keys
The integration SHALL NOT introduce new config keys. `feature_mode` and `instance_dim` from existing config determine hidden state usage and observation dimensionality.

#### Scenario: hidden_states config just works
- **WHEN** loading a config with `feature_mode: hidden_states` and `instance_dim: 4096`
- **THEN** PPO training uses the hidden state extraction path without additional configuration
