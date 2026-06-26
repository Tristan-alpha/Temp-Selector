## MODIFIED Requirements

### Requirement: segment_pooling accepts a tensor

`segment_pooling` SHALL accept `token_tensor: torch.Tensor` of shape `[n_tokens, obs_dim]` and SHALL return a `torch.Tensor` of shape `[n_segments, obs_dim]`. In mean mode, per-segment aggregation SHALL use `chunk.mean(dim=0)`. In concat mode, per-segment aggregation SHALL zero-pad segments shorter than `segment_size` to `segment_size × obs_dim` and SHALL truncate segments exceeding `segment_size` to the first `segment_size` tokens.

#### Scenario: Empty token tensor

- **WHEN** an empty tensor is passed
- **THEN** a `[1, obs_dim]` zero tensor is returned

#### Scenario: Concat mode zero-pads short segment

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has 2 tokens
- **THEN** the output tensor for that segment SHALL have shape `[segment_size × obs_dim]`
- **AND** the first `2 × obs_dim` elements SHALL be real features and the remainder SHALL be zeros

#### Scenario: Concat mode truncates long segment

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has 100 tokens
- **THEN** only the first 64 tokens SHALL be used
- **AND** the output tensor SHALL have shape `[segment_size × obs_dim]`
