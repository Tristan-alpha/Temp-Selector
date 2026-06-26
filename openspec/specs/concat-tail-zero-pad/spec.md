## ADDED Requirements

### Requirement: segment_pooling concat zero-pads short segments

`segment_pooling` in concat mode SHALL zero-pad segments with fewer than `segment_size` tokens to `segment_size × obs_dim`, rather than dropping them. Segments with `chunk.shape[0] >= segment_size` SHALL be truncated to the first `segment_size` tokens before reshaping.

#### Scenario: Last segment shorter than segment_size

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has 2 tokens
- **THEN** the output tensor for that segment SHALL have shape `[segment_size × obs_dim]`
- **AND** the first `2 × obs_dim` elements SHALL contain the real token features
- **AND** the remaining `62 × obs_dim` elements SHALL be zeros

#### Scenario: Segment exactly matches segment_size

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has exactly 64 tokens
- **THEN** the output tensor for that segment SHALL have shape `[segment_size × obs_dim]`
- **AND** all elements SHALL contain real token features

#### Scenario: Segment exceeds segment_size

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has 100 tokens
- **THEN** only the first 64 tokens SHALL be used
- **AND** the output tensor SHALL have shape `[segment_size × obs_dim]`
