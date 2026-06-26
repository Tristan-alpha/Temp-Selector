## ADDED Requirements

### Requirement: batch concat fast path output matches segment_pooling concat

The `batch_build_segment_obs_from_lp` concat fast path SHALL produce output tensors of shape `[1, segment_size × obs_dim]` that are element-wise identical to those produced by per-chain `segment_pooling(mode="concat", segment_size=segment_size)` for the same input. The fast path SHALL NOT depend on `max_tok` for its output dimension.

#### Scenario: max_tok less than segment_size

- **WHEN** all chains in the batch have `n_tok < segment_size`
- **THEN** the concat fast path output SHALL have shape `[1, segment_size × obs_dim]`
- **AND** the first `n_tok × obs_dim` elements SHALL contain real features and the remainder SHALL be zeros

#### Scenario: max_tok greater than segment_size

- **WHEN** any chain in the batch has `n_tok > segment_size`
- **THEN** the concat fast path output SHALL have shape `[1, segment_size × obs_dim]`
- **AND** only the first `segment_size` tokens SHALL be used per chain

#### Scenario: Batch concat fast path matches per-chain output

- **WHEN** `batch_build_segment_obs_from_lp` concat fast path and per-chain `build_segment_obs_from_lp` are given the same inputs
- **THEN** their output tensors SHALL be element-wise identical
