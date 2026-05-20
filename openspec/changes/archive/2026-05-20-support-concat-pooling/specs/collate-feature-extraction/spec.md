## ADDED Requirements

### Requirement: build_segment_obs_from_lp accepts pooling_mode

`build_segment_obs_from_lp` SHALL accept a `pooling_mode: str = "mean"` parameter and SHALL forward it to `segment_pooling(mode=pooling_mode)`. When `pooling_mode == "concat"`, `segment_pooling` SHALL produce `[n_segments, segment_size * obs_dim]` tensors.

#### Scenario: concat pooling via config

- **WHEN** `data.segment_pooling` is `"concat"` and `make_collate_fn` calls `build_segment_obs_from_lp`
- **THEN** the pooling mode SHALL be forwarded to `segment_pooling(mode="concat")`
- **AND** the resulting instance tensor SHALL have shape `[K, segment_size * instance_dim]`
