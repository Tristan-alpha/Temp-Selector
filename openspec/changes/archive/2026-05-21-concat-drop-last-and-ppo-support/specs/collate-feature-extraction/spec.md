## MODIFIED Requirements

### Requirement: build_segment_obs_from_lp accepts pooling_mode

`build_segment_obs_from_lp` SHALL accept a `pooling_mode: str = "mean"` parameter and SHALL forward it to `segment_pooling(mode=pooling_mode)`. In concat mode, `segment_pooling` SHALL drop segments with fewer than `segment_size` tokens (instead of zero-padding), except when it is the only segment.

#### Scenario: concat drops incomplete last segment

- **WHEN** `pooling_mode == "concat"` and a segment has fewer than `segment_size` tokens
- **THEN** the segment SHALL be skipped (dropped) from the output
- **AND** the output SHALL contain only fully-filled segments
