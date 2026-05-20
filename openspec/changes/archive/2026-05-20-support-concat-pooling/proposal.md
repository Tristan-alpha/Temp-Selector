## Why

`segment_pooling` already supports `mode="concat"` but the call chain hardcodes `mode="mean"`. To enable concat pooling as an ablation (vs mean), the pooling mode config key must flow through to `segment_pooling`, and `MILModel`'s `input_dim` must account for the concat expansion (`instance_dim × segment_size`).

## What Changes

- `features/segmenter.py` `build_segment_obs_from_lp`: add `pooling_mode` parameter, pass to `segment_pooling`
- `mil/utils.py` `make_collate_fn`: pass `pooling_mode` to `build_segment_obs_from_lp`
- `mil/training.py`: compute `model_input_dim = instance_dim * segment_size` for concat, `instance_dim` for mean
- Config `configs/training/pool_concat.yaml`: already exists (`instance_dim=64, segment_size=64, segment_pooling=concat`)

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: `build_segment_obs_from_lp` now accepts and forwards `pooling_mode` to `segment_pooling`

## Impact

- `features/segmenter.py`: +1 parameter
- `mil/utils.py`: 1 line — pass pooling_mode
- `mil/training.py`: 1 line — model_input_dim calculation
