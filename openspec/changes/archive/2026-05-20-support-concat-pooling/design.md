## Context

`segment_pooling(mode="concat")` already exists and works correctly. The gap is that callers hardcode `mode="mean"` and the MIL model's `input_dim` doesn't account for concat expansion.

## Goals / Non-Goals

**Goals:**
- `segment_pooling: concat` in config produces correctly-dimensioned segment features
- `MILModel` receives the correct `input_dim` (`instance_dim × segment_size` for concat)
- Pre-computation cache works unchanged (concat features fit in same RAM budget when dims match)

**Non-Goals:**
- Adding projection layers
- Changing `segment_pooling` concat logic itself

## Decisions

### Decision 1: Pass `pooling_mode` through the call chain

**Chosen**: `build_segment_obs_from_lp(..., pooling_mode="mean")` → `segment_pooling(..., mode=pooling_mode)`. Callers pass the config value.

### Decision 2: Compute model_input_dim in training.py

**Chosen**: `model_input_dim = instance_dim * segment_size if pooling_mode == "concat" else instance_dim`. The model sees the expanded dimension directly.

**Alternatives**:
- *Put logic in build_segment_obs_from_lp* → Rejected. Function shouldn't know about model architecture.

## Feature dimension layout

```
mean mode:   instance_dim=4096, segment_size=256
             → [n_segments, 4096]

concat mode: instance_dim=64, segment_size=64
             → [n_segments, 64×64=4096]
             each token contributes 64 dims, 64 tokens concat
```
