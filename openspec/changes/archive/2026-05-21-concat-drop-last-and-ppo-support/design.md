## Context

concat pooling requires:
- All segments have exactly `segment_size` tokens (no padding noise)
- MIL and PPO use the same pooling mode (warm-start dimension match)
- obs_dim = segment_size × instance_dim for both MIL and PPO

## Goals / Non-Goals

**Goals:**
- concat mode drops incomplete last segment
- PPO reads `pooling_mode` from config and passes through
- PPO computes correct `obs_dim` for concat

**Non-Goals:**
- Adding concat-specific logic to PolicyValueNet or MILModel
- Changing mean mode behavior

## Decisions

### Decision 1: Drop incomplete segment unconditionally in concat mode

**Chosen**: `if chunk.shape[0] < segment_size: continue`. This drops the last segment when it's shorter than `segment_size`.

**Alternatives**:
- *Protect sole segment from being dropped* → Unnecessary. Min tokens per response is 102, with segment_size=64 there are always ≥2 segments.

### Decision 2: PPO reads `segment_pooling` from config

**Chosen**: `pooling_mode = cfg["data"].get("segment_pooling", "mean")` in both training and eval, passed to `build_segment_obs_from_lp`.

### Decision 3: PPO obs_dim calculation mirrors MIL

**Chosen**: `obs_dim = instance_dim * segment_size if pooling_mode == "concat" else instance_dim`. Same formula as `mil/training.py`.
