## Context

Current per-token feature vector composition — asymmetric and wasteful:

```
topk_logprobs mode:
  [logp][entropy][                 4096 zeros                    ]  ← only 2 useful dims

hidden_states mode (MIL):
  [            4096 hidden dims            ][      2 zeros      ]  ← missing logp+entropy

hidden_states mode (PPO):
  [logp][entropy][            4096 hidden dims                  ]  ← correct, 4098 dims
```

Goal: both modes fill all 4098 dims with meaningful values, MIL = PPO.

## Goals / Non-Goals

**Goals:**
- topk_logprobs: `[logp, entropy, topk_0...topk_4095]` = 4098
- hidden_states: `[logp, entropy, hidden_0...hidden_4095]` = 4098
- MIL and PPO identical for each mode
- Both modes use `build_segment_obs_from_lp` as the single path

**Non-Goals:**
- Changing `top_k_logprobs` config (4096) or `instance_dim` (4098)

## Decisions

### Decision 1: `include_topk` parameter on `build_segment_obs_from_lp`

**Chosen**: Add `include_topk: bool = False` parameter. When True, `parts = [sampled_logprob, entropy, topk_logprobs]` instead of `[sampled_logprob, entropy]`.

**Alternatives**:
- *Change default behavior (always include topk)* → Rejected. hidden_states mode needs those dims for hidden states, not topk logprobs.

### Decision 2: Remove MIL manual hidden-only path

**Chosen**: `need_logprobs = feature_mode in ("topk_logprobs", "hidden_states")`. Both modes go through `if logprob_tensors is not None` branch. Extra parts differ: `[hidden]` for hidden_states, `None` for topk_logprobs.

### Decision 3: Config-driven include_topk in collate_fn

**Chosen**: MIL collate_fn passes `include_topk=(feature_mode == "topk_logprobs")` to `build_segment_obs_from_lp`. PPO passes `include_topk=(not hs_needed)`.

## Feature vector layout (after fix)

```
topk_logprobs:  [logp][entropy][ topk logprob at bin 0 ][ ... ][ bin 4095 ]
                  1      1                    4096                        = 4098 ✓

hidden_states:   [logp][entropy][ hidden dim 0 ][ hidden dim 1 ][ ... ][ dim 4095 ]
                  1      1                    4096                        = 4098 ✓
```
