## Why

Two issues with MIL/PPO feature vector consistency:

1. **MIL hidden_states mode** discards logprobs entirely (uses only hidden states), while PPO includes `[logprob, entropy, hidden]`. MIL→PPO warm-start sees different feature layouts.

2. **topk_logprobs mode** wastes 4096 feature dimensions on zeros. The top-k logprobs retrieved from vLLM are used only to compute a single entropy scalar, then discarded. They should be included directly in the feature vector.

## What Changes

- `features/segmenter.py` `build_segment_obs_from_lp`: Add `include_topk: bool = False` parameter. When True, append the full top-k logprobs (`lp_tensor[:, 1:]`) to the feature vector instead of discarding them after entropy computation.
- `mil/utils.py` `make_collate_fn`: Always extract logprobs in both modes. Remove the manual `elif hidden_tensors` branch. Pass `include_topk=True` for topk_logprobs mode, `False` for hidden_states mode.
- `ppo/training.py`: Pass `include_topk=True` when `hs_needed=False`.
- Result: both modes produce exactly 4098-dim features — topk mode uses `[logp, entropy, topk_0..topk_4095]`, hidden mode uses `[logp, entropy, hidden_0..hidden_4095]`.

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: `build_segment_obs_from_lp` gains `include_topk` parameter. MIL hidden_states mode always extracts logprobs and combines via the same function. Feature vector layout is now consistent between MIL and PPO for both modes.

## Impact

- `features/segmenter.py`: `build_segment_obs_from_lp` — add `include_topk` parameter, restructure `parts` construction
- `mil/utils.py`: ~15 lines changed
- `ppo/training.py`: 1 line changed (`include_topk=True`)
