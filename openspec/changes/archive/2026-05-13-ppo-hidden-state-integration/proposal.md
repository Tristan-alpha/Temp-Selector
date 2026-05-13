## Why

Hidden state extraction via two-pass vLLM prefill (from `hidden-state-extraction`) only works in `build_dataset.py`. PPO online training generates segment-by-segment and needs per-segment feature extraction. The extractor API supports per-segment prefill (feed accumulated prefix, extract only the new segment's token positions). Integrating it into PPO's `_extract_segment_obs` closes the loop: MIL trained with hidden states can warm-start PPO and provide shaping rewards at matching feature dimensions.

## What Changes

- `ppo/training.py` `_extract_segment_obs`: add hidden state path when `feature_mode` is `hidden_states` or `all`
- `train_ppo`: read `feature_mode` from config, pass to extraction; init `VLLMHiddenStateExtractor` if needed
- Per-segment prefill: accumulate `prompt + seg₀ + ... + seg_{k-1}` as prefix, extract hidden states for new segment's tokens, mean-pool → segment observation
- `obs_dim` automatically matches `instance_dim` from config (4096 for hidden states, 64 otherwise)

## Capabilities

### New Capabilities

- `ppo-hidden-state-integration`: PPO online training uses per-segment vLLM prefill to extract hidden states as segment observations

## Impact

- **Code**: `ppo/training.py` only — `_extract_segment_obs` and `train_ppo`
- **Config**: No new keys — uses existing `feature_mode` and `instance_dim`
- **Performance**: Each segment adds one prefill of the accumulated prefix; latency increases linearly with segment count
- **Docs**: ppo/DESIGN.md
