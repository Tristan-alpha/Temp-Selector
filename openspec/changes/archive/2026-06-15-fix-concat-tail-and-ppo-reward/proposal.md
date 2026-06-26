## Why

`segment_pooling` concat mode silently drops the last segment when it has fewer tokens than `segment_size`, losing the `\boxed{answer}` portion of reasoning chains. Separately, PPO assigns the entire terminal reward (±1) to the final decision step, ignoring that earlier steps drive correctness. MIL attention weights already score error likelihood per segment — they should be used to distribute terminal reward across all steps, not just as shaping bonuses on intermediate steps.

## What Changes

- **segment_pooling concat zero-padding**: Short tail segments are zero-padded to `segment_size × obs_dim` instead of discarded, preserving answer-segment features for MIL training.
- **PPO distributed terminal reward**: Terminal reward is spread across all decision steps weighted by the MIL attention distribution (L1-normalized), replacing the current shaping-only scheme. The last step no longer receives full ±1; intermediate steps no longer require a separate shaping coefficient.

## Capabilities

### New Capabilities
- `concat-tail-zero-pad`: `segment_pooling` in concat mode zero-pads segments shorter than `segment_size` to preserve all tokens.
- `ppo-distributed-terminal-reward`: PPO terminal reward is distributed across all decision steps proportional to MIL attention weights, removing the shaping coefficient hyperparameter.

### Modified Capabilities
- `vectorized-pooling`: `segment_pooling` concat behavior changes from dropping short segments to zero-padding them.
- `ppo-online-generation`: PPO reward construction changes from shaping-only intermediate steps to attention-weighted distribution of the terminal reward across all steps.

## Impact

- `features/segmenter.py` — `segment_pooling` concat path
- `ppo/training.py` — reward construction loop (lines 348-361)
- `ppo/eval.py` — if eval replicates reward logic (it doesn't; eval only uses terminal reward for metrics)
- Configs: `shaping_coef` key removed from `ppo.training`; MIL attention weights become required when MIL model is loaded
