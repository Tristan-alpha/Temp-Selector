## Why

Three gaps in concat pooling support:

1. **Last incomplete segment is zero-padded** — introduces noise. Drop it instead, matching PPO behavior (PPO already skips final segments < segment_size).
2. **PPO training/eval don't pass `pooling_mode`** — they always use `mean`. MIL and PPO must use the same pooling mode for warm-start to work.
3. **PPO `obs_dim` doesn't account for concat** — `obs_dim = instance_dim` is wrong for concat; should be `segment_size * instance_dim`.

## What Changes

- `features/segmenter.py` `segment_pooling`: in concat mode, skip segments with fewer than `segment_size` tokens instead of zero-padding
- `ppo/training.py`: pass `pooling_mode` from config to `build_segment_obs_from_lp`; compute `obs_dim` correctly for concat
- `ppo/eval.py`: same — pass `pooling_mode`, correct `obs_dim`

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: `segment_pooling` concat mode drops incomplete segments
- `ppo-online-generation`: PPO supports concat pooling mode via config

- **Bug fix**: `extract_from_ids` returns hidden tensors on CPU while logprobs are on CUDA → crash in hidden_states mode. Move hidden to requested device.

## Impact

- `features/segmenter.py`: ~5 line change in `segment_pooling` concat branch
- `ppo/training.py`: ~4 line change
- `ppo/eval.py`: ~4 line change
- `inference/vllm_runner.py`: 1 line fix (hidden device)
