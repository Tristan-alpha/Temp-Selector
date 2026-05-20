## 1. segment_pooling: drop incomplete segment in concat mode

- [x] 1.1 In concat branch, skip segments with fewer than `segment_size` tokens instead of zero-padding

## 2. PPO training: support concat pooling

- [x] 2.1 Read `pooling_mode` from config; compute `obs_dim = instance_dim * segment_size` for concat
- [x] 2.2 Pass `pooling_mode` to `build_segment_obs_from_lp` call

## 3. PPO eval: support concat pooling

- [x] 3.1 Read `pooling_mode` from config; compute `obs_dim` correctly for concat
- [x] 3.2 Pass `pooling_mode` to `build_segment_obs_from_lp` call

## 4. Fix hidden_states device mismatch in extract_from_ids

- [x] 4.1 Move `resp_hs` to `cat_device` in `extract_from_ids` (line 338) so hidden tensors are on the same device as logprobs

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 5.2 Run `python -m compileall -q features/segmenter.py ppo/training.py ppo/eval.py inference/vllm_runner.py`
- [x] 5.3 Verify concat config: `pool_concat.yaml` works end-to-end with both MIL and PPO stages
