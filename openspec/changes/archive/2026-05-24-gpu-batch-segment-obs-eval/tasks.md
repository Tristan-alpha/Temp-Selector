## 1. GPU device resolution in eval

- [x] 1.1 In `OnlineTemperatureEvaluator.__init__`, compute `n_gpu` and `self.device` using the same `cuda:n_gpu-1` pattern as `ppo/training.py:123-124`

## 2. Batched segment obs function

- [x] 2.1 Add `batch_build_segment_obs_from_lp` to `features/segmenter.py` — stacks per-chain logprob tensors, runs `exp`/`cat`/`truncate` on GPU, handles `fixed_window`+`mean` pooling on GPU, falls back to per-chain for `step` mode or CPU device
- [x] 2.2 Add CPU-only unit test for `batch_build_segment_obs_from_lp` to `tests/test_segmenter.py` — verifies output matches per-chain `build_segment_obs_from_lp` for a small batch

## 3. Wire into eval loop

- [x] 3.1 In `_evaluate_strategy_batch`, collect logprob tensors and metadata from all active chains after `generate_with_features`, call `batch_build_segment_obs_from_lp` once, then distribute results back to `segment_obs[i][v]`

## 4. Verification

- [x] 4.1 Run `python -m py_compile features/segmenter.py ppo/eval.py` to verify syntax
- [x] 4.2 Run `python -m pytest tests/ -v` — all existing tests must pass
- [x] 4.3 Run `python -m pytest tests/test_segmenter.py -v -k batch` — new batch tests pass
- [ ] 4.4 Manual: run eval with `base.yaml` — verify `build_obs` timing drops from ~100s to ~1s per round
