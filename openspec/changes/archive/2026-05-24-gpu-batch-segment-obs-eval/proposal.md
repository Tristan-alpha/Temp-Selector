## Why

`build_segment_obs_from_lp` is called per-chain in `_evaluate_strategy_batch`, doing `torch.exp` on [32, 4096] + `torch.cat` + zero-padding on CPU for ~1500 chains per round. Timing shows this consumes 68-101 seconds (13-23% of total round time). Each chain's computation is identical in shape and independent — a textbook GPU batch opportunity on the reserved GPU that vLLM leaves idle.

## What Changes

- `features/segmenter.py`: new `batch_build_segment_obs_from_lp` function that stacks per-chain logprob tensors, runs `exp`/`cat`/`truncate`/`mean-pool` on GPU, returns per-chain CPU tensors
- `ppo/eval.py`: resolve `self.device` (same `cuda:n_gpu-1` pattern as training), replace per-chain `build_segment_obs_from_lp` loop with a single batched GPU call

## Capabilities

### New Capabilities
- `gpu-batch-segment-obs`: GPU-batched computation of per-chain segment observations, replacing the per-chain CPU loop in eval

### Modified Capabilities
<!-- None -->

## Impact

- `features/segmenter.py` — new function `batch_build_segment_obs_from_lp`
- `ppo/eval.py` — `__init__` (add `self.device`), `_evaluate_strategy_batch` (replace per-chain loop with batch call)
