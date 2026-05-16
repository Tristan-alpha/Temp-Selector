## Why

`token_to_vec`, `token_to_obs`, `mean_pool_obs`, and `segment_pooling` all operate on `List[List[float]]` with Python nested loops. For 4096-dim vectors across hundreds of tokens, this is a CPU bottleneck. The same mean-pool logic is duplicated 4 times across the codebase. Switching to tensor operations (single `torch.stack`, `torch.mean`) eliminates the Python overhead and unifies the duplicated implementations.

## What Changes

- **BREAKING**: `token_to_vec` returns `torch.Tensor [obs_dim]` instead of `List[float]`
- **BREAKING**: `token_to_obs` returns `torch.Tensor [obs_dim]` instead of `List[float]`
- **BREAKING**: `mean_pool_obs` accepts `List[torch.Tensor]` and returns `torch.Tensor [obs_dim]`
- **BREAKING**: `segment_pooling` accepts `torch.Tensor [n_tokens, obs_dim]` and returns `torch.Tensor [n_segments, obs_dim]`
- Replace duplicate manual mean-pool loops in `_extract_segment_obs` / `_extract_segment_obs_sglang` with `mean_pool_obs`
- Remove `_patch_features` `.tolist()` — store 1D tensor views directly
- Hidden state mean-pool in PPO uses `torch.tensor(hs).mean(dim=0)`

## Capabilities

### New Capabilities

- `vectorized-pooling`: Token vectorization and segment pooling SHALL use PyTorch tensor operations throughout, using `torch.stack` and `.mean(dim=0)` instead of Python nested loops

### Modified Capabilities

- (none — this is a pure internal implementation change)

## Impact

- `features/vectorizer.py` — return types changed to `torch.Tensor`
- `features/segmenter.py` — input/output types changed to `torch.Tensor`, mean mode uses `.mean(dim=0)`
- `mil/training.py` — collate_fn: `torch.stack` tokens before `segment_pooling`, remove `.tolist()` in `_patch_features`
- `ppo/training.py` — two `_extract_*` functions: use `mean_pool_obs` instead of manual loops
- `ppo/eval.py` — `_extract_segment_obs`: trivial type adaptation
- `tests/test_vectorizer.py` — compare tensors
- `tests/test_segmenter.py` — tensor inputs
- `tests/test_mil_training.py` — collate tests fine as-is
