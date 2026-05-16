## 1. Vectorize core functions

- [x] 1.1 `token_to_vec`: return `torch.Tensor [obs_dim]`, add `extracted` parameter for inline tensor consumption
- [x] 1.2 `token_to_obs`: return `torch.Tensor [obs_dim]`
- [x] 1.3 `mean_pool_obs`: accept `List[torch.Tensor]`, return `torch.Tensor` via `stack.mean(dim=0)`
- [x] 1.4 `segment_pooling`: accept `torch.Tensor [n_tokens, obs_dim]`, return `torch.Tensor [n_segments, obs_dim]`, mean via `.mean(dim=0)`

## 2. Update MIL collate_fn — delete _patch_features

- [x] 2.1 Delete `_patch_features` helper
- [x] 2.2 collate_fn: inline per-token loop passes extracted tensors to `token_to_vec` via `extracted` param
- [x] 2.3 collate_fn: `torch.stack(token_vecs)` → `segment_pooling`, drop `torch.tensor(inst_vecs)`

## 3. Update PPO training

- [x] 3.1 `_extract_segment_obs`: replace manual mean-pool with `mean_pool_obs`; hidden mean-pool with `torch.tensor(hs).mean(dim=0)`
- [x] 3.2 `_extract_segment_obs_sglang`: same changes
- [x] 3.3 Call sites: adapt to tensor return types

## 4. Update PPO eval

- [x] 4.1 `_extract_segment_obs`: adapt to tensor return types from `token_to_obs` / `mean_pool_obs`

## 5. Update tests

- [x] 5.1 `test_vectorizer.py`: add tests for `extracted` param, all assert via tensor comparison
- [x] 5.2 `test_segmenter.py`: tensor inputs, tensor output comparisons
- [x] 5.3 `test_mil_training.py`: collate tests should still pass

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests must pass (128 passed)
- [x] 6.2 Run `python -m compileall -q <modified_files>` — syntax check passed
- [x] 6.3 Update `CLAUDE.md` — remove `.tolist()` pitfall, add `extracted` param convention if needed
