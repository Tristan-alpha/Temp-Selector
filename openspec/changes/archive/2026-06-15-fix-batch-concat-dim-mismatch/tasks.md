## 1. Fix concat fast path

- [x] 1.1 In `features/segmenter.py` `batch_build_segment_obs_from_lp`, modify the concat fast path (`segment_mode == "fixed_window" and pooling_mode == "concat"`) to pad or truncate `tok_vecs` to `segment_size` tokens along dim 1 before `reshape(B, segment_size * obs_dim)`

## 2. Tests

- [x] 2.1 Add `test_batch_build_obs_concat_pads_short` — verify concat fast path zero-pads when `max_tok < segment_size`, output shape is `[1, segment_size * obs_dim]`
- [x] 2.2 Add `test_batch_build_obs_concat_output_dim_always_segment_size_times_obs_dim` — verify per-segment dim is always `segment_size * obs_dim`
- [x] 2.3 Add `test_batch_build_obs_concat_matches_per_chain_mixed_tokens` — verify batch output matches per-chain with mixed token counts

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — 142 passed
- [x] 3.2 Run `python -m compileall -q features/segmenter.py` on modified file
- [x] 3.3 Docs: no updates needed — this is a bug fix that aligns implementation with the existing `segment_pooling` concat spec (output always `segment_size × obs_dim`)
