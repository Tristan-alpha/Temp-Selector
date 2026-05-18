## 1. Add generate_with_features to runner

- [x] 1.1 Add `generate_with_features` method to `VLLMFeatureExporter`
- [x] 1.2 Delete `generate_raw` method

## 2. Clean up PPO training

- [x] 2.1 Remove manual TP computation; pass `reserve_training_gpu=True` to VLLMFeatureExporter
- [x] 2.2 Delete `_extract_segment_obs` function
- [x] 2.3 Update training loop to use `generate_with_features` instead of `generate_raw` + `_extract_segment_obs`

## 3. Clean up dead code

- [x] 3.1 Remove `token_to_vec` from `features/vectorizer.py` if no other callers
- [x] 3.2 Check `ppo/eval.py` for any references to deleted functions

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 4.2 Run `python -m compileall -q` on all modified files
- [x] 4.3 Check whether docs need updating (CLAUDE.md references generate_raw — update if needed)
