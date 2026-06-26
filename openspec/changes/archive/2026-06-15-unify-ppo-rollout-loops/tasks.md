## 1. Extract shared helper functions

- [x] 1.1 Implement `_decide_temperature(segment_obs, policy, temp_bins, device, deterministic)` — returns `(temp, action, logp, value)`. When `segment_obs is None`, return default T=0.7 + dummy tensors. When `deterministic=True`, use argmax. When `deterministic=False`, use `sample_action`.
- [x] 1.2 Implement `_process_generated_features(feat_dict, tokenizer, segment_size, instance_dim, device, segment_mode, hs_needed, pooling_mode)` — returns `(text_delta, is_done, next_segment_obs | None)`. Detects EOS/stop/empty → `is_done=True`. Otherwise calls `build_segment_obs_from_lp` and returns the observation.

## 2. Update training rollout

- [x] 2.1 Replace inline temperature decision (lines 214-235) with call to `_decide_temperature(deterministic=False)`. Keep PPO buffer recording (ep_obs/actions/logprobs/values) in the loop since it's training-specific.
- [x] 2.2 Replace inline feature processing (lines 250-270) with call to `_process_generated_features`. Keep text accumulation and active-flag management in the loop.
- [x] 2.3 Verify training rollout: PPO buffer shapes unchanged, reward computation unchanged, batch construction unchanged.

## 3. Update validation rollout

- [x] 3.1 Replace temperature decision (lines 392-399) with `_decide_temperature(deterministic=True)`. Remove the unconditional `val_seg_obs[i][v] = None`.
- [x] 3.2 Fix `generate_with_features` flags: change `return_logprobs=False` to `True`, `return_hidden=False` to `hs_needed`.
- [x] 3.3 Add `_process_generated_features` call in the feature processing loop (line 413), storing `next_segment_obs` into `val_seg_obs[i][v]` for non-terminated chains.

## 4. Verification

- [x] 4.1 Run `python -m compileall -q ppo/training.py` — no syntax errors.
- [x] 4.2 Run `python -m pytest tests/ -v` — all existing tests pass. (PPO tests are CPU-only; no GPU needed.)
- [x] 4.3 Review `CLAUDE.md` for any outdated description of the validation rollout. Update if needed. No changes needed — CLAUDE.md doesn't describe validation rollout internals.
