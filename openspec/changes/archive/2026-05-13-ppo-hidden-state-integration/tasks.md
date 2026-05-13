## 1. Update _extract_segment_obs

- [x] 1.1 Add optional `hidden_states` parameter to `_extract_segment_obs` in `ppo/training.py` — when provided (non-None), mean-pool the hidden states and concatenate with the standard feature vector
- [x] 1.2 Support `all` mode: standard features (logprob/entropy/topk_logits) + hidden states concatenated

## 2. Per-segment extraction in train_ppo

- [x] 2.1 Read `feature_mode` from config, init `VLLMHiddenStateExtractor` when `hidden_states` or `all`
- [x] 2.2 Maintain per-episode accumulated text prefix (`ep_prefixes[i]`)
- [x] 2.3 After each segment generation, if hidden states are needed, call extractor with `(ep_prefixes[i], new_segment_text)` and pass result to `_extract_segment_obs`
- [x] 2.4 Handle first segment (dummy action, no extraction needed)

## 3. Feature dimension alignment

- [x] 3.1 `obs_dim` for PolicyValueNet reads from `instance_dim` — works automatically when config has `instance_dim: 4096`
- [x] 3.2 Mean-pool hidden states from per-token `[M, 4096]` to `[4096]`

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 4.2 Run `python -m compileall -q ppo/training.py`

## 5. Documentation

- [x] 5.1 Update ppo/DESIGN.md per-segment extraction section
