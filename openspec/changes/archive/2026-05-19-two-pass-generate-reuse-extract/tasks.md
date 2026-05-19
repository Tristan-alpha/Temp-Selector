## 1. Fix dtype handling

- [x] 1.1 Remove `.float()` from `resp_hs = hs_1d[-n_tok:].float()` in `generate_with_features`; keep `.float()` only when populating `hs_tensor` in output dict

## 2. Two-pass architecture in generate_with_features

- [x] 2.1 Collect `prompt_token_ids` from Pass 1 output and construct `full_ids = prompt_token_ids + generated_token_ids` per sample
- [x] 2.2 Replace inline hidden-state reading and logprob computation with a call to `self.extract_from_ids(full_ids, prompt_lens, temperatures, top_k, return_logprobs, return_hidden, device)`
- [x] 2.3 Zip `extract_from_ids` results back into the per-prompt dicts (`logprobs`, `hidden_states`)
- [x] 2.4 Remove now-unused `safe_open` import from `generate_with_features`

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 3.2 Run `python -m compileall -q inference/vllm_runner.py` to catch syntax errors
- [ ] 3.3 Manually verify `generate_with_features` returns correct logprobs for generated tokens (e.g., run a quick PPO iteration)
