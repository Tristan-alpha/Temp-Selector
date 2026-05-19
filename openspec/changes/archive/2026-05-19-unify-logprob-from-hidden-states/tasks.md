## 1. Remove max_logprobs

- [x] 1.1 Remove `max_logprobs` parameter from `VLLMFeatureExporter.__init__` and `self.max_logprobs` attribute
- [x] 1.2 Remove `max_logprobs=self.max_logprobs` from `llm_kwargs` in `_lazy_init`

## 2. Unify feature-control interface in generate_with_features

- [x] 2.1 Add `return_logprobs: bool` and `device: torch.device | None` parameters to `generate_with_features`, matching `extract_from_ids`
- [x] 2.2 Remove `logprobs=top_k` from `SamplingParams` in `generate_with_features`
- [x] 2.3 Read hidden states when `return_logprobs=True` or `return_hidden=True`; compute logprobs via `self._llm.apply_model(_LogprobsComputeFn(...))`, following the same pattern as `extract_from_ids`
- [x] 2.4 `return_logprobs=False` → `logprobs` key is `None`; `return_hidden=False` → `hidden_states` key is `None`

## 3. Update callers

- [x] 3.1 Update `ppo/training.py`: pass `return_logprobs=True, return_hidden=(feature_mode=="hidden_states")` to `generate_with_features`
- [x] 3.2 Update `ppo/eval.py`: same explicit `return_logprobs` / `return_hidden` flags

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 4.2 Run `python -m compileall -q inference/vllm_runner.py ppo/training.py ppo/eval.py` to catch syntax errors
- [x] 4.3 Check whether CLAUDE.md needs updating for removed `max_logprobs` parameter
