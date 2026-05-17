## 1. Add extraction methods to VLLMFeatureExporter

- [x] 1.1 Remove ValueError guard for hidden_states/all feature_mode
- [x] 1.2 LLM constructor: add `max_logprobs`, speculative_config for hidden states, kv_transfer_config with tempdir, enforce_eager for prefill preset
- [x] 1.3 Add `extract_logprobs_from_ids` — `SamplingParams(max_tokens=1, prompt_logprobs=top_k)`, slice `out.prompt_logprobs[p_len:]`, convert to tensor
- [x] 1.4 Add `extract_hidden_from_ids` — `SamplingParams(max_tokens=1)`, read safetensors, squeeze, slice `[p_len-1:]`
- [x] 1.5 Delete safetensors file immediately after read; `atexit` cleanup for crashes
- [x] 1.6 Layer ID: 1-indexed, Qwen3-8B last layer = 36

## 2. Wire vLLM backend into MIL training

- [x] 2.1 `mil/training.py`: backend switch (vllm vs sglang)
- [x] 2.2 `mil/eval.py`: same
- [x] 2.3 Config: `backend: vllm` works via existing config key

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — 128 passed
- [ ] 3.2 Run `python -m compileall -q` on modified files
- [ ] 3.3 Delete verify scripts after confirming extraction works
