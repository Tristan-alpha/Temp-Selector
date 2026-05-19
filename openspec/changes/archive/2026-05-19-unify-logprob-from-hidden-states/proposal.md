## Why

`generate_with_features` and `extract_from_ids` use two different mechanisms to obtain logprobs — the former relies on vLLM's `SamplingParams(logprobs=)` while the latter computes them from hidden states via `apply_model`. This duplication complicates the code, and the `LLM(max_logprobs=4096)` + `SamplingParams(logprobs=4096)` configuration adds unnecessary GPU memory pressure that contributes to CUDA illegal memory access errors in PPO training.

## What Changes

- `generate_with_features` computes logprobs from hidden states via `apply_model` (same as `extract_from_ids`), instead of from `SamplingParams(logprobs=)`
- `generate_with_features` gains `return_logprobs` and `device` parameters, matching the explicit feature-control pattern of `extract_from_ids`; `return_hidden` controls only raw hidden state output
- Remove `max_logprobs` parameter from `VLLMFeatureExporter.__init__` and `LLM()` configuration
- Remove `logprobs=top_k` from `SamplingParams` in `generate_with_features`
- Callers (PPO training/eval) explicitly pass `return_logprobs=True` and `return_hidden=(feature_mode=="hidden_states")`
- **BREAKING**: `VLLMFeatureExporter(max_logprobs=)` kwarg removed (all callers use the default, no impact)

## Capabilities

### New Capabilities

- `unified-logprob-computation`: Both `generate_with_features` and `extract_from_ids` compute per-token logprobs from hidden states through the same `apply_model` + `_LogprobsComputeFn` path, eliminating the dependency on vLLM's built-in logprob mechanism.

### Modified Capabilities

- `ppo-online-generation`: `generate_with_features` no longer passes `logprobs=top_k` to `SamplingParams`; logprobs are computed from hidden states instead. The return dict remains unchanged (`logprobs` key is still a `[n_tok, top_k+1]` tensor).

## Impact

- `inference/vllm_runner.py`: `__init__`, `_lazy_init`, `generate_with_features`
- `configs/training/base.yaml`: `inference.top_k_logprobs` still used as `top_k` parameter for `apply_model` logprob width — no config change needed
