## Why

SGLang prefill throughput for feature extraction is too slow for MIL training. vLLM's `LLM.generate()` with prompt_token_ids and max_tokens=0/1 should provide faster prefill for the same extraction workload. The existing `VLLMFeatureExporter` already wraps vLLM for generation (build_dataset, PPO) but lacks the extraction interface that `SGLangRunner` now provides.

## What Changes

- Lift `VLLMFeatureExporter` constructor restriction on `feature_mode="hidden_states"/"all"` — vLLM now supports hidden state return
- Add `extract_hidden_from_ids(full_ids, prompt_lens) -> List[torch.Tensor]` — returns per-response-token hidden states
- Add `extract_logprobs_from_ids(full_ids, prompt_lens, temperatures?, top_k) -> List[torch.Tensor]` — returns per-response-token top-k logprob tensors
- Use `SamplingParams(prompt_logprobs=top_k, max_tokens=0)` + `prompt_token_ids` for logprob extraction
- Use `LLM` hidden state support for hidden extraction
- Same interface signature as `SGLangRunner` — drop-in replacement for MIL training collate_fn

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: collate_fn SHALL work with either SGLangRunner or VLLMFeatureExporter as the extractor

## Impact

- `inference/vllm_runner.py` — add extraction methods, remove ValueError guard
- `mil/training.py` — optional backend switch (no changes to collate_fn signature)
- `mil/eval.py` — same
- `configs/base.yaml` — `backend: vllm` support
