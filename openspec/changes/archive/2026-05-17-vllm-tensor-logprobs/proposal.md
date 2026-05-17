## Why

vLLM's current `extract_logprobs_from_ids` uses `prompt_logprobs=4096` which returns Python `Logprob` objects — 983K iterations per batch to extract `.logprob` from per-token dicts. Scheme C (`apply_model` + `compute_logits` + `compute_topk_logprobs`) runs entirely on GPU with native tensors, avoiding Python overhead. Additionally, `feature_mode="all"` can get both hidden states and logprobs from a single generate call.

## What Changes

- **BREAKING**: `extract_logprobs_from_ids` rewritten to use `apply_model` → `model.compute_logits` → `compute_topk_logprobs`
- `extract_hidden_from_ids` removed — hidden states and logprobs come from the same generate call in `feature_mode="all"`
- LLM constructor always enables speculative hidden state extraction for `feature_mode != "basic"`
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` added to shell scripts
- `feature_mode="all"`: one generate call produces both hidden states (via safetensors) and logprobs (via apply_model)
- `feature_mode="topk_logprobs"`: one generate call, apply_model for logprobs only

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: vLLM extraction SHALL use `apply_model` for tensor-based logprob computation instead of per-token Python iteration

## Impact

- `inference/vllm_runner.py` — rewrite extraction methods
- `scripts/run_pipeline.sh` — env var
- `mil/training.py`, `mil/eval.py` — no changes (same interface)
