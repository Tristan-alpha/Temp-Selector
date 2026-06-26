## Why

`VLLMFeatureExporter` currently uses two passes to obtain per-token hidden states for logprob computation: Pass 1 generates tokens, Pass 2 re-feeds the full sequence via `extract_from_ids` followed by `apply_model`. This wastes ~50% of inference time. vLLM's `include_output_tokens` feature (via `SamplingParams.extra_args`) exports hidden states for **all** tokens except the very last generated one in a single `generate()` call — and the missing last token is irrelevant because `hs[t]` predicts `token[t+1]`, so the hidden states needed for response logprobs are fully covered.

## What Changes

- **`SamplingParams.extra_args`**: Pass `{"kv_transfer_params": {"include_output_tokens": True}}` in Pass 1 so the connector saves hidden states for `all_token_ids[:-1]` (prompt + all generated tokens except the last).
- **`generate_with_features`**: Remove Pass 2 (`extract_from_ids` + `apply_model`). Compute logprobs directly from Pass 1's hidden states using the same `_LogprobsComputeFn` / `compute_topk_logprobs` path, just operating on the single-pass hs tensor.
- **`extract_from_ids`**: May be retained as a standalone utility or removed if no other callers exist.
- **Hidden state loading**: Use `load_hidden_states` / `cleanup_hidden_states` from `example_hidden_states_connector` to correctly block on async disk writes via `flock`.

## Capabilities

### New Capabilities
- `single-pass-hs-logprobs`: Extract hidden states **and** compute per-token logprobs in a single `llm.generate()` call, eliminating the two-pass pattern in `VLLMFeatureExporter`.

### Modified Capabilities
- `mil-online-hidden-extract`: The requirement "Feature extraction is pre-compute once, not per-batch" no longer needs the `extract_from_ids` + `apply_model` two-pass flow; single-pass generation replaces it.

## Impact

- `inference/vllm_runner.py` — `generate_with_features` and `extract_from_ids` rewritten or simplified
- `mil/utils.py`, `mil/training.py` — continue to call `make_collate_fn` which internally uses `extract_from_ids`; may benefit from single-pass if collate_fn is updated
- `ppo/training.py`, `ppo/eval.py` — `_process_generated_features` calls `build_segment_obs_from_lp` directly (not `extract_from_ids`); no change needed
- `scripts/verify_hidden_states.py` — verification script confirming `include_output_tokens` works
