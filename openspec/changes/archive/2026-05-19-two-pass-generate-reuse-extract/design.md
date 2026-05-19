## Context

`extract_hidden_states` speculative decode captures hidden states only during prefill. `generate_with_features` sends text prompts, so the generated `segment_size` tokens are decode-phase — their hidden states are NOT in the safetensors output. The existing single-pass implementation slices `hs[-n_tok:]` which returns the last `n_tok` PROMPT hidden states, not generated-token hidden states.

## Goals / Non-Goals

**Goals:**
- Fix logprobs/hidden states for generated tokens
- Eliminate code duplication between `generate_with_features` and `extract_from_ids`
- Fix dtype mismatch (float32 vs bfloat16) in `_LogprobsComputeFn`

**Non-Goals:**
- Changing `extract_from_ids` interface or behavior
- Changing callers of `generate_with_features`

## Decisions

### Decision 1: Two-pass architecture — generate then extract

**Choice**: Pass 1 generates with `llm.generate(prompts, SamplingParams(max_tokens=segment_size))` to get `token_ids`, `text`, `finish_reason`. Pass 2 calls `self.extract_from_ids(full_ids_list, prompt_lens, ...)` where `full_ids = prompt_token_ids + generated_token_ids`. This is a full prefill, so hidden states cover the entire sequence.

**Why**: `extract_from_ids` is already correct for pre-tokenized sequences. Reusing it guarantees correct hidden states and logprobs. The two vLLM calls are sequential (same engine), avoiding the "multiple engines" constraint.

### Decision 2: delegate feature extraction to `extract_from_ids`

**Choice**: Call `self.extract_from_ids()` from within `generate_with_features` instead of duplicating the safetensors + apply_model logic inline.

**Alternatives considered**:
- Extract shared helper methods (`_read_hidden_states`, `_compute_logprobs`). Rejected: still introduces a new abstraction when `extract_from_ids` already does exactly the job.

### Decision 3: Preserve original dtype until `_LogprobsComputeFn`

**Choice**: Remove `.float()` call on `resp_hs` before passing to logprob computation. Let `_LogprobsComputeFn.__call__` cast to model's dtype.

## Risks / Trade-offs

- **Two vLLM calls per segment**: Each PPO segment round costs one extra `llm.generate()` call. Pass 2 uses `max_tokens=1` and is a pure prefill (fast, ~no generation overhead). Acceptable trade-off for correctness.
- **Prefix caching**: Pass 1 caches the prompt KV. Pass 2 rereads the same prompt — APC should still benefit from shared prompt prefixes across segments.
