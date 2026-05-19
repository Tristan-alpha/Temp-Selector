## Context

`VLLMFeatureExporter` currently has two logprob computation paths:

- **`generate_with_features`** (PPO / Stage 1): `SamplingParams(logprobs=4096)` tells vLLM to compute per-token logprobs during generation. Results parsed from `output.outputs[0].logprobs`. Hidden states are only read from safetensors when `return_hidden=True`.

- **`extract_from_ids`** (MIL): generates 1 token with `SamplingParams(max_tokens=1)`, always reads hidden states from safetensors, then calls `self._llm.apply_model(_LogprobsComputeFn(...))` to compute logprobs from hidden states.

The vLLM engine is always configured with `speculative_config(extract_hidden_states)` and `kv_transfer_config(ExampleHiddenStatesConnector)`, which writes hidden states to `/dev/shm` for every generation request — even when `return_hidden=False`. The I/O cost is already being paid.

The `LLM(max_logprobs=4096)` configuration allocates per-token logprob buffers in vLLM's internal state. Combined with `max_model_len=10240` (8192 + 2048), this adds memory pressure during generation.

## Goals / Non-Goals

**Goals:**
- Unify logprob computation: both methods use `apply_model` + `_LogprobsComputeFn` from hidden states
- Unify feature-control interface: `generate_with_features` gains `return_logprobs` and `device` parameters, mirroring `extract_from_ids`
- Remove `max_logprobs` from `LLM()` configuration to reduce GPU memory pressure
- Remove `logprobs=top_k` from `SamplingParams` in `generate_with_features`
- Preserve identical output dict format (`logprobs` key is still `[n_tok, top_k+1]` tensor)

**Non-Goals:**
- Changing feature mode semantics (`topk_logprobs` vs `hidden_states`)
- Changing `extract_from_ids` interface (already correct, used as reference)
- Changing config schema

## Decisions

### Decision 1: Compute logprobs from hidden states in `generate_with_features`

**Choice**: After `llm.generate()`, read hidden states from safetensors, then call `apply_model(_LogprobsComputeFn)` — the same pattern as `extract_from_ids`.

**Alternatives considered**:
- A) Keep using `SamplingParams(logprobs=)` and just remove `LLM(max_logprobs=4096)`. Rejected: `SamplingParams(logprobs=4096)` can't work without `LLM(max_logprobs>=4096)`.
- B) Switch back to pure SGLang for logprobs. Rejected: SGLang was already removed from the codebase.

### Decision 2: Align `generate_with_features` signature with `extract_from_ids`

**Choice**: Add `return_logprobs` and `device` parameters to `generate_with_features`. Both methods now use the same pattern: callers explicitly opt into logprob and/or hidden state output via boolean flags derived from `feature_mode`.

**Why**: Currently `generate_with_features` always returns logprobs (implicit) while `extract_from_ids` uses explicit `return_logprobs`. Adding the flag makes the interface symmetrical and callers self-documenting. The `device` parameter controls where the logprob tensor is assembled, matching `extract_from_ids`.

### Decision 3: Always read hidden states in `generate_with_features`

**Choice**: Read `hidden_states_path` from every output unconditionally (to compute logprobs). The `return_hidden` flag controls only whether raw hidden state tensors appear in the returned dict.

**Why**: Hidden states are already written to `/dev/shm` by the speculative config regardless of `return_hidden` — the disk I/O is sunk cost. Reading them into CPU memory adds negligible overhead (~512 tokens × 4096 hidden_dim × 4 bytes ≈ 8 MB per segment in PPO).

### Decision 4: Remove `max_logprobs` from `__init__` and `LLM()`

**Choice**: Delete `max_logprobs` parameter entirely from `VLLMFeatureExporter.__init__` and from `llm_kwargs` in `_lazy_init`.

**Why**: After removing `SamplingParams(logprobs=)`, vLLM's logprob buffer is never used. No caller passes `max_logprobs` explicitly — all use the default. Removing it reduces GPU memory pressure and simplifies the API.

## Risks / Trade-offs

- **`apply_model` adds a CPU→GPU→CPU round-trip per segment**: In PPO, each `generate_with_features` call produces ~512 tokens. `extract_from_ids` already chunks this into 1024-token windows, so it's 1 `apply_model` call per segment. This is negligible compared to the generation cost.

- **Hidden state reading from `/dev/shm` is now always on**: Previously skipped when `return_hidden=False`. But the speculative config already writes these files — we were just ignoring them. Reading is cheap.

- **Logprob numerical equivalence**: vLLM's built-in logprob computation and `apply_model` + `compute_topk_logprobs` both use the same model parameters. Results should be identical modulo floating-point precision, but this should be verified after implementation.

## Open Questions

None.
