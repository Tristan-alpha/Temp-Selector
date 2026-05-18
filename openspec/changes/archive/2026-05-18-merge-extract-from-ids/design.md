## Context

Two methods exist for online feature extraction from pre-tokenized IDs:
- `extract_logprobs_from_ids`: runs `llm.generate()` + `apply_model()` for logprobs
- `extract_hidden_from_ids`: runs `llm.generate()` for hidden states from safetensors

Both hit the same `llm.generate()` path with the same `full_ids` input. Hidden state reading is identical in both.

## Goals / Non-Goals

**Goals:**
- Single `llm.generate()` call per batch
- Single safetensors read per sample
- Clean dict-based return interface

**Non-Goals:**
- Changing the `_LogprobsComputeFn` logic
- Changing the collate_fn tensor construction

## Decisions

### Decision 1: dict return over tuple

**Alternative:** Return `(logprob_tensors, hidden_tensors)` tuple.

**Rationale:** Dict with string keys makes call sites self-documenting and handles the case where only one is requested.

### Decision 2: Keep `_LogprobsComputeFn` unchanged

**Rationale:** The compute function is correct and tested. Only the orchestration layer changes.

## Risks / Trade-offs

- **Hidden state shape different in current implementations**: `extract_hidden` returns `[R+2, hidden_dim]` (no trimming), while `extract_logprobs` returns `[R, top_k+1]` after trimming. The merged method must standardize on trimmed `[R, ...]`.
