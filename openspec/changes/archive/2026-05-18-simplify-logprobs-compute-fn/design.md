## Context

`_LogprobsComputeFn.__call__` contains a CHUNK_SIZE loop that chunks hidden states, computes logits+topk_logprobs per chunk, and cats results on CPU. The return value is `torch.stack([lp, ids])` where `ids` is never used by any caller.

## Goals / Non-Goals

**Goals:**
- Remove unused `id_chunks` computation and transmission
- Simplify `_LogprobsComputeFn` to process a single chunk
- Move chunk loop + cat to `extract_from_ids` with device control

**Non-Goals:**
- Changing `compute_topk_logprobs` or the math

## Decisions

### Decision 1: N apply_model calls over 1 with list return

**Alternative:** Keep single `apply_model` call, return list of CPU tensors from `__call__`.

**Rationale:** Returning a list through pickle is less clean than returning a single tensor. Per-chunk `apply_model` calls have the same total data transfer (actually less, since ids are dropped). For typical R ≤ 1024 (256 tokens), N = 1 — zero overhead. For R = 8192, N = 8 calls with ~1ms fixed overhead each = negligible vs seconds of prefill.

### Decision 2: device parameter on extract_from_ids

**Alternative:** Cat always on CPU, let collate_fn handle device transfer.

**Rationale:** Cat on training GPU saves one `.to(device)` call per sample. Default `None` means CPU cat (backward compatible).
