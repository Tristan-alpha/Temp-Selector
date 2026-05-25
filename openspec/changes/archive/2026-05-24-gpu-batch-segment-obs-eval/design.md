## Context

`_evaluate_strategy_batch` calls `build_segment_obs_from_lp` per-chain in a Python loop. Each call does `torch.exp(lp)` on [~32, 4096], `torch.cat`, pad/truncate, and `segment_pooling`. For ~1500 active chains this takes 68-101 seconds on CPU. All chains entering this path have identical `n_tok = segment_size` (EOS chains are skipped via `continue`). The vLLM runner already reserves one GPU (`reserve_training_gpu=True`) that sits idle during eval.

## Goals / Non-Goals

**Goals:**
- Move the token-level math (`exp`, `cat`, pad/truncate) to GPU in a single stacked batch
- For the common `fixed_window` + `mean` pooling case, also batch the pooling on GPU
- Resolve GPU device in `__init__` using the same `cuda:n_gpu-1` pattern as `ppo/training.py`
- Fall back to per-chain CPU path when no GPU is available

**Non-Goals:**
- Changing the MIL training path (uses same `build_segment_obs_from_lp` but with DataLoader parallelism, different constraints)
- Batching across rounds (each round depends on previous round's generation output)
- Batching the policy decision loop (already measured at ~1.4s, not a bottleneck)

## Decisions

### Decision 1: New function vs inline GPU code in eval

**Chosen: New `batch_build_segment_obs_from_lp` in `features/segmenter.py`.**

- **Alternative A**: Inline the GPU logic directly in `_evaluate_strategy_batch`. Rejected — the function is already long; the GPU math is a self-contained operation with clear inputs/outputs that belongs with the existing `build_segment_obs_from_lp`.
- **Alternative B**: Modify `build_segment_obs_from_lp` to accept batched inputs. Rejected — would break the MIL training path which calls it per-sample in `collate_fn` with different `num_workers=0` constraints.

### Decision 2: Pooling on GPU vs CPU after batch

**Chosen: For `fixed_window` mode, pool on GPU. For `step` mode, pool per-chain on CPU after GPU batch.**

Rationale: In `fixed_window` mode, every chain has the same span `[Segment(0, segment_size)]` and `mean` pooling is just `.mean(dim=1)` — trivially batchable. GPU→CPU transfer after pooling is only `[B, obs_dim]` (~24MB) instead of `[B, max_tok, obs_dim]` (~750MB). In `step` mode, spans differ per chain, so pooling must be per-chain.

### Decision 3: Handling variable n_tok

**Chosen: Assert uniform n_tok in the eval path, with padding as a safety fallback.**

All active chains generate exactly `segment_size` tokens per round. If this invariant breaks (e.g., a chain produces fewer tokens), `torch.stack` will raise a clear error. A padding fallback in the batch function handles the rare edge case where tokens differ (e.g., final partial round).

### Decision 4: GPU device resolution

**Chosen: Same `cuda:n_gpu-1` pattern as `ppo/training.py`.**

```python
n_gpu = torch.cuda.device_count()
self.device = torch.device(f"cuda:{max(0, n_gpu - 1)}") if n_gpu > 0 else torch.device("cpu")
```

This picks the last GPU, which vLLM leaves free when `reserve_training_gpu=True`. When no GPU is available, falls back to CPU (existing behavior).

## Risks / Trade-offs

- **Risk**: GPU memory pressure — stacking 1500 × 32 × 4097 float32 = ~750MB. Plus vLLM KV cache on other GPUs. **Mitigation**: The reserved GPU is otherwise idle; 750MB is well within a 48GB RTX 5880's capacity.
- **Risk**: `step` mode pooling per-chain on CPU still costs ~50ms (negligible vs the 68s saved from token-level math).
- **Trade-off**: The batch function duplicates some logic from `build_segment_obs_from_lp` (the `sampled`/`entropy`/`parts` construction). Acceptable for a function with different dimensionality (2D per-chain → 3D batched).
