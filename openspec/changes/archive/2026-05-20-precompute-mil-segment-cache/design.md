## Context

MIL training currently does online feature extraction in `collate_fn`:
1. `extract_from_ids()` — vLLM prefill to get logprobs/hidden states per token
2. `build_segment_obs_from_lp()` — convert to per-token feature vectors
3. `segment_pooling()` — pool to [K, 4098] per sample

Steps 2-3 are cheap CPU operations. Step 1 is the bottleneck — a GPU round-trip per batch, repeated every epoch, while the segment features are static (the underlying model weights don't change).

Pre-computing the [K, 4098] segment tensors once and caching in RAM eliminates vLLM from the training loop entirely.

## Goals / Non-Goals

**Goals:**
- Pre-compute segment features for all rows once before MIL training loop
- Cache fits in system RAM (~3-13 GB for 54K samples)
- Training epochs use a lightweight `make_cached_collate_fn` that does cache lookup + padding only
- Use existing `mil.training.batch_size` config key for pre-computation batching
- Remove `TokenBatchSampler` (no longer needed — token counts are irrelevant after pooling)

**Non-Goals:**
- Caching raw per-token features (too large: ~TB range)
- Changing the MIL model or loss
- Applying caching to PPO (PPO generates with different temps each time — features are not static)
- GPU memory caching (use system RAM)

## Decisions

### Decision 1: Cache at segment level, not token level

**Chosen**: Store `[K_i, 4098]` float32 tensors after segment pooling.

**Alternatives considered**:
- *Cache per-token features [n_tok, 4098]* → Rejected. 54K × 2048 × 4098 × 4B ≈ 1.65 TB.
- *Cache raw vLLM output* → Rejected. Even larger + requires re-running segment construction.

The 512:1 compression from segment pooling is what makes caching feasible.

### Decision 2: Cache in system RAM as List[Dict]

**Chosen**: Python list of dicts: `[{"instances": Tensor[K_i,4098], "label": float, "temp_idx": int}, ...]`. Indexed by row position.

**Alternatives considered**:
- *Single padded tensor [N, max_K, 4098]* → Rejected. Max_K=16, but most samples have K=4-8, wasting 50%+ memory.
- *Disk-backed (mmap / h5py)* → Rejected. 3-13 GB fits in RAM comfortably on any ML server.

### Decision 3: Pre-computation uses existing collate_fn

**Chosen**: The pre-computation pass uses the existing `make_collate_fn` (with `extractor`) to batch-process rows. Results are split per-row and stored in cache. After pre-computation, training switches to `make_cached_collate_fn`.

**Alternatives considered**:
- *Separate pre-computation code path* → Rejected. Duplicates segment construction logic.

### Decision 4: Simple sequential batching, remove TokenBatchSampler

**Chosen**: Replace `TokenBatchSampler` with `DataLoader(batch_size=N, shuffle=True)`. Since all rows now produce small fixed-size tensors, token-count-based batching provides no benefit.

**Alternatives considered**:
- *Keep TokenBatchSampler with dummy counts* → Rejected. Adds complexity for zero benefit.

### Decision 5: Pre-computation batch size from existing config

**Chosen**: Use `mil.training.batch_size` for the pre-computation batching. Optionally allow override via `precompute_batch_size`.

**Alternatives considered**:
- *Hardcoded batch size* → Rejected. Different GPUs have different limits for the vLLM prefill call.
- *New required config key* → Rejected. `batch_size` already expresses the right constraint.

## Risks / Trade-offs

- **[Risk] RAM pressure on small servers**: 3-13 GB is fine for most ML servers, but if someone runs on a 16 GB laptop it could be tight.
  → **Mitigation**: Document the RAM requirement. Can be skipped if `precompute: false` in config.

- **[Trade-off] Cache invalidation**: If the dataset changes, cache must be regenerated.
  → Acceptable. Regeneration is automatic (computed fresh each training run).

- **[Risk] First epoch startup time**: Pre-computation takes ~10-30 min for 54K samples.
  → Acceptable trade-off. Saves ~1.4-5.5 hours over 50 epochs.

## Migration Plan

1. Add `make_cached_collate_fn` to `mil/utils.py`
2. Add pre-computation pass to `mil/training.py`
3. Replace `TokenBatchSampler` with simple `DataLoader(batch_size=...)`
4. Remove `TokenBatchSampler` class and import
5. Add `precompute_batch_size` config key (optional, defaults to `batch_size`)
