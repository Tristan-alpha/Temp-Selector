## ADDED Requirements

### Requirement: MIL pre-computes segment features before training

`mil/training.py` SHALL pre-compute segment-level instance vectors for all training rows before the epoch loop. Pre-computation SHALL use the existing `make_collate_fn` with the vLLM extractor to batch-process rows. The resulting per-row tensors [K_i, instance_dim] SHALL be stored in a system RAM cache indexed by row position.

#### Scenario: Pre-computation completes before first epoch

- **WHEN** MIL training starts
- **THEN** all segment features SHALL be computed once before the first training epoch
- **AND** subsequent epochs SHALL use cached features without calling `extract_from_ids`

### Requirement: Cached collate_fn skips feature extraction

`mil/utils.py` SHALL provide a `make_cached_collate_fn(segment_cache, train_device)` factory. The returned collate_fn SHALL construct batch tensors by reading pre-computed segment tensors from the cache, without calling any vLLM methods. It SHALL still pad to max K within the batch and return the same dict format as the original collate_fn.

#### Scenario: Cached collate_fn produces identical output shape

- **WHEN** `make_cached_collate_fn` processes a batch of indices
- **THEN** the returned dict SHALL contain keys `instances`, `mask`, `label`, `temp_idx` with the same shapes as the original collate_fn

#### Scenario: Cache lookup by row index

- **WHEN** a row at position `idx` is included in a batch
- **THEN** the collate_fn SHALL read `segment_cache[idx]["instances"]`, `segment_cache[idx]["label"]`, and `segment_cache[idx]["temp_idx"]`

### Requirement: Cache fits in system RAM

The segment cache SHALL store only post-pooling segment tensors [K, 4098] float32, not per-token features. For 54,000 samples, total cache size SHALL be under 15 GB in system RAM.

#### Scenario: Cache size is bounded

- **WHEN** segment features are pre-computed for N rows with average 8 segments per row
- **THEN** total cache size SHALL be approximately N × 8 × 4098 × 4 bytes

### Requirement: Pre-computation uses token-based batching

Pre-computation and eval SHALL use `mil.utils.token_batches(rows, max_tokens_per_batch)` to group rows into batches. This ensures each vLLM `extract_from_ids` call stays within the GPU memory budget controlled by `mil.training.max_tokens_per_batch`.

#### Scenario: Token-based batching for vLLM calls

- **WHEN** pre-computing segment features or running online eval
- **THEN** rows SHALL be grouped such that `sum(len(row["_full_ids"]))` per batch ≤ `max_tokens_per_batch`
