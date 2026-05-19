## ADDED Requirements

### Requirement: make_cached_collate_fn exists alongside make_collate_fn

`mil/utils.py` SHALL export `make_cached_collate_fn(segment_cache, instance_dim, train_device)`. This factory SHALL return a collate function that reads pre-computed segment tensors, labels, and temp indices from `segment_cache` (a list of dicts). The returned collate_fn SHALL pad instances to max K within the batch and output the same dict format as `make_collate_fn`.

#### Scenario: make_cached_collate_fn returns valid collate_fn

- **WHEN** `make_cached_collate_fn(cache, instance_dim=4098, train_device=device)` is called
- **THEN** it SHALL return a callable that accepts a list of row indices and returns `{instances, mask, label, temp_idx, _batch_tokens}`

## REMOVED Requirements

### Requirement: TokenBatchSampler

**Reason**: With pre-computed segment features, per-sample token counts are irrelevant for batching. A simple `DataLoader(batch_size=N)` replaces it.

**Migration**: Replace `TokenBatchSampler(token_counts, max_tokens, shuffle=True)` with `DataLoader(dataset, batch_size=N, shuffle=True, collate_fn=...)`. Remove the `TokenBatchSampler` class from `mil/utils.py`.
