## Why

MIL training currently calls `extract_from_ids` (vLLM prefill) in `collate_fn` for every batch, every epoch. For 54,000 samples, this means ~200 vLLM calls per epoch × 50 epochs = ~10,000 expensive GPU round-trips, while the resulting segment features [K, 4098] are static throughout training. Pre-computing these features once and caching them in RAM (~3-13 GB) eliminates repeated vLLM calls and speeds up training by an order of magnitude.

## What Changes

- `mil/training.py`: Pre-compute segment features for all rows in one pass after pre-tokenization, using the existing `mil.training.batch_size` config key. Store results in a RAM cache (list of per-row tensors).
- `mil/utils.py`: Add a `make_cached_collate_fn` that reads from the pre-computed cache instead of calling `extract_from_ids`. The original `make_collate_fn` is retained for the pre-computation pass itself.
- **BREAKING**: `TokenBatchSampler` is removed. With pre-computed segments, token counts are no longer the batching constraint. A simple sequential `BatchSampler` or `DataLoader` with `batch_size` replaces it.
- Config: Add `mil.training.precompute_batch_size` (defaults to `batch_size`) for the pre-computation pass if a different batch size is desired.

## Capabilities

### New Capabilities

- `mil-segment-cache`: Pre-compute segment-level instance vectors [K, 4098] for all MIL training/validation rows once, cache them in system RAM, and serve them via a lightweight collate_fn during training.

### Modified Capabilities

- `mil-online-hidden-extract`: Feature extraction moves from "every collate_fn call" to "once before training." The `extract_from_ids` call is still used but only during the pre-computation phase. Training epochs use cached features.
- `collate-feature-extraction`: A new `make_cached_collate_fn` factory coexists with the existing `make_collate_fn`. Cached collate_fn skips extraction and does only padding/stacking.

## Impact

- `mil/training.py`: Pre-computation pass + simple DataLoader with `batch_size`
- `mil/utils.py`: Add `make_cached_collate_fn`; remove `TokenBatchSampler`
- `configs/training/base.yaml`: Add `mil.training.precompute_batch_size`
- `mil/eval.py`: No change (eval uses the same collate_fn; pre-computation can optionally apply)
- RAM usage: ~3-13 GB for 54K samples (system memory, not GPU)
