## Why

MIL training spends ~90% of wall-clock time in vLLM `extract_from_ids` to pre-compute segment features for train and validation sets. These features are deterministic given the same dataset and extraction config — recomputing them on every run is wasteful. Caching them to disk enables instant reload on subsequent runs.

## What Changes

- Add a cache file naming convention: `datasets/cache/{split}-{segment_mode}-{pooling_mode}-{feature_mode}-{instance_dim}-{segment_size}.pt`
- `train()` and `evaluate_mil()` check for cache existence before vLLM extraction; load from disk if hit, compute and save if miss
- No config schema changes — cache path is derived from config values automatically
- No breaking changes to training or eval APIs

## Capabilities

### New Capabilities
- `mil-segment-cache`: deterministic disk cache for pre-computed MIL segment features, keyed by split name and extraction parameters

## Impact

- Affected files: `mil/training.py`, `mil/eval.py`, new `datasets/cache/` directory
- Dependencies: `torch.save` / `torch.load` (already available)
- Disk usage: ~12 GB for concat pooling with 48K bags, ~200 MB for mean pooling
