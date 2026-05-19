## 1. Add make_cached_collate_fn to mil/utils.py

- [x] 1.1 Implement `make_cached_collate_fn(segment_cache, instance_dim, train_device)` that returns a collate_fn reading from pre-computed cache; pads instances to max K, outputs same dict format as original collate_fn
- [x] 1.2 Remove `TokenBatchSampler` class from `mil/utils.py`
- [x] 1.3 Remove `from torch.utils.data import Sampler` (if unused after removal)

## 2. Add pre-computation pass to mil/training.py

- [x] 2.1 After pre-tokenization, add pre-computation loop: iterate dataset rows in batches of `batch_size`, use existing `make_collate_fn` with extractor to build features, split per-row and store in `segment_cache: List[Dict]`
- [x] 2.2 Replace training `DataLoader` with simple `DataLoader(dataset, batch_size=N, shuffle=True, collate_fn=cached_collate_fn)`. Remove `TokenBatchSampler` usage.
- [x] 2.3 Apply same cache logic to validation DataLoader (pre-compute val features once, use cached collate_fn for validation)
- [x] 2.4 Add `precompute_batch_size` config read (default to `batch_size`)

## 3. Config updates

- [x] 3.1 Add `precompute_batch_size: 64` to `configs/training/base.yaml` under `mil.training` (defaults to `batch_size` if absent)

## 4. Documentation

- [x] 4.1 Update `mil/DESIGN.md`: document pre-computation flow; remove TokenBatchSampler references
- [x] 4.2 Update `CLAUDE.md`: if TokenBatchSampler is mentioned, remove or update references

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests must pass; add test for `make_cached_collate_fn`
- [x] 5.2 Run `python -m compileall -q mil/utils.py mil/training.py` to catch syntax errors
- [x] 5.3 Verify cache memory usage: for a test dataset of 1000 rows, confirm cache size is proportional (~60-250 MB)
