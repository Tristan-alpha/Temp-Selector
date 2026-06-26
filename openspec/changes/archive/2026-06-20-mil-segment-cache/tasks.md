## 1. Cache helper

- [x] 1.1 Add `_build_cache_path(split, segment_mode, pooling_mode, feature_mode, instance_dim, segment_size) -> str` to `mil/utils.py`
- [x] 1.2 Add `_load_or_build_segment_cache(dataset_rows, runner, collate_fn, cache_path, max_tokens_per_batch, logger) -> List[Dict]` to `mil/utils.py` — checks cache, runs vLLM extraction on miss, saves on miss

## 2. Training integration

- [x] 2.1 In `mil/training.py` `train()`: replace the inline "Pre-compute segment features" loops for train and val with `_load_or_build_segment_cache()` calls
- [x] 2.2 Auto-create `datasets/cache/` directory on first cache write

## 3. Evaluation integration

- [x] 3.1 In `mil/eval.py` `evaluate_mil()`: use `_load_or_build_segment_cache()` for the eval dataset's segment features
- [x] 3.2 Remove duplicate pre-compute logic from eval path

## 4. Tests

- [x] 4.1 Add tests for `_build_cache_path` — verify correct path for various config combinations
- [x] 4.2 Add test for cache hit path — create a temp cache file, verify it's loaded without vLLM
- [x] 4.3 Add test for cache miss path — verify torch.save is called after extraction

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 5.2 Run `python -m compileall -q mil/utils.py mil/training.py mil/eval.py`
