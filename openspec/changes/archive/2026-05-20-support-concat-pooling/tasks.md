## 1. build_segment_obs_from_lp — add pooling_mode parameter

- [x] 1.1 Add `pooling_mode: str = "mean"` parameter; pass to `segment_pooling(mode=pooling_mode)`

## 2. make_collate_fn — forward pooling_mode

- [x] 2.1 Pass `pooling_mode` kwarg to `build_segment_obs_from_lp` call

## 3. mil/training.py — compute model_input_dim

- [x] 3.1 Compute `model_input_dim = instance_dim * segment_size` when `pooling_mode == "concat"`, else `instance_dim`

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 4.2 Run `python -m compileall -q features/segmenter.py mil/utils.py mil/training.py`
- [x] 4.3 Run `python scripts/estimate_cache_memory.py` to confirm concat config memory is reasonable
