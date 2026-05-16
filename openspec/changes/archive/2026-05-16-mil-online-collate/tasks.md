## 1. Rewrite BagDataset as metadata-only store

- [x] 1.1 Strip BagDataset.__init__: remove SGLang extraction, _extract_and_pool, segment building, pooling — store rows as list[dict] with feature_mode/temp_bins/instance_dim/segment_* attrs for collate_fn
- [x] 1.2 Change __getitem__ to return raw row dict (delete RowTensor usage here)
- [x] 1.3 Delete RowTensor dataclass (no longer used)

## 2. Create collate_fn with per-batch extraction

- [x] 2.1 Add `make_collate_fn(extractor, feature_mode, instance_dim, segment_mode, segment_size, pooling_mode, temp_bins)` factory that returns a collate function
- [x] 2.2 Implement collate_fn: for feature_mode requiring extraction, call extractor.extract_logprobs / extractor.extract_hidden, patch token_features, build segments, pool, pad
- [x] 2.3 Handle feature_mode="basic" path (no SGLang calls, build from existing token features)
- [x] 2.4 Delete old collate_rows function (replaced by collate_fn)

## 3. Update MIL training entry point

- [x] 3.1 Update train() to pass extractor to make_collate_fn instead of BagDataset, set num_workers=0
- [x] 3.2 Update val DataLoader to use same collate_fn, num_workers=0

## 4. Update MIL eval

- [x] 4.1 Update mil/eval.py evaluate_mil() to create collate_fn from runner and pass to DataLoader

## 5. Update config

- [x] 5.1 Remove `hidden_batch_size` from all config YAML files (mil.training section)
- [x] 5.2 Set `mil.training.batch_size` to 128 in base.yaml

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests must pass (126 passed)
- [x] 6.2 Run `python -m compileall -q mil/training.py mil/eval.py` — syntax check passed
- [x] 6.3 Update CLAUDE.md if any changed conventions (num_workers=0, collate_fn pattern)
