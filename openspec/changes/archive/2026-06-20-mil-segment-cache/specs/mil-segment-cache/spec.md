## ADDED Requirements

### Requirement: Cache path is derived from config and split name

The system SHALL construct a cache file path using the pattern:

```
datasets/cache/{split}-{segment_mode}-{pooling_mode}-{feature_mode}-{instance_dim}-{segment_size}.pt
```

All components are read from the training config at runtime. Dashes separate components; underscores within values (e.g., `fixed_window`, `topk_logprobs`) are preserved.

#### Scenario: Cache path for concat pooling config

- **WHEN** split="train", segment_mode="fixed_window", pooling_mode="concat", feature_mode="topk_logprobs", instance_dim=64, segment_size=64
- **THEN** the cache path is `datasets/cache/train-fixed_window-concat-topk_logprobs-64-64.pt`

#### Scenario: Cache path for mean pooling config

- **WHEN** split="val", segment_mode="fixed_window", pooling_mode="mean", feature_mode="topk_logprobs", instance_dim=64, segment_size=32
- **THEN** the cache path is `datasets/cache/val-fixed_window-mean-topk_logprobs-64-32.pt`

### Requirement: MIL training loads from cache on hit

The MIL training loop SHALL, before running vLLM extraction for a given split, check whether the cache file exists. If it does, the training loop SHALL load the segment features from the cache file and skip vLLM extraction entirely for that split.

#### Scenario: Cache hit during training

- **WHEN** `train()` is called and `datasets/cache/train-fixed_window-concat-topk_logprobs-64-64.pt` exists
- **THEN** segment features are loaded via `torch.load` from the cache file
- **AND** the vLLM `extract_from_ids` call is skipped for the training set

#### Scenario: Cache miss during training

- **WHEN** `train()` is called and the cache file does not exist
- **THEN** vLLM extraction runs normally
- **AND** the resulting segment features are saved via `torch.save` to the cache path

### Requirement: MIL evaluation loads from cache on hit

The MIL evaluation function SHALL also check for and use cache files following the same convention.

### Requirement: Cache directory is auto-created

The system SHALL create the `datasets/cache/` directory on first cache write if it does not already exist.
