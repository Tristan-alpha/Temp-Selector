## ADDED Requirements

### Requirement: Feature extraction is pre-compute once, not per-batch

MIL training SHALL call `extract_from_ids` only during the pre-computation phase (once before epoch loop). Training epochs SHALL use the RAM-cached segment features. The pre-computation phase SHALL still use `make_collate_fn` with the extractor; training epochs SHALL use `make_cached_collate_fn`.

#### Scenario: No vLLM calls during training epochs

- **WHEN** MIL training enters the epoch loop
- **THEN** no call to `extract_from_ids` SHALL occur during collation
- **AND** all segment tensors SHALL come from the pre-computed cache

## MODIFIED Requirements

### Requirement: engine is provided to collate_fn, not BagDataset

The VLLMFeatureExporter SHALL be passed to `make_collate_fn` during pre-computation (via `functools.partial`). During training epochs, `make_cached_collate_fn` SHALL NOT receive an extractor. `BagDataset.__init__` SHALL accept only `data_path` and no extractor parameter.

#### Scenario: Engine passed to collate_fn during pre-computation

- **WHEN** `train_mil()` pre-computes segment features
- **THEN** it SHALL use `make_collate_fn(extractor=runner, ...)` for the pre-computation DataLoader
- **AND** it SHALL switch to `make_cached_collate_fn(cache, ...)` for training DataLoader
