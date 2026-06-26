## MODIFIED Requirements

### Requirement: Feature extraction is pre-compute once, not per-batch

MIL training SHALL call `extract_from_ids` only during the pre-computation phase (once before epoch loop). Training epochs SHALL use the RAM-cached segment features. The pre-computation phase SHALL still use `make_collate_fn` with the extractor; training epochs SHALL use `make_cached_collate_fn`.

`extract_from_ids` remains the correct tool for MIL pre-computation because it accepts pre-tokenized `full_ids` and returns logprobs + hidden states for arbitrary token sequences. The new single-pass path in `generate_with_features` is for online generation use cases (PPO training / eval) where tokens are being generated incrementally.

#### Scenario: No vLLM calls during training epochs

- **WHEN** MIL training enters the epoch loop
- **THEN** no call to `extract_from_ids` SHALL occur during collation
- **AND** all segment tensors SHALL come from the pre-computed cache

#### Scenario: extract_from_ids still used for pre-computation

- **WHEN** MIL training pre-computes segment features before the epoch loop
- **THEN** `extract_from_ids` SHALL be called via `make_collate_fn` with the VLLMFeatureExporter
- **AND** the two-pass flow inside `extract_from_ids` SHALL remain unchanged
