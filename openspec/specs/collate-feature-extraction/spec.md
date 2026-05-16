## ADDED Requirements

### Requirement: Collate function extracts features per training batch

The MIL training collate function SHALL, for each training batch, invoke the SGLangRunner to extract per-token logprob features (and optionally hidden states), then compute per-segment instance vectors via `token_to_vec` and `segment_pooling`, and pad to a uniform batch tensor. Batch size SHALL be determined by a `TokenBatchSampler` that limits total tokens per batch to `max_tokens_per_batch` rather than a fixed bag count.

#### Scenario: Token-based batch sizing

- **WHEN** `TokenBatchSampler` accumulates sample indices with `max_tokens_per_batch=100000`
- **THEN** each yielded batch contains as many samples as fit without exceeding 100K total `(prompt + response)` tokens
- **AND** a single sample exceeding the limit is yielded alone in its own batch

#### Scenario: Deterministic eval batches

- **WHEN** `TokenBatchSampler` is used with `shuffle=False` for the val loader
- **THEN** batch composition is deterministic across epochs

#### Scenario: Collate with logprob extraction (topk_logprobs mode)

- **WHEN** collate_fn receives a batch of rows with `feature_mode="topk_logprobs"` or `"all"`
- **THEN** it calls `extractor.extract_logprobs(prompts, responses, temperatures=[...])` with per-sample temperatures
- **AND** passes extracted tensors to `token_to_vec` via the `extracted` parameter (NOT stored in row dicts)
- **AND** builds segment-level instance vectors via `token_to_vec` → `build_segments` → `segment_pooling`
- **AND** returns a padded batch dict

#### Scenario: Collate with basic mode (no extraction)

- **WHEN** collate_fn receives a batch with `feature_mode="basic"`
- **THEN** no SGLang call is made; instance vectors are built directly from existing token features (logprob, entropy)

### Requirement: Collate function receives extractor via closure

The collate function SHALL receive the SGLangRunner instance through `make_collate_fn` so it can be shared across train and val DataLoaders without duplicating the engine.

#### Scenario: Shared extractor across loaders

- **WHEN** `train_mil()` creates one SGLangRunner and two DataLoaders (train + val)
- **THEN** both loaders use `make_collate_fn(extractor=runner, ...)` with the same runner instance
- **AND** the runner's engine is kept alive until training completes
