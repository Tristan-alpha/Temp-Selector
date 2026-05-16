## MODIFIED Requirements

### Requirement: Collate function extracts features per training batch

The MIL training collate function SHALL, for each training batch, invoke the SGLangRunner to extract per-token logprob features (and optionally hidden states), then compute per-segment instance vectors via `token_to_vec` and `segment_pooling`, and pad to a uniform batch tensor. Batch size SHALL be determined by a `TokenBatchSampler` that limits total tokens per batch to `max_tokens_per_batch` rather than a fixed bag count.

#### Scenario: Token-based batch sizing

- **WHEN** `TokenBatchSampler` accumulates sample indices with `max_tokens_per_batch=100000`
- **THEN** each yielded batch contains as many samples as fit without exceeding 100K total `(prompt + response)` tokens
- **AND** a single sample exceeding the limit is yielded alone in its own batch

#### Scenario: Deterministic eval batches

- **WHEN** `TokenBatchSampler` is used with `shuffle=False` for the val loader
- **THEN** batch composition is deterministic across epochs
