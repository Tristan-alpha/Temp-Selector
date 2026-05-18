## MODIFIED Requirements

### Requirement: Unified extract_from_ids method

`VLLMFeatureExporter` SHALL provide a single `extract_from_ids` method. Online feature extraction SHALL always be active — `_lazy_init` SHALL always configure speculative decode. `make_collate_fn` SHALL always call `extract_from_ids` when an extractor is available, regardless of `feature_mode`.

#### Scenario: Always-on extraction

- **WHEN** `make_collate_fn` is called with an extractor
- **THEN** `extract_from_ids` SHALL be called to compute online features (logprobs for `topk_logprobs` mode, hidden states for `hidden_states` mode)

## REMOVED Requirements

### Requirement: feature_mode basic

**Reason**: Fake logprobs (-20.0) in JSONL replaced by always-on online extraction.

**Migration**: Use `feature_mode: topk_logprobs` and `extract_from_ids(return_logprobs=True)`.

### Requirement: feature_mode all

**Reason**: Unused in any config. Both modes can be achieved separately.

**Migration**: Use appropriate mode (`topk_logprobs` or `hidden_states`) depending on which features are needed.
