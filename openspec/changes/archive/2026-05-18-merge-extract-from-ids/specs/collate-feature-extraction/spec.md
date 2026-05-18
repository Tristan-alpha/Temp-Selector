## ADDED Requirements

### Requirement: Unified extract_from_ids method

`VLLMFeatureExporter` SHALL provide a single `extract_from_ids` method that replaces `extract_logprobs_from_ids` and `extract_hidden_from_ids`. The method SHALL accept `return_logprobs: bool` and `return_hidden: bool` flags and return a dict with optional `"logprobs"` and `"hidden"` keys.

#### Scenario: Both logprobs and hidden requested

- **WHEN** `extract_from_ids(full_ids, prompt_lens, temperatures=temps, return_logprobs=True, return_hidden=True)` is called
- **THEN** a single `llm.generate()` call SHALL be made, and the returned dict SHALL contain both `"logprobs"` and `"hidden"` tensors

#### Scenario: Only logprobs requested

- **WHEN** `return_logprobs=True, return_hidden=False`
- **THEN** the returned dict SHALL contain only `"logprobs"` key

#### Scenario: Only hidden requested

- **WHEN** `return_logprobs=False, return_hidden=True`
- **THEN** the returned dict SHALL contain only `"hidden"` key

## REMOVED Requirements

### Requirement: extract_logprobs_from_ids

**Reason**: Merged into `extract_from_ids`
**Migration**: Use `extract_from_ids(full_ids, prompt_lens, temperatures=temps, return_logprobs=True)["logprobs"]`

### Requirement: extract_hidden_from_ids

**Reason**: Merged into `extract_from_ids`
**Migration**: Use `extract_from_ids(full_ids, prompt_lens, return_hidden=True)["hidden"]`
