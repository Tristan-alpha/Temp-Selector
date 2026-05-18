## MODIFIED Requirements

### Requirement: Unified extract_from_ids method

`VLLMFeatureExporter` SHALL provide a single `extract_from_ids` method. The method SHALL accept `return_logprobs: bool`, `return_hidden: bool`, and `device: torch.device | None` parameters. Logprob computation SHALL be chunked and cat'd at the `extract_from_ids` level, not inside `_LogprobsComputeFn`. When `device` is provided, the cat SHALL happen on that device.

#### Scenario: Both logprobs and hidden requested with training GPU cat

- **WHEN** `extract_from_ids(full_ids, prompt_lens, temperatures=temps, return_logprobs=True, return_hidden=True, device=train_device)` is called
- **THEN** a single `llm.generate()` call SHALL be made, logprob chunks SHALL be cat'd on `train_device`, and the returned dict SHALL contain both `"logprobs"` and `"hidden"` tensors

#### Scenario: device default

- **WHEN** `extract_from_ids` is called without `device`
- **THEN** logprob chunks SHALL be cat'd on CPU
