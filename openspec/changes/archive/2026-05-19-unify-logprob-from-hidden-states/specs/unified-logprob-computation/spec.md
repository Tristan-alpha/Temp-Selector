## ADDED Requirements

### Requirement: generate_with_features computes logprobs from hidden states

`generate_with_features` SHALL compute per-token logprobs from hidden states via `apply_model` + `_LogprobsComputeFn`, the same mechanism used by `extract_from_ids`. It SHALL NOT pass `logprobs=` to `SamplingParams`. It SHALL accept `return_logprobs: bool` and `device: torch.device | None` parameters, mirroring `extract_from_ids`. Logprobs SHALL only be computed and returned when `return_logprobs=True`.

#### Scenario: Logprobs from hidden states

- **WHEN** `generate_with_features(..., return_logprobs=True)` is called with valid prompts and temperatures
- **THEN** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`, computed from hidden states via `apply_model`
- **AND** when `return_logprobs=False`, the `logprobs` key SHALL be `None`

#### Scenario: Hidden states read when any feature is needed

- **WHEN** `generate_with_features` is called with `return_logprobs=True` or `return_hidden=True`
- **THEN** hidden states SHALL be read from `kv_transfer_params.hidden_states_path` for every output
- **AND** when both flags are `False`, hidden states MAY be skipped

### Requirement: max_logprobs removed

`VLLMFeatureExporter.__init__` SHALL NOT accept a `max_logprobs` parameter. `LLM()` SHALL NOT receive `max_logprobs` in its kwargs.

#### Scenario: VLLMFeatureExporter instantiation

- **WHEN** `VLLMFeatureExporter(model_name_or_path="...", max_new_tokens=8192)` is constructed
- **THEN** construction SHALL succeed without a `max_logprobs` argument
- **AND** the vLLM `LLM` engine SHALL be initialized without `max_logprobs`

#### Scenario: SamplingParams without logprobs

- **WHEN** `generate_with_features` creates `SamplingParams` for generation
- **THEN** `SamplingParams` SHALL NOT include `logprobs=`
