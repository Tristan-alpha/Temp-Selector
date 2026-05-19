## MODIFIED Requirements

### Requirement: generate_with_features computes logprobs from hidden states

`generate_with_features` SHALL obtain hidden states for generated tokens by performing a second vLLM prefill call with the full sequence (prompt + generated tokens) and delegating to `extract_from_ids`. It SHALL NOT attempt to read hidden states from the generation call's `kv_transfer_params`. It SHALL NOT pass `logprobs=` to `SamplingParams` in the generation call.

#### Scenario: Logprobs from hidden states via two-pass

- **WHEN** `generate_with_features(..., return_logprobs=True)` is called
- **THEN** Pass 1 SHALL generate tokens via `llm.generate(prompts, SamplingParams(max_tokens=segment_size))`
- **AND** Pass 2 SHALL compute logprobs by calling `self.extract_from_ids(full_ids, prompt_lens, temperatures, top_k, return_logprobs=True, return_hidden=False, device=device)`
- **AND** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`

#### Scenario: Hidden states via two-pass

- **WHEN** `generate_with_features(..., return_hidden=True)` is called
- **THEN** Pass 2 SHALL include `return_hidden=True` in the `extract_from_ids` call
- **AND** hidden states SHALL be converted to float32 only when populating the output dict
