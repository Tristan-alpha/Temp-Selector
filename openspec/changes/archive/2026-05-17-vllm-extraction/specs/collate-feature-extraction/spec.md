## MODIFIED Requirements

### Requirement: Collate function extracts features per training batch

The MIL training collate function SHALL accept either a `SGLangRunner` or `VLLMFeatureExporter` as its extractor. `VLLMFeatureExporter` SHALL provide `extract_hidden_from_ids` and `extract_logprobs_from_ids` with the same signature as `SGLangRunner`, making the backend pluggable without collate_fn changes.

#### Scenario: vLLM as extraction backend

- **WHEN** `make_collate_fn(extractor=vllm_exporter, feature_mode="topk_logprobs")` is used
- **THEN** `extract_logprobs_from_ids(full_ids, prompt_lens, temperatures, top_k)` is called with the same parameters as SGLangRunner
- **AND** returns `List[torch.Tensor]` of shape `[n_response_tokens, top_k]`

#### Scenario: vLLM hidden state extraction

- **WHEN** `make_collate_fn(extractor=vllm_exporter, feature_mode="all")` is used
- **THEN** `extract_hidden_from_ids(full_ids, prompt_lens)` returns per-response-token hidden states
- **AND** `extract_logprobs_from_ids` also works
