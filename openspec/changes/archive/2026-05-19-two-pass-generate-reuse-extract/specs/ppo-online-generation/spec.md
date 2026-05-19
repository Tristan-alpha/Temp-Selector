## MODIFIED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method that generates text and returns per-token logprob and hidden state tensors. The method SHALL use a two-pass approach: Pass 1 generates tokens, Pass 2 extracts features via `extract_from_ids` on the full prompt+generated sequence. Speculative decode SHALL always be configured. The method SHALL accept `prompts`, `temperatures`, `segment_size`, `top_k`, `return_logprobs`, `return_hidden`, `n`, and `device`. It SHALL return a list of dicts with keys `token_ids`, `tokens`, `text`, `all_texts`, `logprobs`, `hidden_states`, `finish_reason`.

#### Scenario: Generation with two-pass feature extraction

- **WHEN** `generate_with_features(prompts=["..."]*2, temperatures=[0.7, 0.3], segment_size=512, top_k=4096, return_logprobs=True)` is called
- **THEN** the first vLLM call SHALL generate `segment_size` tokens per prompt
- **AND** the second vLLM call SHALL extract features by passing full pre-tokenized sequences to `extract_from_ids`
- **AND** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`, computed from the full prefill hidden states
