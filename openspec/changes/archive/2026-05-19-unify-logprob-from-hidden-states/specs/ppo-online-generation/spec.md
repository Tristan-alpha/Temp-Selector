## MODIFIED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method that generates text and returns per-token logprob and hidden state tensors. Logprobs SHALL be computed from hidden states via `apply_model`, not from vLLM's built-in `SamplingParams(logprobs=)`. Speculative decode SHALL always be configured. The method SHALL accept `prompts`, `temperatures`, `segment_size`, `top_k`, `return_logprobs`, `return_hidden`, `n`, and `device`. It SHALL return a list of dicts with keys `token_ids`, `tokens`, `text`, `all_texts`, `logprobs`, `hidden_states`, `finish_reason`.

#### Scenario: Generation with logprobs from hidden states

- **WHEN** `generate_with_features(prompts=["..."]*2, temperatures=[0.7, 0.3], segment_size=512, top_k=4096, return_logprobs=True)` is called
- **THEN** each returned dict SHALL contain `logprobs` as a `torch.Tensor` of shape `[n_tokens, top_k+1]`, computed from hidden states via `apply_model`

#### Scenario: PPO eval uses runner

- **WHEN** `OnlineTemperatureEvaluator` is constructed
- **THEN** it SHALL create a `VLLMFeatureExporter` with `reserve_training_gpu=True`
