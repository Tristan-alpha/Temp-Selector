## MODIFIED Requirements

### Requirement: vLLM extraction uses tensor-based logprob computation

`VLLMFeatureExporter.extract_logprobs_from_ids` SHALL compute per-response-token top-k logprobs by reading hidden states from vLLM's speculative decode output, then calling `apply_model` with `model.compute_logits` + `compute_topk_logprobs` inside the worker. No Python `Logprob` object iteration SHALL occur in the extraction path.

#### Scenario: Tensor-based logprob extraction

- **WHEN** `extract_logprobs_from_ids(full_ids, prompt_lens, top_k=4096)` is called
- **THEN** hidden states are read from safetensors, sliced to response tokens, and logprobs are computed via `apply_model` as a `[R, 4097]` float32 tensor (col 0 = sampled, cols 1: = top-k)

#### Scenario: Speculative config only when needed

- **WHEN** `feature_mode="basic"` is used
- **THEN** the LLM is created WITHOUT speculative_config (no hidden state extraction overhead)
