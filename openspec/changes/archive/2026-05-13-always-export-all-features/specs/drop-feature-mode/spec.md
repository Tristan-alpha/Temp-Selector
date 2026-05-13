## ADDED Requirements

### Requirement: No feature_mode in configs
No config file SHALL contain a `feature_mode` key. Stage 1 always exports logprob, entropy, and topk_logits from in-memory vLLM outputs.

### Requirement: Runners always export all in-memory features
`vllm_runner.py` and `api_runner.py` SHALL always set `topk_logits` on every `TokenFeature`. No `feature_mode` dispatch logic remains.

#### Scenario: TokenFeature always has topk_logits
- **WHEN** any runner creates a TokenFeature
- **THEN** `topk_logits` is set to the top-k logprob distribution

### Requirement: build_dataset.py has no feature_mode references
`scripts/build_dataset.py` SHALL pass no `feature_mode` argument to any exporter method.

### Requirement: TokenFeature.hidden is preserved
`TokenFeature.hidden` field SHALL remain in the schema (optional, defaults to None). This is for a future hidden state extraction path using vLLM's speculative decoding + KV connector API — not part of this change.
