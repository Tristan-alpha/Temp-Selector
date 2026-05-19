## Why

`generate_with_features` currently reads hidden states from `extract_hidden_states` speculative decode output, which only covers prefill tokens — not the generated tokens. This means logprobs and hidden states returned for generated tokens are wrong (they belong to prompt tokens instead). Additionally, the hidden-state-to-logprobs logic is duplicated between `generate_with_features` and `extract_from_ids`, and a dtype mismatch bug causes `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16`.

## What Changes

- `generate_with_features` adopts a two-pass architecture: Pass 1 generates tokens via `llm.generate(prompts, ...)`, Pass 2 reuses `self.extract_from_ids()` to get hidden states and compute logprobs from the full sequence (prompt + generated tokens, all prefill)
- Remove the duplicated hidden-state reading and logprob computation from `generate_with_features`
- Fix dtype mismatch by not calling `.float()` on hidden states before they enter `_LogprobsComputeFn`

## Capabilities

### Modified Capabilities

- `unified-logprob-computation`: `generate_with_features` now obtains logprobs via a two-pass approach that delegates to `extract_from_ids`, instead of attempting to compute logprobs from hidden states obtained during generation (which only covered prefill tokens).
- `ppo-online-generation`: `generate_with_features` method now performs two vLLM calls per segment — first to generate tokens, second to extract features from the full prefill sequence.

## Impact

- `inference/vllm_runner.py`: `generate_with_features` rewritten as two-pass; `_LogprobsComputeFn` dtype fix
