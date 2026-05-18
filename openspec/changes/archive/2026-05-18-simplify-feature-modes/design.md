## Context

`feature_mode` was originally designed for an offline pipeline where all features were pre-computed and stored in JSONL. Since the move to online extraction, `basic` mode stores fake logprobs (-20.0) for every token — 80% of the JSONL data is useless padding. `all` mode is unused in any config. The pipeline now separates concerns: `build_dataset` only generates text; MIL/PPO training always extracts features online.

## Goals / Non-Goals

**Goals:**
- Remove `basic` and `all` feature modes
- Always configure speculative decode in `_lazy_init`
- `build_dataset` uses raw `vllm.LLM` — no speculative decode overhead
- JSONL format simplified: `token_ids` + `tokens` instead of `token_features` list
- Delete `GenerationOutput`, `_to_generation_output`, `_build_feature_payload`, `export_token_features_multi_temp`

**Non-Goals:**
- Changing the segment pooling or MIL model architecture
- Changing PPO algorithm

## Decisions

### Decision 1: Two feature modes

| mode | online extraction | instance_dim usage |
|------|------------------|-------------------|
| `topk_logprobs` | logprob + entropy + top-k via `extract_from_ids(return_logprobs=True)` | 2 + top_k = 4098 |
| `hidden_states` | hidden states via `extract_from_ids(return_hidden=True)` | hidden_dim = 4096 |

`feature_mode` is no longer passed to `VLLMFeatureExporter.__init__` — the runner always configures speculative decode.

### Decision 2: build_dataset uses raw vLLM

```python
from vllm import LLM, SamplingParams
llm = LLM(model=..., tensor_parallel_size=...)
params = [SamplingParams(n=V, temperature=T, max_tokens=...)]
outputs = llm.generate(prompts, params)
```

Returns raw `RequestOutput` — text, token_ids, tokens are extracted directly. No logprob requests, no speculative decode.

### Decision 3: Simplified JSONL format

```json
{
  "sample_id": "...",
  "prompt": "math problem",
  "response": "generated text",
  "label": 0,
  "temperature": 0.7,
  "token_ids": [123, 456],
  "tokens": ["Hello", " world"],
  "metadata": {"rendered_prompt": "...", "gold_answer": "...", ...}
}
```

`make_collate_fn` updated to read `token_ids`/`tokens` directly instead of `token_features` list.
