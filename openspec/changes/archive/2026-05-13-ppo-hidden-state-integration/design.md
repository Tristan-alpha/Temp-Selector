## Context

`VLLMHiddenStateExtractor` extracts per-token hidden states by feeding text through vLLM prefill. PPO generates segment-by-segment. Each new segment can be extracted by passing the accumulated prefix + new segment through the extractor.

## Goals / Non-Goals

**Goals:**
- PPO `_extract_segment_obs` supports hidden state extraction via `feature_mode`
- Per-segment prefix accumulation and mean-pooling of hidden states
- `obs_dim` auto-aligned with `instance_dim`

**Non-Goals:**
- Changing MIL or eval code
- Changing config schema

## Decisions

### D1: Extract at `_extract_segment_obs` call site

Current call site (after `llm.generate`):
```python
if out0.logprobs:
    obs = _extract_segment_obs(new_tokens, out0.logprobs, obs_dim, top_k_logits)
    segment_obs[i] = torch.tensor(obs, dtype=torch.float32) if obs else None
```

New path adds hidden state extraction alongside:
```python
if out0.logprobs:
    obs = _extract_segment_obs(new_tokens, out0.logprobs, obs_dim, top_k_logits,
                               hidden_states=extracted_hs_for_this_segment)
```

### D2: Prefix accumulation

```python
ep_prefixes: List[str] = [rendered[i]] * N    # initial prefix = rendered prompt
...
# after generation:
ep_prefixes[i] += req_outputs[0].text          # append new segment
# before next extraction:
hs = extractor.extract([ep_prefixes[i]], [new_segment_text])
```

### D3: Mean-pool hidden states per segment

Each segment has M tokens, each with a 4096-dim hidden state. Mean-pool → [4096]. If `feature_mode` is `all`, this is concatenated with the standard logprob/topk_logits features (64-dim) → [4160].

### D4: Extractor lifecycle

Single `VLLMHiddenStateExtractor` instance created in `train_ppo()` if `feature_mode` requires it. This second LLM instance uses minimal GPU memory (prefill-only, max_tokens=1).

## Risks

- **[Latency]** Per-segment prefill adds overhead → **Mitigation**: Only enabled with `hidden_states` or `all` feature_mode
- **[Memory]** Two LLM instances → **Mitigation**: Extractor instance uses `gpu_memory_utilization=0.30`
