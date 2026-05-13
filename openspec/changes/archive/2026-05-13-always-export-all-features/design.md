## Context

`feature_mode` dispatch is unnecessary — the only in-memory features available from vLLM's standard offline inference API are logprob + entropy + topk_logits. Hidden states require a completely separate extraction path (speculative decoding + KV connector + disk I/O) that is impractical for batch dataset generation.

## Goals / Non-Goals

**Goals:**
- Remove `feature_mode` from configs and runner dispatch
- Always export logprob + entropy + topk_logits

**Non-Goals:**
- Implementing the speculative-decoding hidden state extraction path (future work)
- Removing `TokenFeature.hidden` field (kept for future use)
- Deleting `configs/hidden_states.yaml`

## Decisions

### D1: Runner simplification

```python
# Before: feature_mode dispatch
if feature_mode == "topk_logits":
    topk_logits = dist

# After: always set
topk_logits = dist  # always available from SamplingParams(logprobs=top_k)
```

### D2: Schema unchanged

`TokenFeature` keeps the `hidden` field (optional, defaults to None). Hidden states are available in vLLM 0.18 through a speculative-decoding + KV connector pipeline, but that requires a dedicated backend not implemented in this change.
