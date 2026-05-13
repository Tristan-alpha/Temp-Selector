## Why

1. `feature_mode` dispatch (`basic` / `topk_logits` / `hidden_states`) is unnecessary — the only in-memory features available from vLLM are logprob + entropy + topk_logits. Hidden states require a completely separate extraction path (speculative decoding + KV connector + disk I/O, per the official vLLM example) that is impractical for batch dataset generation.
2. Since all configs share Stage 1 data, there's no reason to selectively export features — always export everything that's available in-memory.

## What Changes

- Remove `feature_mode` key from `configs/dataset.yaml`
- Remove feature_mode dispatch from `vllm_runner.py` and `api_runner.py` — always export logprob + entropy + topk_logits
- Remove `feature_mode` parametrization from `scripts/build_dataset.py`
- `TokenFeature.hidden` field and `configs/hidden_states.yaml` are **kept** (hidden state extraction requires a dedicated speculative-decoding backend — future work)

## Capabilities

### New Capabilities

- `drop-feature-mode`: Remove feature_mode dispatch; always export all in-memory token features

## Impact

- **Configs**: `dataset.yaml` — remove `feature_mode` key
- **Code**: `inference/vllm_runner.py`, `inference/api_runner.py`, `scripts/build_dataset.py` — simplified dispatch
- **Docs**: README config table, PIPELINE.md
