## Why

`TokenFeature.hidden` is always `None` — vLLM's standard generation API does not expose per-token hidden states. We can extract them via a two-pass trick: generate normally (Pass 1), then feed the concatenated prompt+response back through vLLM's prefill with `speculative_config` (`extract_hidden_states`) to capture hidden states for ALL tokens (Pass 2). The prefill computes hidden states end-to-end, and the speculative method saves them to disk with zero additional GPU memory or compute overhead.

## What Changes

- Add `vllm_hidden_extractor.py` — a new file that takes a list of (prompt, response) pairs and returns per-token hidden states via the two-pass trick
- Integrate into `scripts/build_dataset.py`: after generation, run the hidden state extractor on prompt+response pairs, populate `TokenFeature.hidden`
- Four-tier `feature_mode` dispatch: `basic` / `topk_logits` / `hidden_states` / `all` — hidden state extraction triggered by `hidden_states` or `all`
- Update `configs/dataset.yaml` with hidden state layer configuration

## Capabilities

### New Capabilities

- `hidden-state-extraction`: Two-pass vLLM prefill trick extracts per-token final hidden states for generated responses

## Impact

- **New file**: `inference/vllm_hidden_extractor.py`
- **Modified**: `scripts/build_dataset.py`, `inference/vllm_runner.py`, `configs/dataset.yaml`
- **Data**: `TokenFeature.hidden` will now contain actual hidden state vectors for generated tokens when `feature_mode: all`
- **Performance**: Pass 2 adds ~one prefill forward pass per (prompt, response) pair; I/O overhead from safetensors write/read; significantly slower than basic/topk_logits mode
- **Docs**: PIPELINE.md feature extraction section
