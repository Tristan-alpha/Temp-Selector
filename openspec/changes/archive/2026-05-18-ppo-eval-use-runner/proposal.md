## Why

`ppo/eval.py` creates a raw `vllm.LLM()` and manually parses logprob dicts via `_extract_segment_obs` — the same pattern just removed from `ppo/training.py`. It also duplicates `_resolve_tp` logic. Reusing `VLLMFeatureExporter` + `generate_with_features` deletes ~50 lines and makes eval consistent with training.

## What Changes

- Replace `vllm.LLM(...)` + `_resolve_tp` with `VLLMFeatureExporter(reserve_training_gpu=True)`
- Replace `llm.generate()` + `_extract_segment_obs` with `generate_with_features`
- Delete `_resolve_tp` (14 lines), `_extract_segment_obs` (26 lines)
- Update `_evaluate_strategy_batch` to use tensor-based feature extraction

## Capabilities

### Modified Capabilities

- `ppo-online-generation`: `ppo/eval.py` now uses `VLLMFeatureExporter` instead of raw `LLM`

## Impact

- `ppo/eval.py`: delete `_resolve_tp`, `_extract_segment_obs`; replace `LLM` with `VLLMFeatureExporter`
- `features/vectorizer.py`: `token_to_obs`, `compute_entropy`, `mean_pool_obs` may become dead code (only used by eval's old extraction path)
