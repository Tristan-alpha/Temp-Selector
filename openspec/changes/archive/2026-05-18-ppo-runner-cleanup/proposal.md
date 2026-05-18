## Why

PPO training has three problems inherited from the SGLang era: (1) manual TP computation duplicates `_resolve_parallel_size`, (2) no training GPU reservation — vLLM and training compete for the last GPU, (3) `generate_raw` leaks raw vLLM objects to callers, makes hidden states impossible, and forces manual logprob parsing via `_extract_segment_obs` (55 lines).

## What Changes

- **BREAKING**: Remove `generate_raw` from `VLLMFeatureExporter`
- **BREAKING**: Remove `_extract_segment_obs` from `ppo/training.py`
- Add `generate_with_features` method: takes prompts + temperatures + segment_size, returns per-token logprob/hidden tensors
- Remove manual TP computation in `train_ppo`; pass `reserve_training_gpu=True`
- Update PPO training loop to use `generate_with_features`
- Update `ppo/eval.py` similarly

## Capabilities

### New Capabilities

- `ppo-online-generation`: Segment-by-segment generation with inline feature extraction via `generate_with_features`

## Impact

- `inference/vllm_runner.py`: delete `generate_raw`, add `generate_with_features`
- `ppo/training.py`: delete `_extract_segment_obs`, update training loop, add `reserve_training_gpu=True`, remove manual TP
- `ppo/eval.py`: update to use new interface
- `features/vectorizer.py`: `token_to_vec` may become dead code (only used by `_extract_segment_obs`)
