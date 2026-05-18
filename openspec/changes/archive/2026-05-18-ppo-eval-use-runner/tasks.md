## 1. Replace LLM with VLLMFeatureExporter

- [x] 1.1 Replace `vllm.LLM(...)` with `VLLMFeatureExporter(..., reserve_training_gpu=True)`
- [x] 1.2 Delete `_resolve_tp` static method

## 2. Replace manual extraction with generate_with_features

- [x] 2.1 Replace `llm.generate()` + `_extract_segment_obs` with `generate_with_features` + `segment_pooling`
- [x] 2.2 Delete `_extract_segment_obs` method

## 3. Clean up

- [x] 3.1 Remove unused imports (`token_to_obs`, `compute_entropy`, `mean_pool_obs`, `vllm.LLM`)
- [x] 3.2 Check `features/vectorizer.py` — remove dead functions if no callers remain

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 4.2 Run `python -m compileall -q ppo/eval.py`
