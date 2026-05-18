## 1. Simplify runner

- [x] 1.1 Remove `feature_mode` from `VLLMFeatureExporter.__init__`; always configure speculative decode in `_lazy_init`
- [x] 1.2 Delete `export_token_features_multi_temp`, `_build_feature_payload`, `_to_generation_output`, `GenerationOutput`
- [x] 1.3 Remove `from features.schema import TokenFeature` from runner

## 2. build_dataset uses raw LLM

- [x] 2.1 Replace `VLLMFeatureExporter` with raw `vllm.LLM` + `SamplingParams` for generation
- [x] 2.2 Write simplified JSONL (token_ids + tokens, no token_features/BagSample)
- [x] 2.3 Delete hidden_states/all branch; remove `from features.schema import BagSample`

## 3. Simplify schema

- [x] 3.1 Remove `topk_logprobs` and `hidden` from `TokenFeature`
- [x] 3.2 Delete `BagSample` class and `BagSample.to_dict`

## 4. Update collate_fn and callers

- [x] 4.1 `mil/utils.py`: simplify feature_mode checks; read `token_ids`/`tokens` from new JSONL format
- [x] 4.2 `mil/training.py`: remove `feature_mode` from runner construction
- [x] 4.3 `mil/eval.py`: remove `feature_mode` from runner construction
- [x] 4.4 `ppo/training.py`: remove `feature_mode` from runner construction
- [x] 4.5 `ppo/eval.py`: remove `feature_mode` from runner construction

## 5. Update configs

- [x] 5.1 `dataset_small_10.yaml`: `basic` → `topk_logprobs`
- [x] 5.2 `dataset_small_100.yaml`: `basic` → `topk_logprobs`
- [x] 5.3 `dataset_small_500.yaml`: `basic` → `topk_logprobs`
- [x] 5.4 `base.yaml`: remove `feature_mode` key (now defaults to `topk_logprobs`)

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests must pass; update tests for new JSONL format
- [x] 6.2 Run `python -m compileall -q` on all modified files
