## Why

`feature_mode` currently has 4 values (`basic`, `topk_logprobs`, `hidden_states`, `all`) but `basic` stores fake logprobs (-20.0) and `all` is unused. `build_dataset` uses `VLLMFeatureExporter` which configures speculative decode unnecessarily for pure generation. Simplifying to 2 modes (`topk_logprobs`, `hidden_states`) with always-on online extraction eliminates fake data and makes the pipeline cleaner.

## What Changes

- **BREAKING**: Remove `basic` and `all` feature modes; keep only `topk_logprobs` and `hidden_states`
- Remove `feature_mode` from `VLLMFeatureExporter.__init__` — speculative decode is always configured
- Delete `export_token_features_multi_temp`, `_build_feature_payload`, `_to_generation_output`, `GenerationOutput` from runner
- `build_dataset.py`: use raw `vllm.LLM` for generation (no speculative decode), write simplified JSONL
- Simplify `TokenFeature` (drop `topk_logprobs`, `hidden`), delete `BagSample` from schema
- `make_collate_fn`: always do online extraction via `extract_from_ids`
- Update 3 configs: `basic` → `topk_logprobs`

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: always-on online extraction; 2 feature modes
- `ppo-online-generation`: `feature_mode` check simplified
- `gpu-allocation`: `_lazy_init` always configures speculative decode

## Impact

- `inference/vllm_runner.py`: delete ~110 lines, always speculative decode
- `scripts/build_dataset.py`: raw LLM, ~50 lines deleted
- `features/schema.py`: delete `BagSample`, trim `TokenFeature`
- `mil/utils.py`: simplify feature mode checks
- `configs/`: 3 files `basic` → `topk_logprobs`
- `ppo/training.py`, `ppo/eval.py`, `mil/training.py`, `mil/eval.py`: adapt to removed modes
