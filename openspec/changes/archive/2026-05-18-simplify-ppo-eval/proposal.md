## Why

`ppo/eval.py` has ~50 lines of code that duplicate logic already present elsewhere: feature extraction from logprob tensors (same as `ppo/training.py`), `_render_prompt` (same as `VLLMFeatureExporter.render_messages`), and dead helpers (`load_prompts`, `load_config`, `_get_question`). Cleaning these up removes duplication and eliminates stale code.

## What Changes

- Extract duplicated feature construction into shared helper in `features/segmenter.py`
- Replace `_render_prompt` with `runner.build_math_messages()` + `runner.render_messages()`
- Delete `load_prompts` (dead), inline `load_config` and `_get_question`
- Remove `OnlineResult.errors` (never set)
- Fix `--parallel-size` CLI arg type to `int`

## Capabilities

### Modified Capabilities

- `ppo-online-generation`: eval uses shared feature extraction helper and runner's prompt rendering

## Impact

- `ppo/eval.py`: delete `_render_prompt`, `load_prompts`, `load_config`, `_get_question`, `errors` field
- `features/segmenter.py`: add shared `build_segment_obs_from_features` helper
- `ppo/training.py`: use shared helper for feature extraction
