## Why

SGLang was kept as an alternative backend but is no longer used — its prefill throughput is slower than vLLM, hidden state extraction requires IPC, and maintaining two backends adds `if backend == "vllm": ... else: ...` branches in every training/eval/build script. Dropping SGLang eliminates this complexity.

## What Changes

- **BREAKING**: Delete `inference/sglang_runner.py` (~400 lines)
- Remove all `backend` branching in `mil/training.py`, `mil/eval.py`, `ppo/training.py`, `scripts/build_dataset.py`
- Delete `_extract_segment_obs_sglang` from `ppo/training.py` (~60 lines)
- Remove `backend`, `base_gpu_id`, `parallel_size` from all config YAML files
- Remove `--backend` CLI arg from PPO training and build_dataset
- Remove SGLang-related env vars from `run_pipeline.sh`

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: VLLMFeatureExporter SHALL be the only extraction backend; collate_fn SHALL receive a VLLMFeatureExporter instance, not a generic extractor

## Impact

- Delete: `inference/sglang_runner.py`
- Modify: `mil/training.py`, `mil/eval.py`, `ppo/training.py`, `scripts/build_dataset.py`, `scripts/run_pipeline.sh`, all `configs/*.yaml`, `CLAUDE.md`
