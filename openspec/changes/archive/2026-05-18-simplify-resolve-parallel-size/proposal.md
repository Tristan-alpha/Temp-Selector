## Why

`_resolve_parallel_size` currently has three fallback layers (CUDA_VISIBLE_DEVICES → torch.cuda.device_count() → default 1), a vague `engine_preset` parameter that actually means "reserve one GPU for training", and `parallel_size` accepts three different types. Simplifying to direct `torch.cuda.device_count()` with explicit errors removes brittle env-var parsing and makes GPU allocation intent clear.

## What Changes

- **BREAKING**: Replace `engine_preset: str` parameter with `reserve_training_gpu: bool` in `VLLMFeatureExporter.__init__`
- **BREAKING**: `parallel_size` type changes from `int | str | None` to `int | None` (None = auto-detect all GPUs)
- Rewrite `_resolve_parallel_size`: use `torch.cuda.device_count()` only, raise `RuntimeError` if no GPUs available or if reservation leaves no GPUs
- Update callers: `mil/training.py`, `mil/eval.py`, `scripts/build_dataset.py`

## Capabilities

### New Capabilities

- `gpu-allocation`: GPU device detection and allocation for vLLM, including optional training-GPU reservation with explicit error handling

### Modified Capabilities

<!-- None — existing spec-level behavior is unchanged. -->

## Impact

- `inference/vllm_runner.py`: `__init__` signature, `_resolve_parallel_size` (~19 lines → ~12)
- `mil/training.py`: constructor call
- `mil/eval.py`: constructor call
- `scripts/build_dataset.py`: constructor call
- Not touching: `ppo/training.py`, `ppo/eval.py` (separate review later)
