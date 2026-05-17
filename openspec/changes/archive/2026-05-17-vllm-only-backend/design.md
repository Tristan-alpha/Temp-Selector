## Design

### Before / After

```
mil/training.py:
  if backend == "vllm": VLLMFeatureExporter(...)
  else:                 SGLangRunner(...)
→ VLLMFeatureExporter(...)  # always

ppo/training.py:
  if backend == "vllm": runner.generate_raw(...)
  else:                 runner.generate_raw(...)    # same interface!
→ runner.generate_raw(...)

  _extract_segment_obs()        # vLLM logprobs path
  _extract_segment_obs_sglang() # SGLang flat logprob path
→ _extract_segment_obs()        # only one

scripts/build_dataset.py:
  if backend == "vllm": VLLMFeatureExporter(...)
  else:                 SGLangRunner(...)
→ VLLMFeatureExporter(...)
```

### Config cleanup

Remove: `backend`, `parallel_size` (vLLM auto-detects), `base_gpu_id`.

### File deletions

- `inference/sglang_runner.py`

### Functions deleted

- `ppo/training.py::_extract_segment_obs_sglang`
- `ppo/training.py::train_ppo(backend=...)` parameter
- `scripts/build_dataset.py::build_dataset(backend=...)` parameter

### Unchanged

- `ppo/eval.py` — already vLLM-only (uses `LLM` directly)
- `inference/vllm_runner.py` — keeps `engine_preset` for prefill vs decode
