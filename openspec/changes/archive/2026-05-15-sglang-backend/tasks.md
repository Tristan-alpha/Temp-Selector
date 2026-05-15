## 1. SGLang backend implementation

- [x] 1.1 Add `sglang` to `requirements.txt`
- [x] 1.2 Create `inference/sglang_runner.py` with `SGLangFeatureExporter` class:
  - `__init__(model_path, max_new_tokens, tp_size, gpu_memory_utilization)` → creates `sglang.Engine`
  - `export_token_features_multi_temp(prompts, temperatures, feature_mode, top_k_logits, use_math_chat_prompt, system_prompt, num_votes)` → returns payload list matching vLLM format
  - `export_token_features_batch(...)` → single-temperature variant
  - Extract hidden states from `output["meta_info"]["hidden_states"]` when `return_hidden_states=True`
  - Map `gpu_memory_utilization` config key to SGLang's `mem_fraction_static`

## 2. Update existing callers

- [x] 2.1 Modify `scripts/build_dataset.py` to support `--backend sglang` (new default), keep `--backend vllm` as legacy
- [x] 2.2 Modify `ppo/training.py` to use SGLang as default backend:
  - Replace vLLM `LLM()` with `sglang.Engine()` for generation
  - Use `return_hidden_states=True` to get hidden states inline (no separate extractor)
  - Remove dual-instance hack code (sleep/wake_up, destroy/recreate)
  - Keep vLLM code path under `if backend == "vllm"`

## 3. Configuration

- [x] 3.1 Add `backend: sglang` to `configs/dataset.yaml` inference section
- [x] 3.2 Add `backend: sglang` to `configs/base.yaml` inference section
- [x] 3.3 Map `gpu_memory_utilization` → `mem_fraction_static` in SGLang runner

## 4. Tests & cleanup

- [x] 4.1 Add SGLang runner unit tests (mock engine, verify payload format)
- [x] 4.2 Run existing test suite — verify no regressions

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 5.2 Run `python -m compileall -q inference/sglang_runner.py scripts/build_dataset.py ppo/training.py`
- [x] 5.3 Update `CLAUDE.md` and `PIPELINE.md` for SGLang backend, new default, config changes
