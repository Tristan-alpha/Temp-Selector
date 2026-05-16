## 1. Rewrite SGLangRunner (merge extractor)

- [x] 1.1 Rename `SGLangFeatureExporter` → `SGLangRunner`. Add `extract(prompts, responses) -> List[torch.Tensor]` method. Fix hidden state offset: `hs[prompt_len - 1:]` (h[i] → token[i+1]).
- [x] 1.2 vLLM runner: `__init__` raises `ValueError` if `feature_mode in {"hidden_states", "all"}`.

## 2. Rename tensor_parallel_size → parallel_size

- [x] 2.1 All configs: `tensor_parallel_size` → `parallel_size`.
- [x] 2.2 SGLangRunner / VLLMFeatureExporter: parameter renamed, internal mapping (SGLang → `dp_size`, vLLM → `tp_size`).
- [x] 2.3 All call sites: `build_dataset.py`, `ppo/training.py`, `mil/training.py`, `mil/eval.py` — read `parallel_size` from config.

## 3. Delete unused backends

- [x] 3.1 Delete `inference/api_runner.py`
- [x] 3.2 Delete `inference/vllm_hidden_extractor.py`
- [x] 3.3 Delete `inference/sglang_hidden_extractor.py`

## 4. Update build_dataset.py

- [x] 4.1 Remove api backend branch and `--backend api` CLI option.
- [x] 4.2 Remove vLLM hidden extraction block; import SGLangRunner.

## 5. Update PPO training

- [x] 5.1 Use `SGLangRunner` instead of bare `sglang.Engine` for SGLang path.
- [x] 5.2 Keep vLLM path (`--backend vllm`) with vllm_runner.

## 6. Update MIL training & eval

- [x] 6.1 `mil/training.py`: SGLangRunner replaces bare Engine + SGLangHiddenStateExtractor.
- [x] 6.2 `mil/eval.py`: same.

## 7. Config & docs

- [x] 7.1 Remove api config section from dataset.yaml.
- [x] 7.2 Update CLAUDE.md and PIPELINE.md.

## 8. Tests & verification

- [x] 8.1 `python -m pytest tests/ -v` — all pass.
- [x] 8.2 `python -m compileall -q` — all modified files compile.
