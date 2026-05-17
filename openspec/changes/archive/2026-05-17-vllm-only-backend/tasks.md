## 1. Delete SGLang code

- [x] 1.1 Delete `inference/sglang_runner.py`
- [x] 1.2 Delete `_extract_segment_obs_sglang` from `ppo/training.py`

## 2. Remove backend branching

- [x] 2.1 `mil/training.py`: remove `backend`, always VLLMFeatureExporter
- [x] 2.2 `mil/eval.py`: same
- [x] 2.3 `ppo/training.py`: remove backend param/CLI/branches
- [x] 2.4 `scripts/build_dataset.py`: remove SGLang branch + --backend

## 3. Config cleanup

- [x] 3.1 Remove `backend`, `parallel_size` from all configs
- [x] 3.2 Remove SGLang LD_LIBRARY_PATH from `run_pipeline.sh`

## 4. Docs

- [x] 4.1 Update `CLAUDE.md`

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — 128 passed
- [x] 5.2 Run `python -m compileall -q` on modified files
- [ ] 5.3 Delete verify scripts
