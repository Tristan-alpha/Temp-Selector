## 1. Merge methods in vllm_runner.py

- [x] 1.1 Create `extract_from_ids` method merging the shared `llm.generate()` + safetensors read + shared per-sample loop
- [x] 1.2 Delete `extract_logprobs_from_ids` and `extract_hidden_from_ids`

## 2. Update callers

- [x] 2.1 `mil/training.py`: update collate_fn to use `extract_from_ids` with flags
- [x] 2.2 `mil/eval.py`: update collate_fn to use `extract_from_ids` with flags (imports make_collate_fn from mil/training.py — no changes needed)

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 3.2 Run `python -m compileall -q` on all modified files
- [x] 3.3 Check whether docs need updating (no references to old method names found — skip)
