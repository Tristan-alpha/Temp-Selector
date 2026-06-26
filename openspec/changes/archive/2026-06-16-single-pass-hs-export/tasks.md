## 1. Core: Single-pass hidden state export

- [x] 1.1 Add `os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")` at the top of `VLLMFeatureExporter._lazy_init`, before any vLLM import.
- [x] 1.2 In `generate_with_features`, add `extra_args={"kv_transfer_params": {"include_output_tokens": True}}` to Pass 1 `SamplingParams`.
- [x] 1.3 Replace the `safe_open` + manual `os.remove` with `load_hidden_states` / `cleanup_hidden_states` from `example_hidden_states_connector` to correctly synchronize on async disk writes.
- [x] 1.4 Move the logprob computation (`compute_topk_logprobs` via `_LogprobsComputeFn` / `apply_model`) from `extract_from_ids` to `generate_with_features`, operating on the single-pass hidden states.
- [x] 1.5 Remove the Pass 2 call to `extract_from_ids` inside `generate_with_features` — the function now returns after computing logprobs from Pass 1 hidden states.

## 2. Cleanup

- [x] 2.1 `safe_open` still used by `extract_from_ids` (MIL pre-computation path) — kept.
- [x] 2.2 Keep `extract_from_ids` as a standalone method — it is still used by MIL training pre-computation via `make_collate_fn`.

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — 157 passed.
- [ ] 3.2 Run `CUDA_VISIBLE_DEVICES=0 python scripts/verify_hidden_states.py --model <path> --max-tokens 64` — confirm `resp_hs_available=64/64` with `include_output_tokens=True`.
- [x] 3.3 Run `python -m compileall -q inference/vllm_runner.py` — syntax OK.
- [x] 3.4 Updated `features/DESIGN.md` line 24 to document single-pass extraction.
