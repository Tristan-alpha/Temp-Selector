## 1. Update VLLMFeatureExporter

- [x] 1.1 LLM constructor: speculative config always on when `feature_mode != "basic"`
- [x] 1.2 Rewrite `extract_logprobs_from_ids` — hidden states → apply_model → compute_logits → compute_topk_logprobs
- [x] 1.3 `extract_hidden_from_ids` kept — reads from safetensors (independent generate call, shared with logprobs when both needed)
- [x] 1.4 Temperature support: `/ temperature` before `compute_topk_logprobs` in worker
- [x] 1.5 Clean up: remove `prompt_logprobs`-based extraction, remove `logprobs_mode` from LLM kwargs

## 2. Shell scripts

- [x] 2.1 `scripts/run_pipeline.sh`: `export VLLM_ALLOW_INSECURE_SERIALIZATION=1`

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — 128 passed
- [ ] 3.2 Run `python -m compileall -q` on modified files
- [ ] 3.3 Delete verify scripts after confirming extraction works
