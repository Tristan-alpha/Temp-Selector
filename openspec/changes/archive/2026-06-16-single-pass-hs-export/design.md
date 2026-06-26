## Context

`VLLMFeatureExporter.generate_with_features` currently runs two `llm.generate()` calls per round: Pass 1 generates `segment_size` tokens with speculative decode (hidden states auto-saved to tempfiles but only covering prompt tokens), Pass 2 calls `extract_from_ids` on the full `prompt + generated` sequence to get hidden states for **all** tokens, then computes logprobs via `_LogprobsComputeFn` / `compute_topk_logprobs`.

vLLM's `example_hidden_states_connector` has an `include_output_tokens` feature (controlled via `SamplingParams.extra_args={"kv_transfer_params": {"include_output_tokens": True}}`). When set, the connector saves hidden states for `request.all_token_ids[:-1]` — i.e., prompt + all generated tokens except the very last. The verification script `scripts/verify_hidden_states.py` confirmed this gives exactly enough hidden states to compute logprobs for all response tokens (`hs[p_len-1:][:n_resp]` = 64/64).

## Goals / Non-Goals

**Goals:**
- Eliminate Pass 2 in `generate_with_features` — compute logprobs from Pass 1 hidden states directly.
- Keep `extract_from_ids` available for MIL training pre-computation (where it's called once, not per-round).
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` in `_lazy_init` so LLP can be used as a library without explicit env setup.
- Backward-compatible: MIL training, PPO training, and PPO eval must all work without config changes.

**Non-Goals:**
- Remove `extract_from_ids` entirely — MIL training still uses it for segment cache pre-computation.
- Change the PPO feature pipeline — `_process_generated_features` still calls `build_segment_obs_from_lp` as before.
- Support `num_speculative_tokens > 1` — the `extract_hidden_states` proposer requires exactly 1.

## Decisions

### Decision 1: Pass `include_output_tokens=True` via `SamplingParams.extra_args`

**Choice**: Add `extra_args={"kv_transfer_params": {"include_output_tokens": True}}` to Pass 1 `SamplingParams` in `generate_with_features`.

**Alternatives considered**:
- Set `include_output_tokens` globally via `kv_transfer_config` → reject, flag is per-request (affects token_ids saved), not per-connector-instance.
- Pass via a separate `llm.generate(extra_args=...)` parameter → not supported; only `SamplingParams.extra_args` flows through to `request.kv_transfer_params`.

### Decision 2: Use `load_hidden_states` / `cleanup_hidden_states` from connector

**Choice**: Import and use the connector's own `load_hidden_states` (which blocks on `flock(LOCK_SH)` until the async disk write finishes) and `cleanup_hidden_states` (which removes `.safetensors` + `.lock`).

**Alternatives considered**:
- Direct `safe_open` + `os.remove` → fails with `FileNotFoundError` when async write hasn't completed yet.
- Busy-wait with `os.path.exists` → fragile, no synchronization guarantee.

### Decision 3: Keep `extract_from_ids` for MIL training

**Choice**: Do NOT remove `extract_from_ids`. MIL training's `make_collate_fn` calls it once per pre-computation batch (not per training round), so the latency benefit of single-pass is negligible there. Keeping it avoids touching the MIL data pipeline.

**Alternatives considered**:
- Also convert MIL pre-computation to single-pass → requires changing `collate_fn` to pass `extra_args` through the batch row dicts, which is a larger change with no performance benefit (pre-computation is one-time, not per-iteration).

### Decision 4: Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` in `_lazy_init`

**Choice**: Add `os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")` at the top of `_lazy_init`, before any vLLM engine core fork. The default in `vllm/envs.py` is `"fork"` which crashes CUDA.

**Alternatives considered**:
- Require users to set it themselves → fragile, already caused issues in verification script.
- Set in pipeline entry scripts → doesn't cover all usage patterns.

## Risks / Trade-offs

- **Async disk write race**: `load_hidden_states` blocks on `flock` → safe. However, if the writer process crashes before releasing the lock, the reader hangs. Mitigation: the connector cleans up lock fds in `get_finished` for aborted requests. Unlikely in normal operation.
- **Last token logprob missing**: `all_token_ids[:-1]` excludes the final generated token's hidden state. This token was never input to a forward pass, so there's no `hs[last]` to predict `token[last+1]`. All response-token logprobs are still available → no practical impact.
- **Hidden state shape**: The tensor from `load_hidden_states` is `[seq_len, 1, hidden_dim]` (with the extra dim-1 from speculative decode). The existing slicing `hs[:, -1, :]` handles this correctly.

## Migration Plan

1. Update `generate_with_features` to use single-pass.
2. Add `VLLM_WORKER_MULTIPROC_METHOD` guard in `_lazy_init`.
3. Verify all existing tests pass.
4. Run `scripts/verify_hidden_states.py` to confirm single-pass works with the target model.
5. No config changes, no API changes. Rollback: revert `generate_with_features` to two-pass.
