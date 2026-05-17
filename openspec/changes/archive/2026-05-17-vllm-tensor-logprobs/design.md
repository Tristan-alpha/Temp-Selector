## Design

### spec_config: always on (except basic)

```python
need_spec = self.feature_mode != "basic"
```

When `need_spec=True`, LLM created with `speculative_config={extract_hidden_states}`. Both hidden extraction and logprob computation use the same generate call's hidden states.

### extract_logprobs_from_ids (new)

```python
def extract_logprobs_from_ids(self, full_ids, prompt_lens, temperatures=None, top_k=4096):
    outputs = self._llm.generate(full_ids, [SamplingParams(max_tokens=1, top_p=1.0, top_k=0)])
    results = []
    for out, p_len in zip(outputs, prompt_lens):
        # Read hidden states from safetensors
        with safe_open(out.kv_transfer_params["hidden_states_path"], "pt") as f:
            hs = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
        os.remove(hs_path)
        # Slice response hidden states [R, hidden_dim]
        resp_hs = hs[p_len - 1:, -1, :][:num_response_tokens]  # TODO: need R
        # Compute logprobs via apply_model
        raw = self._llm.apply_model(
            _make_compute_fn(resp_hs.cpu(), token_ids, top_k)
        )[0]
        results.append(raw[0])  # logprobs tensor [R, top_k+1]
    return results
```

**Problem**: `apply_model` runs in worker, can only pass CPU tensors. Hidden states must be `.cpu()` first. Still faster than Python object iteration.

### extract_hidden_from_ids

Removed. When both hidden + logprobs needed, a single `extract_features_from_ids` method or the collate_fn calls extract_logprobs which internally reads hidden states. Hidden states from safetensors are already available.

Actually, simpler: keep `extract_hidden_from_ids` but make it read from the same safetensors file. Both methods share the same `out.kv_transfer_params["hidden_states_path"]`.

### Shell scripts

`VLLM_ALLOW_INSECURE_SERIALIZATION=1` in `run_pipeline.sh` so `apply_model` can pickle closures.

### Risks

- `apply_model` has ~20ms overhead (tested) — acceptable for MIL batch extraction
- `model.model.norm(h)` must be called before `model.compute_logits` (EAGLE hidden states are pre-norm)
