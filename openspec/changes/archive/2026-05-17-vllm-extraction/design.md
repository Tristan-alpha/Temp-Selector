## Design

### Interface — mirror SGLangRunner

```python
class VLLMFeatureExporter:
    # existing: __init__, generate_raw, tokenizer, export_token_features_*

    def extract_logprobs_from_ids(
        self, full_ids: List[List[int]], prompt_lens: List[int],
        temperatures: List[float] | None = None, top_k: int = 4096,
    ) -> List[torch.Tensor]:
        """Per-response top-k logprobs via prompt_logprobs=top_k."""

    def extract_hidden_from_ids(
        self, full_ids: List[List[int]], prompt_lens: List[int]
    ) -> List[torch.Tensor]:
        """Per-response hidden states via speculative extract_hidden_states."""
```

### Verified findings (vLLM 0.18.0, tfinder env, GPU 7)

| Feature | Result |
|---------|--------|
| `prompt_logprobs=4096` | ✓ Works with `LLM(max_logprobs=4096)` (default: 20) |
| `max_tokens=0` | ✗ vLLM requires ≥1; use `max_tokens=1` |
| Token ID input | ✓ `LLM.generate([full_ids], sp)` auto-detects `List[int]` |
| `prompt_logprobs` format | `List[Dict[int, Logprob]]`, `Logprob` has `.logprob`, `.rank`, `.decoded_token` |
| `prompt_logprobs[P:]` alignment | ✓ Verified: response token logprobs at `[P:]`, 1:1 match |
| Hidden states via SamplingParams | ✗ Not available in RequestOutput/CompletionOutput |

### Logprob extraction

```python
def extract_logprobs_from_ids(self, full_ids, prompt_lens, temperatures=None, top_k=4096):
    outputs = self._llm.generate(
        [full_ids],  # List[List[int]]
        SamplingParams(max_tokens=1, prompt_logprobs=top_k),
    )
    results = []
    for out, p_len in zip(outputs, prompt_lens):
        plp = out.prompt_logprobs  # List[Optional[Dict[int, Logprob]]]
        resp_slice = plp[p_len:]   # response portion
        # Convert to tensor: [[lp.logprob for lp in d.values()] sorted by rank? or just values]
        lp_tensor = ...
        results.append(lp_tensor)
    return results
```

**LLM constructor requirements**: `max_logprobs=top_k` (default 20, too low for our 4096).

### Hidden state extraction (speculative decode approach)

vLLM supports hidden states via speculative decoding with `"extract_hidden_states"` method:

```python
llm = LLM(
    model=model_path,
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": [last_layer_id],  # 1-indexed!
            }
        },
    },
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": "/tmp/hs"},
    },
)
```

Output: `kv_transfer_params["hidden_states_path"]` → safetensors file with `hidden_states` tensor `[seq_len, num_layers, hidden_dim]`.

**Layer indexing is 1-indexed**. Qwen3-8B has 36 layers; the last layer is `36`. (Proof: official example uses `[3, 18, 33, 36]` — 36 would be out of range for 0-indexed 0-35.)

**Verified (v0.18.0, tfinder env, GPU 7)**:
- Shape: `[seq_len, num_layers, hidden_dim]` (seq_len first, not layers first)
- Response slice: `hs[:, -1, :][P - 1 :]` → `[R+1, hidden_dim]` — verified correct
- File delete after `safe_open`: tensor stays valid in memory ✓
- Multi-round generate after file delete: works ✓ (0.024s/round)
- Batch support: each sample gets its own `.safetensors` file in shared tempdir ✓
- Crash safety: `atexit.register(cleanup)` removes tempdir on exit ✓

### Collate_fn integration

No changes to collate_fn — same `extractor` parameter accepts either `SGLangRunner` or `VLLMFeatureExporter`.
