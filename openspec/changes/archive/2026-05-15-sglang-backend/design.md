## Context

vLLM 0.18 requires two LLM instances for hidden state extraction (generation + `extract_hidden_states` speculative config). These two instances conflict on GPU resources and EngineCore multiprocessing, leading to unreliable behavior. SGLang natively supports `return_hidden_states=True` in a single engine, eliminating the dual-instance problem entirely.

## Goals / Non-Goals

**Goals:**
- Replace vLLM as the default inference backend with SGLang
- Single engine handles both generation and hidden state extraction
- Maintain the same hidden state output interface (`List[torch.Tensor]` per sample, native dtype bf16)
- Preserve the JSONL + safetensors sidecar dataset format unchanged
- Keep vLLM as a legacy `--backend vllm` option

**Non-Goals:**
- Rewrite the MIL or PPO training logic
- Change the dataset format or I/O layer
- Add multi-node / distributed SGLang support
- Remove vLLM entirely

## Decisions

### Decision 1: SGLang `Engine` (not `vllm serve` + HTTP)

**Chosen**: Use SGLang's Python `sglang.Engine` API directly in-process.

**Rationale**: The `Engine` class provides a synchronous `generate()` method that returns hidden states when `return_hidden_states=True`. No HTTP server, port management, or inter-process communication needed. This mirrors the existing vLLM `LLM` API pattern.

```python
engine = sglang.Engine(model_path=..., dtype="bfloat16", ...)
output = engine.generate(prompt, sampling_params, return_hidden_states=True)
hidden_states = output["meta_info"]["hidden_states"]  # [n_tokens, hidden_dim]
```

SGLang API reference: `sglang.Engine` accepts `model_path`, `tp_size`, `mem_fraction_static`, `max_prefill_tokens`, `context_length`.

### Decision 2: Mirror vLLM extractor interface

**Chosen**: `SGLangFeatureExporter` returns the same payload structure as `VLLMFeatureExporter`. Hidden states extracted from SGLang are returned as `List[torch.Tensor]` matching the vLLM extractor's output type.

**Rationale**: `build_dataset.py` and `ppo/training.py` already consume `List[torch.Tensor]` from the extractor. By keeping the same type, the backend switch is transparent to downstream code.

For the build script, extracting hidden states from SGLang output:
```python
# SGLang returns hidden_states per request
hs_tensor = output["meta_info"]["hidden_states"]  # [n_total_tokens, hidden_dim]
# Slice off prompt tokens (SGLang might include them)
response_hs = hs_tensor[prompt_len:]  # [n_response_tokens, hidden_dim]
```

### Decision 3: `mem_fraction_static` replaces `gpu_memory_utilization`

**Chosen**: Map config key `gpu_memory_utilization` to SGLang's `mem_fraction_static` parameter.

**Rationale**: Same semantics (fraction of GPU memory to reserve). Keep the config key name for backward compatibility, map internally.

### Decision 4: Multi-temperature batching

**Chosen**: SGLang handles multi-temperature generation through separate `generate()` calls per (prompt, temperature) pair. APC (automatic prefix caching) still applies within SGLang.

**Rationale**: vLLM's `export_token_features_multi_temp` interleaves temperatures for KV-cache sharing. SGLang's radix cache provides equivalent sharing automatically — same prompt at different temperatures shares the prefix cache. We can loop over temperatures and SGLang caches the prompt automatically.

### Decision 5: Remove PPO dual-instance hack

**Chosen**: In PPO training with SGLang backend, hidden states are obtained from the same `engine.generate()` call that generates the segment. No separate extractor object, no destroy/recreate/sleep logic.

**Rationale**: With `return_hidden_states=True`, each generation call already returns hidden states. The `_extract_segment_obs` function receives hidden states directly from the generation output. All dual-instance code paths (sleep/wake_up, destroy/recreate) are removed from the SGLang path.

## Risks / Trade-offs

- **New dependency**: `sglang` added to `requirements.txt`. SGLang is under active development but may have its own stability issues.
- **API differences**: SGLang's `SamplingParams` and token output format differ from vLLM's. Adapter code needed in `SGLangFeatureExporter`.
- **Memory**: Single engine with `return_hidden_states=True` may use slightly more GPU memory per request (hidden states kept in buffer until response is consumed). Mitigated by only enabling for the first vote (`V=1` for extraction).
- **Token alignment**: SGLang's hidden states may include prompt tokens. Need to verify with the actual API and slice appropriately.
