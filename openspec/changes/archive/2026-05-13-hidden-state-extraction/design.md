## Context

vLLM prefill computes hidden states for all input tokens but doesn't expose them. The `extract_hidden_states` speculative method intercepts these during prefill and saves them to a safetensors file. By concatenating prompt+response and running it through prefill, we can capture hidden states for generated tokens.

## Goals / Non-Goals

**Goals:**
- Extract per-token final hidden state for generated response tokens via two-pass trick
- Four-tier feature_mode dispatch: `basic` / `topk_logits` / `hidden_states` / `all`
- Zero additional GPU memory (separate minimal LLM instance for Pass 2)
- Clean up temp files

**Non-Goals:**
- Extracting hidden states during PPO online eval (add later if needed)
- Supporting intermediate layer features (only last layer for now)
- Batch-optimizing the extraction (can be done later)

## Decisions

### D0: feature_mode four-tier dispatch

| mode | logprob | entropy | topk_logits | hidden_states |
|---|---|---|---|---|
| `basic` | ✓ | ✓ | | |
| `topk_logits` | ✓ | ✓ | ✓ | |
| `hidden_states` | ✓ | ✓ | | ✓ |
| `all` | ✓ | ✓ | ✓ | ✓ |

Runner dispatch: `if mode in {"topk_logits", "all"}: set topk_logits`. build_dataset: `if mode in {"hidden_states", "all"}: run extractor`.

## Decisions

### D1: Architecture

```
Pass 1: VLLMFeatureExporter.generate_batch(prompts, temps, num_votes)
          → token_ids, logprobs, text, topk_logits

Pass 2: VLLMHiddenStateExtractor.extract(prompt_response_pairs)
          → per-token hidden_states

Merge:  zip Pass 1 token positions with Pass 2 hidden states
          → populated TokenFeature.hidden
```

### D2: LLM sharing vs separate instances

Option A: Same LLM instance. Add speculative_config at init time. Pass 1 also runs with speculative overhead. Simple but may slow generation.

Option B: Two LLM instances. Pass 1 = clean LLM (fast generate). Pass 2 = LLM with speculative_config (prefill-only). Double GPU memory.

**Decision**: Option B. Pass 2 can use `gpu_memory_utilization` much lower (~0.1-0.2) since it only does prefill with max_tokens=1. The second instance uses minimal GPU memory.

### D3: Token position alignment

Pass 1 generated `N` tokens for each vote. Pass 2 feeds `concat(prompt, response_v)` and gets hidden states for all `prompt_len + N` tokens. Slice off the first `prompt_len` positions to get the response hidden states.

Token count might differ between prefill and generation due to tokenization differences. Validate by checking lengths match.

### D4: Layer selection

Qwen3-8B has 28 transformer layers (index 0-27). The final hidden state before the LM head is layer 27 (0-indexed). Config uses 1-indexed layer IDs:

```yaml
inference:
  eagle_aux_hidden_state_layer_ids: [28]  # last layer only
```

### D5: PPO online per-segment hidden state extraction

PPO generates segment-by-segment. Hidden states for each new segment can be extracted by pre-filling the accumulated prefix:

```
Round 0: generate seg₀ → prefill(prompt + seg₀)               → hidden states for seg₀
Round 1: generate seg₁ → prefill(prompt + seg₀ + seg₁)         → hidden states for seg₁
Round 2: generate seg₂ → prefill(prompt + seg₀ + seg₁ + seg₂)  → hidden states for seg₂
```

Each round extracts only the NEWLY ADDED token positions. Prefill cost grows linearly with prefix length, but prefill is orders of magnitude faster than auto-regressive decode. MIL trained with hidden states can warm-start PPO and provide shaping rewards at matching feature dimensions.

This change ships the reusable extractor. `build_dataset.py` is the first consumer. PPO integration is a follow-up.

## Risks

- **[Speed]** Pass 2 adds one prefill per (prompt, response) pair. For 48K samples, this is significant → **Mitigation**: Only enabled with `feature_mode: hidden_states` or `all`. Use `basic` or `topk_logits` for fast generation.
- **[Per-segment PPO latency]** Each segment round adds a prefill of the accumulated prefix → **Mitigation**: For PPO this is a follow-up; prefill is IO-bound not compute-bound at these token counts.
- **[Memory]** Two LLM instances → **Mitigation**: Pass 2 instance uses minimal GPU memory (no KV cache for decode, max_tokens=1)
- **[Token misalignment]** Raw text concatenation avoids chat template differences → **Mitigation**: Validate token count alignment.
- **[Disk I/O]** Safetensors per-request → **Mitigation**: Shared temp directory per batch, clean up after
