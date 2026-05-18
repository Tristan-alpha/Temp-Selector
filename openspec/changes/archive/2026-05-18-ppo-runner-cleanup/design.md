## Context

PPO generates text segment-by-segment: each step picks a temperature (from the policy) and generates `segment_size` tokens. The runner currently exposes `generate_raw` which returns raw vLLM `RequestOutput` objects. The caller manually parses `logprobs` dicts via `_extract_segment_obs` to build segment observation vectors.

This design has three flaws:
1. Hidden states are impossible — `generate_raw` doesn't configure speculative decode
2. The caller must understand vLLM internals (`Logprob` objects, dict structures)
3. Feature extraction logic (`_extract_segment_obs`) duplicates what the runner already does internally

## Goals / Non-Goals

**Goals:**
- Replace `generate_raw` with a method that returns ready-to-use tensors
- Enable hidden state extraction for PPO (speculative decode path)
- Simplify the PPO training loop
- Reserve training GPU (like MIL)

**Non-Goals:**
- Changing the PPO algorithm (GAE, clipping, reward structure)
- Changing the MIL runner interface

## Decisions

### Decision 1: New method `generate_with_features` instead of wrapping `generate_raw`

**Alternative:** Keep `generate_raw` but add post-processing in the caller.

**Rationale:** The runner already knows how to compute logprobs (via `_LogprobsComputeFn`) and read hidden states (via speculative decode). Exposing these as pre-computed tensors removes the need for callers to parse vLLM internals.

### Decision 2: Return per-token tensors, not per-segment pooled vectors

**Alternative:** Return segment-level pooled observation vectors directly.

**Rationale:** The PPO caller may need per-token information (e.g., for debugging or alternative pooling). Keep the method general; segment pooling stays in the caller (or mil/utils.py).

### Decision 3: Method signature

```python
def generate_with_features(
    self,
    prompts: List[str],
    temperatures: List[float],
    segment_size: int,
    top_k: int = 4096,
    return_hidden: bool = False,
) -> List[Dict[str, Any]]:
```

Returns per prompt:
```python
{
    "token_ids": List[int],
    "tokens": List[str],
    "text": str,
    "logprobs": torch.Tensor,        # [n_tokens, top_k+1] or None
    "hidden_states": torch.Tensor,   # [n_tokens, hidden_dim] or None
    "finish_reason": str | None,
}
```

Logprobs are computed inline via `apply_model(_LogprobsComputeFn)` from the hidden states. Hidden states come from speculative decode (already configured in `_lazy_init` when `feature_mode != "basic"`).

### Decision 4: Remove `generate_raw` entirely

**Rationale:** PPO was the only caller. The method has no other users.

## Risks / Trade-offs

- **PPO eval uses its own LLM instance** (`ppo/eval.py:103`): This doesn't use `VLLMFeatureExporter` at all — it creates a raw `LLM()` and has its own `_extract_segment_obs`. Will be addressed in a separate review per user's preference.
- **`token_to_vec` may become dead code**: Once `_extract_segment_obs` is deleted, `token_to_vec` has no callers. Evaluate during implementation.
