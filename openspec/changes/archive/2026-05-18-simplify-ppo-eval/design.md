## Context

After the recent runner refactor, `ppo/training.py` and `ppo/eval.py` both call `generate_with_features` and then build segment observations from the returned logprob tensor. The feature construction logic is identical (~25 lines each) but duplicated.

## Goals / Non-Goals

**Goals:**
- Single shared function for feature construction from `generate_with_features` output
- Use runner's prompt rendering
- Remove dead/unused code

**Non-Goals:**
- Changing the evaluation strategies
- Changing PPO training algorithm

## Decisions

### Decision 1: Shared helper in `features/segmenter.py`

The helper is a natural companion to `build_segments` and `segment_pooling` which already live there — it's a feature construction step, not an MIL utility.

```python
def build_segment_obs_from_lp(
    lp_tensor: torch.Tensor,
    tokens: List[str],
    text: str,
    segment_size: int,
    obs_dim: int,
    device: torch.device | None = None,
) -> torch.Tensor:
```

Computes: logprob + entropy → cat → pad/trunc → build_segments → segment_pooling.

### Decision 2: Use runner for prompt rendering

Replace `_render_prompt` with `runner.build_math_messages(question)` + `runner.render_messages(messages)`. The runner already has these methods and handles chat template fallback identically.
