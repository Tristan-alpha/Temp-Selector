## Context

`ppo/eval.py` line 84 reads `self.feature_mode` from config but never uses it. The eval loop at lines 165-191 hardcodes:

```python
feats = self.runner.generate_with_features(
    round_prompts, round_temps, self.segment_size,
    top_k=self.top_k_logprobs,
    return_logprobs=True,       # return_hidden never passed
)
...
obs = build_segment_obs_from_lp(
    ...,
    include_topk=True,          # hardcoded True
    # extra_parts never passed
)
```

Meanwhile `ppo/training.py` lines 83-84 and 243-269 respect `feature_mode`:

```python
hs_needed = feature_mode == "hidden_states"
...
feats = runner.generate_with_features(
    ...,
    return_hidden=hs_needed,
)
...
extra = [f["hidden_states"]] if f["hidden_states"] is not None else None
obs = build_segment_obs_from_lp(
    ...,
    extra_parts=extra,
    include_topk=(not hs_needed),
)
```

These two call sites are within the same project, use the same runner and helper, and serve the same purpose (build features → feed to policy). They should use identical arguments.

## Goals / Non-Goals

**Goals:**
- Make `_evaluate_strategy_batch` pass the same `return_hidden`, `extra_parts`, `include_topk` arguments as `train_ppo`, driven by `self.feature_mode`
- Preserve exact behavior when `feature_mode: topk_logprobs` (current default)

**Non-Goals:**
- Extracting a shared helper function (unnecessary abstraction for 2 call sites, 3 lines of logic)
- Changing training code
- Adding support for additional feature modes

## Decisions

### Decision 1: Mirror training logic inline vs extract shared helper

**Chosen: Mirror inline.** The logic is 3 lines:

```python
hs_needed = self.feature_mode == "hidden_states"
```

and two boolean values derived from it. Extracting a helper introduces a new function that two call sites import — more indirection than the logic justifies. The duplication is mechanical and unlikely to diverge.

### Decision 2: Where to put the derived flags

**Chosen: Compute in `__init__` and store as instance attributes.**

```python
self.hs_needed = self.feature_mode == "hidden_states"
```

This is computed once, consistent with how `self.instance_dim`, `self.pooling_mode` etc. are already stored. The `_evaluate_strategy_batch` method just reads `self.hs_needed`.

### Decision 3: What to pass as `device`

Training passes `device=device` to both `generate_with_features` and `build_segment_obs_from_lp`. Eval currently passes neither. For feature parity:

**Chosen: Skip `device` in eval.** The `device` parameter in `build_segment_obs_from_lp` is only used for moving the tensor; without it, operations stay on CPU. Eval doesn't have a dedicated training GPU and `.tolist()` immediately converts to Python lists anyway. Passing `device` would be a no-op change. Omitting it keeps the diff minimal.

## Risks / Trade-offs

- **Risk**: If someone introduces a third `feature_mode` (e.g., `"attention_maps"`), eval would again be inconsistent with training. **Mitigation**: A `ValueError` on unrecognized `feature_mode` in `__init__` would catch this early. Not in scope for this fix, but worth noting.
- **Trade-off**: Duplicating 3 lines of logic instead of extracting a helper. Acceptable for 2 call sites.
