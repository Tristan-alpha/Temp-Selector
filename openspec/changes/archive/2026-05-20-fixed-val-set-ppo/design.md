## Context

Current PPO validation splits each iteration's 64 training prompts into 51 train / 13 val. The val set changes every iteration, making `val_value` unstable for early stopping.

## Goals / Non-Goals

**Goals:**
- Stable val signal via fixed held-out dataset
- Minimal overhead: small val set, reduced rollout size

**Non-Goals:**
- Changing training data loading
- Changing PPO update logic

## Decisions

### Decision 1: Fixed val set from `val_dataset`

**Choice**: Load prompts from `paths.val_dataset`, randomly select a fixed subset (e.g., 16) once at init. Run a val rollout after each PPO update.

**Why**: Same mechanism as MIL's best-temp search. Fixed questions → `val_value` changes only from policy improvement.

### Decision 2: Remove 80/20 split

**Choice**: All training rollout samples go to PPO update; no held-out split.

**Why**: The fixed val set replaces this function. `val_ratio` config key is kept but ignored (for backward compatibility).

### Decision 3: Val rollout runs in `torch.no_grad()`

**Choice**: Single forward pass with argmax actions (no sampling). No PPO update from val data.

**Why**: Fits in existing loop structure; consistent with the spirit of validation.
