## Why

PPO currently uses an 80/20 split of each iteration's randomly sampled training prompts for validation. The val set changes every iteration, so `val_value` fluctuations come from three sources: random split noise, topic difficulty variance, and actual policy improvement. Early stopping on this signal is unreliable.

## What Changes

- Load a fixed validation set once from `paths.val_dataset` at training start
- Run a separate rollout on this fixed set after each PPO update, using the current policy
- Compute `val_value` from the fixed set's trajectories, for stable early stopping
- Remove the per-iteration 80/20 split (`val_ratio` config key becomes unused)

## Capabilities

### Modified Capabilities

- `ppo-online-generation`: PPO validation now uses a fixed held-out dataset instead of a random split of training data.

## Impact

- `ppo/training.py`: Add fixed val set loading + per-iteration val rollout; remove 80/20 split
- `configs/training/base.yaml`: `val_ratio` marked deprecated
