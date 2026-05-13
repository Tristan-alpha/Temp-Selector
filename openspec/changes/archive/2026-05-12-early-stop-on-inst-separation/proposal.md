## Why

The current MIL early stopping uses `bag_accuracy` (from `bag_head`) as the validation metric. However, the ultimate goal of MIL training is to produce high-quality `inst_logit` for PPO shaping rewards. `bag_accuracy` is a proxy that does not directly measure instance-level discriminability. Switching to `inst_logit_separation` (mean inst_logit on error bags minus mean on correct bags) aligns the early stopping criterion with the true downstream objective.

## What Changes

- Replace `_validate()` in `mil/training.py`: compute `inst_logit_separation` instead of `bag_accuracy`
- Higher separation = better (positive means error segments score higher than correct segments)
- `best_val_acc` → `best_separation`; early stop when separation stops improving

## Capabilities

### New Capabilities

- `inst-early-stop-metric`: MIL early stopping uses inst_logit_separation instead of bag_accuracy

## Impact

- **Code**: `mil/training.py` `_validate()` function only
- **Config**: No changes
- **Docs**: mil/DESIGN.md early stop section
