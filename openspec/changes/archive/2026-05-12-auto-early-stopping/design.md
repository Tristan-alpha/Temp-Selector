## Context

Two related changes: (1) the current train/eval split must become train/val/test so that early stopping and final evaluation use disjoint data; (2) both MIL and PPO training loops need early stopping.

## Goals / Non-Goals

**Goals:**
- Three-way group-aware split: train (~80%), val (~10%), test (~10%)
- MIL early stop on `val.jsonl` bag_accuracy
- PPO early stop on `val_value`
- Best checkpoint retained (not last)
- Config keys updated; old keys removed

**Non-Goals:**
- Changing model architecture, optimizer, or loss

## Decisions

### D1: Split ratios

```bash
python scripts/split_jsonl.py --config configs/base.yaml \
    --val-ratio 0.1 --test-ratio 0.1 --group-by sample_prefix --seed 42
```

Outputs: `datasets/train.jsonl` (~80%), `datasets/val.jsonl` (~10%), `datasets/test.jsonl` (~10%).

All splits are group-aware: same prompt × 15 temps × 8 votes goes to exactly one split.

### D2: Config paths

```yaml
paths:
  train_dataset: datasets/train.jsonl
  val_dataset:   datasets/val.jsonl       # NEW — early stopping + best-temp selection
  test_dataset:  datasets/test.jsonl      # NEW — final evaluation only
  # eval_dataset removed
```

### D3: Split ownership

| Dataset | Used by |
|---|---|
| `train.jsonl` | `mil/training.py`, `ppo/training.py` (prompts for online rollout) |
| `val.jsonl` | `mil/training.py` (early stop), `ppo/training.py` + `ppo/eval.py` (best-fixed temp selection) |
| `test.jsonl` | `mil/eval.py` (final bag_accuracy, AUC, etc.), `features/dataset_eval.py` (final statistics) |

### D4: MIL early stop config

```yaml
mil:
  training:
    max_epochs: 50            # was epochs: 15
    early_stop_patience: 5    # NEW
```

After each epoch: load `val.jsonl`, compute `bag_accuracy`. If improved → save ckpt + reset patience. If not → `patience_counter += 1`. Stop when counter reaches patience.

### D5: PPO early stop config

```yaml
ppo:
  training:
    max_iterations: 200          # was iterations: 80
    early_stop_patience: 10      # NEW
```

`val_value` is already computed each iteration. Track best (minimum). If improved → save ckpt + reset patience. If not → `patience_counter += 1`.

### D6: Best-fixed temperature selection

Currently `load_temperature_labels(data_path)` reads from `data_path` which is the raw prompt JSONL. This function reads per-temperature correctness from a dataset JSONL. It should read from `val.jsonl` (not `test.jsonl`) since the best-fixed temperature is a training-time decision.

## Risks / Trade-offs

- **[Less training data]** 80/10/10 reduces MIL training data from 90% to 80% vs previous train/eval split → **Mitigation**: 80% of ~N×15 temps is still very large; early stopping prevents overfitting on the smaller training set
- **[Config breakage]** Key renames affect 7 configs and multiple scripts → all updated in one pass

## Migration Plan

1. Update `scripts/split_jsonl.py` for three-way split
2. Update `run_pipeline.sh` for three-way split invocation
3. Update all 7 configs
4. Add validation + early stop to `mil/training.py`
5. Add early stop to `ppo/training.py`
6. Update data consumers to use val/test appropriately
7. Run tests, compile check
8. Update docs
