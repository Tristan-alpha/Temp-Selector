## Context

The MIL validation metric currently uses `bag_accuracy`. The true objective is `inst_logit` quality. `inst_logit_separation` directly measures whether error-bag segments score higher than correct-bag segments.

## Goals / Non-Goals

**Goal:** Switch early stop metric from bag_accuracy to inst_logit_separation.

**Non-Goal:** Changing any other aspect of training.

## Decisions

### D1: Per-bag averaging

Compute the mean inst_logit per bag first, then average across bags. This ensures each bag contributes equally regardless of segment count.

```python
for each batch:
    for each sample i:
        n_valid = mask[i].sum()
        bag_mean = inst_logit[i, :n_valid].mean()
        if label[i] > 0.5: pos_means.append(bag_mean)
        else:              neg_means.append(bag_mean)

separation = mean(pos_means) - mean(neg_means)
```

### D2: Metric direction

Higher separation = better. The early stop tracks the maximum seen; stops when no new max for `patience` epochs.

## Risks

- **[Separation can be noisy]** Per-epoch separation estimates have variance → the existing patience mechanism (5 epochs) handles this naturally
