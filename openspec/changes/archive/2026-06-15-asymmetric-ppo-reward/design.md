## Context

The current reward scheme applies the same L1-normalized MIL attention weights
to both positive and negative terminal rewards.  MIL attention is trained with
bag_bce only — it learns "which segment suggests this bag contains an error,"
not "which segment contributes most to correctness."

## Goals / Non-Goals

**Goals:**
- Incorrect chains receive attention-weighted negative reward
- Correct chains receive uniform positive reward
- MIL-absent fallback (uniform) unchanged

**Non-Goals:**
- Changing how MIL attention weights are computed
- Adding new hyperparameters

## Decisions

### Asymmetric distribution: attention for errors, uniform for correct

**Chosen**: `reward[t] = -1 × weights[t]` for errors, `reward[t] = +1 / n` for correct.

**Rationale**: MIL attention is an error-localization signal.  On incorrect chains
it provides meaningful credit assignment (which step was likely wrong).  On
correct chains the attention distribution has no ground-truth correlate — the
bag label alone doesn't identify which segment is "more correct" than others.
Uniform reward is the least-biased choice.

## Risks / Trade-offs

- **Weaker gradient on correct chains**: 1/n per step vs attention-weighted.
  Mitigated by the fact that correct chains outnumber incorrect ones in practice
  (majority voting already selects the correct answer more often), so the total
  signal volume is still significant.
