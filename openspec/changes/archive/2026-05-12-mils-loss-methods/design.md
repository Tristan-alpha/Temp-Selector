## Context

The current hard-coded `k = max(1, n_valid // 3)` in the MIL instance loss has no theoretical basis. Three alternatives from recent literature offer zero-hyperparameter approaches. Making this configurable enables A/B comparison within the same training pipeline.

## Goals / Non-Goals

**Goals:**
- Four instance loss methods selectable via config key
- Default changed from `topk` to `pure` in base.yaml
- Two new ablation configs
- All existing tests pass

**Non-Goals:**
- Changing bag_loss, temp_loss, or smoothness_loss
- Changing MILModel architecture
- Auto-selecting the best method

## Decisions

### D1: Config key `mil.training.instance_loss`

```yaml
mil:
  training:
    instance_loss: pure   # topk | pure | soft_pseudo_label | contrastive
```

Default is `pure` (k=1). The old behavior is preserved as `topk`.

### D2: Loss implementations

All variants use `beta_inst_aux` weighting (unchanged). Only the per-sample loss computation changes.

**topk** (unchanged):
```
positive bag: top-k (k=n_valid//3) → target=1, rest → target=0
negative bag: all → target=0
```

**pure** (k=1):
```
positive bag: top-1 → target=1, rest → target=0
negative bag: all → target=0
```
Identical to topk with k=1. Simple but theoretically grounded.

**soft_pseudo_label**:
```
positive bag: target = clamp(sigmoid(inst_logit).detach(), min=0.5 for max only)
negative bag: all → target=0
```
`.detach()` prevents gradient feedback loops. The min-clamp on the max-scoring instance ensures the MIL assumption never degenerates (at least one instance always has target ≥ 0.5).

**contrastive**:
```
positive bag: loss = logsumexp(scores) - max(scores)
negative bag: loss = scores.pow(2).mean()
```
Encourages one instance to stand out without specifying which one. The logsumexp term penalizes all high scores collectively; the -max term rewards one instance being distinctly high.

### D3: New ablation configs

Two new files derive from base.yaml:
- `configs/instance_soft_pseudo_label.yaml`: `instance_loss: soft_pseudo_label`, dedicated output paths
- `configs/instance_contrastive.yaml`: `instance_loss: contrastive`, dedicated output paths

They differ from base only in `mil.training.instance_loss` and path prefixes.

## Risks / Trade-offs

- **[Behavior change]** Default switching from topk to pure may change MIL training outcomes → **Mitigation**: old behavior preserved as config option; if pure underperforms, switch back
- **[Contrastive loss scale]** Contrastive loss magnitude differs from BCE → **Mitigation**: `beta_inst_aux` can be tuned per-method, but the contrastive formulation is designed to produce comparable magnitudes

## Migration Plan

1. Add `instance_loss` key to `base.yaml` and 2 new ablation configs
2. Implement dispatching logic in `mil/training.py`
3. Implement `pure`, `soft_pseudo_label`, `contrastive` loss functions
4. Add tests
5. Update docs
