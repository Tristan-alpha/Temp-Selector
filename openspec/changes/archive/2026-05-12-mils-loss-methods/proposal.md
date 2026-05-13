## Why

The current MIL instance loss uses a hard-coded `k = max(1, n_valid // 3)` to select which segments in a wrong answer are labeled as errors. This hyperparameter has no theoretical justification beyond empirical intuition. Three alternative methods from recent literature (2024-2025) offer zero-hyperparameter approaches: pure MIL (k=1) based on [FocusMIL (Liu et al., 2024)](https://arxiv.org/abs/2408.09449), soft pseudo-label self-training based on [SeLa-MIL (Ma et al., 2024)](https://arxiv.org/abs/2408.04813), and contrastive instance loss based on [NDI-MIL (IEEE TNNLS, 2025)](https://ieeexplore.ieee.org). Making the instance loss method configurable enables systematic ablation.

## What Changes

- Add `mil.training.instance_loss` config key with values `topk` (current), `pure` (k=1), `soft_pseudo_label`, `contrastive`
- Set `mil.training.instance_loss: pure` as default in `base.yaml`
- Implement all four variants in `mil/training.py` under a dispatcher
- Add `configs/instance_soft_pseudo_label.yaml` and `configs/instance_contrastive.yaml` ablation configs
- Add tests for each variant
- Update PIPELINE.md and mil/DESIGN.md

## Capabilities

### New Capabilities

- `instance-loss-methods`: Configurable MIL instance loss with four methods: topk (legacy), pure (k=1), soft pseudo-label, contrastive

## Impact

- **Code**: `mil/training.py` (loss computation), `configs/base.yaml` + 2 new configs, tests
- **Behavior**: Default changes from `topk (k=n_valid//3)` to `pure (k=1)` in base config
- **Docs**: PIPELINE.md loss section, mil/DESIGN.md
