## 1. Config

- [x] 1.1 Add `instance_loss: pure` to `mil.training` in `configs/base.yaml`
- [x] 1.2 Create `configs/instance_soft_pseudo_label.yaml` (copy base, set `instance_loss: soft_pseudo_label`, dedicated paths)
- [x] 1.3 Create `configs/instance_contrastive.yaml` (copy base, set `instance_loss: contrastive`, dedicated paths)

## 2. Implementation

- [x] 2.1 Add `instance_loss` config read in `mil/training.py`
- [x] 2.2 Implement `pure` method (k=1) in the instance loss loop
- [x] 2.3 Implement `soft_pseudo_label` method with anti-degeneration clamp
- [x] 2.4 Implement `contrastive` method (logsumexp - max for pos, MSE for neg)
- [x] 2.5 Keep `topk` as fallback / default-unchanged path

## 3. Tests

- [x] 3.1 Add tests for `pure` method in `tests/test_mil_training.py`
- [x] 3.2 Add tests for `soft_pseudo_label` method
- [x] 3.3 Add tests for `contrastive` method
- [x] 3.4 Verify existing `topk` tests still pass

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 4.2 Run `python -m compileall -q` on modified files
- [x] 4.3 Verify all 3 configs parse: `python -c "import yaml; yaml.safe_load(open('configs/<name>.yaml'))"`

## 5. Documentation

- [x] 5.1 Update PIPELINE.md loss function section
- [x] 5.2 Update mil/DESIGN.md instance loss section
- [x] 5.3 Add config variants to README.md table
