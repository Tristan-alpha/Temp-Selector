## 1. Implementation

- [x] 1.1 Rewrite `_validate()` in `mil/training.py` to compute `inst_logit_separation` (per-bag mean inst_logit → average over error bags and correct bags → difference)
- [x] 1.2 Rename `best_val_acc` → `best_separation`; adjust early stop direction (higher = better, already correct)
- [x] 1.3 Update log messages: `val_acc` → `separation`

## 2. Verification

- [x] 2.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 2.2 Run `python -m compileall -q mil/training.py`

## 3. Documentation

- [x] 3.1 Update mil/DESIGN.md early stop section
- [x] 3.2 Update PIPELINE.md early stop description
