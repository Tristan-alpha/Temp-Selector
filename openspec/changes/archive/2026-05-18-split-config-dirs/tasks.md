## 1. Reorganize config directories

- [x] 1.1 Create `configs/dataset/` and `configs/training/` directories
- [x] 1.2 Move 4 dataset configs to `configs/dataset/`
- [x] 1.3 Move 9 training configs to `configs/training/`

## 2. Add split section to dataset configs

- [x] 2.1 Add `split: {val_ratio: 0.1, test_ratio: 0.1}` to 4 dataset configs; seed reused from global

## 3. Update build_dataset.py

- [x] 3.1 Read `val_ratio`, `test_ratio`, `split_seed` from config `split:` section; override with CLI args

## 4. Update scripts and docs

- [x] 4.1 Update `run_pipeline.sh` for new config paths
- [x] 4.2 Update `CLAUDE.md` for new config layout

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 5.2 Run `python -m compileall -q scripts/build_dataset.py`
