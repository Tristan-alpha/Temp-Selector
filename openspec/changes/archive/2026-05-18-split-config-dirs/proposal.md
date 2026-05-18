## Why

Config files are mixed in one directory: dataset-generation configs (`dataset_small_10.yaml`, `dataset_full.yaml`) sit alongside training configs (`base.yaml`, ablation variants). They have completely different schemas (no `data`/`mil`/`ppo` keys in dataset configs). `val_ratio`/`test_ratio`/`split_seed` are also hardcoded as CLI defaults instead of config-driven.

## What Changes

- Move configs into two subdirectories: `configs/dataset/` and `configs/training/`
- Add `split:` section to dataset configs (`val_ratio`, `test_ratio`; `seed` from global)
- `build_dataset.py`: read split params from config, CLI args override
- Update `run_pipeline.sh` for new paths
- Update `CLAUDE.md` for new config layout

## Capabilities

### Modified Capabilities

- `gpu-allocation`: config paths updated (no behavior change)

## Impact

- `configs/`: 13 files moved into 2 subdirectories, + `split:` keys in dataset configs
- `scripts/build_dataset.py`: ~5 lines changed
- `scripts/run_pipeline.sh`: paths updated
- `CLAUDE.md`: config path references updated
