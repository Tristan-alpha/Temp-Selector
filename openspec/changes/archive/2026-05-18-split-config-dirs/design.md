## Context

13 config YAML files live in one directory with two distinct schemas. Dataset configs have `paths` + `inference`; training configs have `data` + `mil` + `ppo` + `inference` + `paths`. Splitting clarifies usage and enables adding split parameters to dataset configs.

## Goals / Non-Goals

**Goals:**
- `configs/dataset/`: 4 configs for `build_dataset.py`
- `configs/training/`: 9 configs for MIL + PPO stages
- Dataset configs own `split:` section
- `build_dataset.py` reads split params from config

**Non-Goals:**
- Changing training config schema
- Refactoring config inheritance/hierarchy

## Decisions

### Decision 1: Flat subdirectory, no inheritance

**Rationale:** KISS — each config is self-contained. No YAML anchors/aliases. The 4 dataset configs already duplicate inference sections, and that's fine for 4 files.

### Decision 2: Split params in config, CLI overrides; seed from global

```python
split_cfg = cfg.get("split", {})
val_ratio = args.val_ratio if args.val_ratio is not None else split_cfg.get("val_ratio", 0.1)
test_ratio = args.test_ratio if args.test_ratio is not None else split_cfg.get("test_ratio", 0.1)
split_seed = args.split_seed if args.split_seed is not None else int(cfg.get("seed", 42))
```
