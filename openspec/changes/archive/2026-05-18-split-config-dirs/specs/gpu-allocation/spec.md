## MODIFIED Requirements

### Requirement: CLI-only parallel_size

Config files SHALL be organized in two subdirectories: `configs/dataset/` for dataset generation and `configs/training/` for MIL/PPO training. Dataset configs SHALL include a `split:` section with `val_ratio`, `test_ratio`, and `seed` keys. `build_dataset.py` SHALL read split parameters from config, with CLI args as overrides.

#### Scenario: Split params from config

- **WHEN** `build_dataset.py` is invoked with `--config configs/dataset/full.yaml`
- **THEN** `val_ratio`, `test_ratio`, and `split_seed` SHALL be read from the config's `split:` section, with CLI `--val-ratio` / `--test-ratio` / `--split-seed` overriding if provided
