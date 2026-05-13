## ADDED Requirements

### Requirement: All configs share base dataset paths
All 9 config files SHALL use identical `all_dataset`, `train_dataset`, `val_dataset`, `test_dataset` paths. Only `mil_ckpt` and `ppo_ckpt` paths are config-specific.

#### Scenario: All configs have same data paths
- **WHEN** loading any config
- **THEN** `paths.train_dataset` is `datasets/train.jsonl`

#### Scenario: Only ckpt paths differ
- **WHEN** comparing `base.yaml` and `arch_mlp_only.yaml`
- **THEN** dataset paths are identical; only `mil_ckpt` and `ppo_ckpt` differ (e.g., `checkpoints/mil_mlp_ckpt.pt`)

### Requirement: New ppo_control.yaml config
`configs/ppo_control.yaml` SHALL exist with `shaping_coef: 0.0` and shared dataset paths, for PPO terminal-reward-only control experiments.

#### Scenario: ppo_control loads
- **WHEN** loading `configs/ppo_control.yaml`
- **THEN** `ppo.training.shaping_coef` is 0.0
