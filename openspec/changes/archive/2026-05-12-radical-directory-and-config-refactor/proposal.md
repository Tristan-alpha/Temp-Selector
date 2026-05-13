## Why

The current directory structure scatters MIL and PPO code across `models/`, `training/`, `rl/`, and `eval/` directories with no clear module boundaries. The `configs/base.yaml` uses ambiguous section names (`model`, `training`) that don't distinguish MIL from PPO parameters. This causes actual bugs (e.g., the online evaluator reads `model.hidden_dim` instead of `ppo.policy_hidden_dim`) and makes the codebase harder to navigate. A radical reorganization that gives MIL and PPO each a self-contained module with corresponding config sections eliminates the ambiguity at the root.

## What Changes

- **BREAKING**: Delete `models/` directory — content moves to `mil/model.py` and `mil/temp_predictor.py`
- **BREAKING**: Delete `training/` directory — content moves to `mil/training.py`
- **BREAKING**: Rename `rl/` directory to `ppo/` — content redistributed into `ppo/model.py`, `ppo/training.py`, `ppo/eval.py`
- **BREAKING**: Move `eval/mil_eval.py` to `mil/eval.py` and `eval/online_evaluate.py` to `ppo/eval.py`
- **BREAKING**: Eliminate duplicate `token_to_vec()` implementations in `training/train_mil.py` and `rl/ppo_trainer.py` — single implementation in `features/segmenter.py`
- **BREAKING**: Restructure `configs/base.yaml`: lift `instance_dim` and `temp_bins` to `data` section; delete top-level `model` and `training` sections; create `mil: {model, training}` and `ppo: {model, training}` sections; rename `ppo.policy_hidden_dim` to `ppo.model.hidden_dim`
- Update all Python import paths, shell scripts, and test files to match new structure
- Rename `ep_labels` to `ep_correct` in PPO training to align with its actual semantics (1=correct, 0=wrong)
- Remove dead `[:inst_labels_cat.size(0)]` truncation in MIL eval and replace with proper alignment assertion

## Capabilities

### New Capabilities

- `mil-module`: Self-contained MIL module under `mil/` with model definition (`mil/model.py`), temperature predictor (`mil/temp_predictor.py`), training (`mil/training.py`), and evaluation (`mil/eval.py`)
- `ppo-module`: Self-contained PPO module under `ppo/` with policy/value model and helpers (`ppo/model.py`), online training (`ppo/training.py`), and online evaluation (`ppo/eval.py`)
- `config-schema-v2`: New config structure with `data` (shared), `inference` (shared), `mil: {model, training}`, and `ppo: {model, training}` sections
- `shared-feature-utils`: Unified `token_to_vec()` in `features/segmenter.py` eliminating code duplication

### Modified Capabilities

<!-- No existing specs to modify -->

## Impact

- **Config**: `configs/base.yaml` fully rewritten — all consumers must use new key paths
- **Python imports**: Every file that imports from `models.*`, `training.*`, `rl.*`, `eval.mil_eval`, or `eval.online_evaluate` must be updated
- **Shell scripts**: `scripts/run_pipeline.sh`, `scripts/stage2_mil.sh`, `scripts/stage3_ppo.sh`, `scripts/stage4_eval.sh` — module paths updated (`python -m training.train_mil` → `python -m mil.training`, etc.)
- **Tests**: 6 test files reorganized into 7 files with updated import paths
- **Checkpoints**: No impact — `torch.save`/`torch.load` uses state_dict keys, not class paths
- **External consumers**: None (no external dependencies on internal module layout)
