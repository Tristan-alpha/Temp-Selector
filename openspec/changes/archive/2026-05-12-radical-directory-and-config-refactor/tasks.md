## 1. Create new files and shared utilities

- [x] 1.1 Create `mil/__init__.py` and `ppo/__init__.py`
- [x] 1.2 Create `utils/math.py` with `safe_div`
- [x] 1.3 Create `features/vectorizer.py` with `token_to_vec`, `token_to_obs`, `mean_pool_obs`, `compute_entropy` (single source of truth, extracted from duplicated code in `training/train_mil.py`, `rl/ppo_trainer.py`, `eval/online_evaluate.py`)
- [x] 1.4 Move `features/answer_verifier.py` → `utils/answer_verifier.py`

## 2. Build MIL module

- [x] 2.1 Merge `models/mil.py` + `models/temp_predictor.py` → `mil/model.py` (single file: MILModel, InstanceEncoder, SinusoidalPositionalEncoding, AttentionAggregator, GlobalTempHead, DynamicTempHead, smoothness_loss)
- [x] 2.2 Move `eval/metrics.py` content into `mil/eval.py` (compute_bag_metrics, compute_calibration, compute_multiclass_metrics, compute_confusion_matrix, compute_auc, compute_attention_metrics) + `utils/math.py` (safe_div)
- [x] 2.3 Move `training/train_mil.py` → `mil/training.py` with changes:
  - Remove all DDP code: `DistributedDataParallel`, `DistributedSampler`, `setup_distributed`, `barrier`, `cleanup_distributed`, `emit_startup_self_check` imports and usage
  - Simplify device to `torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")`
  - Replace `DistributedSampler` with plain `shuffle=True`
  - Remove `use_ddp` logic entirely
  - Remove duplicate `token_to_vec`, add `from features.vectorizer import token_to_vec`
  - Update config key paths to `data.*` and `mil.*` paths
  - Update imports for temp heads: `from mil.model import GlobalTempHead, DynamicTempHead, smoothness_loss`
- [x] 2.4 Move `eval/mil_eval.py` → `mil/eval.py` with changes:
  - Append all metric functions from `eval/metrics.py` (step 2.2 merges them in)
  - Replace inline attention computation with call to `compute_attention_metrics`
  - Update imports: `mil.model` for MILModel and temp heads, `utils.math` for safe_div
  - Update config key reads to `mil.*` paths
  - Bug 4 fix already applied — preserved

## 3. Build PPO module

- [x] 3.1 Create `ppo/model.py` extracting `PolicyValueNet`, `compute_gae`, `sample_action`, `load_mil_encoder_for_warmstart` from `rl/ppo_trainer.py`
- [x] 3.2 Create `ppo/training.py` extracting `train_ppo` and helpers (`_load_config`, `_load_prompts`, `_extract_segment_obs`, `render_prompt`) from `rl/ppo_trainer.py` with changes:
  - Remove duplicate `token_to_vec` and `_compute_entropy`
  - Add `from features.vectorizer import token_to_vec, compute_entropy`
  - Rename all `ep_labels` → `ep_correct`
  - Add first-segment comment (Bug 5)
  - Update all config key paths to `data.*`, `mil.*`, `ppo.*`
- [x] 3.3 Move `eval/online_evaluate.py` → `ppo/eval.py` with changes:
  - Remove duplicate `token_to_obs`, `mean_pool_obs`, `compute_entropy_from_logprobs`
  - Add `from features.vectorizer import token_to_obs, mean_pool_obs, compute_entropy`
  - Update imports: `from ppo.model import PolicyValueNet`
  - Fix config key for PolicyValueNet hidden_dim: `cfg["ppo"]["model"]["hidden_dim"]` (Bug 3 fix)
  - Replace `from eval.metrics import safe_div` with `from utils.math import safe_div`
  - Replace `from eval.dataset_eval import load_temperature_labels` with `from features.dataset_eval import load_temperature_labels`

## 4. Move remaining files and delete legacy

- [x] 4.1 Move `eval/dataset_eval.py` → `features/dataset_eval.py`, replace `from eval.metrics import safe_div` with `from utils.math import safe_div`
- [x] 4.2 Update `features/build_dataset.py`: config reads `cfg["model"]["instance_dim"]` → `cfg["data"]["instance_dim"]`, `cfg["model"]["temp_bins"]` → `cfg["data"]["temp_bins"]`
- [x] 4.3 Delete `eval/` directory (all content moved: `evaluate.py` deleted, `dataset_eval.py` → features/, `mil_eval.py` → mil/, `online_evaluate.py` → ppo/, `metrics.py` → mil/eval.py + utils/math.py)
- [x] 4.4 Delete `models/` directory (content moved to `mil/model.py`)
- [x] 4.5 Delete `training/` directory (content moved to `mil/training.py`)
- [x] 4.6 Delete `rl/` directory (content moved to `ppo/model.py` + `ppo/training.py`)
- [x] 4.7 Delete `utils/distributed.py` (DDP removed — no remaining consumers)

## 5. Rewrite all config files

- [x] 5.1 Rewrite `configs/base.yaml`: add `data: {instance_dim, temp_bins}`, add `mil: {model, training}`, add `ppo: {model, training}`, remove top-level `model` and `training` sections, remove all `use_ddp` keys
- [x] 5.2 Rewrite `configs/arch_mlp_only.yaml`: same structure, preserve `mil.model.{use_position: false, use_gru: false}` and separate paths
- [x] 5.3 Rewrite `configs/baseline_fixed_window.yaml`: same structure, preserve `data.{segment_mode: fixed_window, segment_size: 32}` and separate paths
- [x] 5.4 Rewrite `configs/pool_concat.yaml`: same structure, preserve `data.{instance_dim: 2048, segment_pooling: concat}` and separate paths
- [x] 5.5 Rewrite `configs/temp_heads.yaml`: same structure, preserve `mil.training.alpha_temp: 0.1` and separate paths
- [x] 5.6 Verify all 5 configs are valid YAML

## 6. Reorganize tests

- [x] 6.1 Create `tests/test_segmenter.py` with segment_pooling and step_segment tests (from `test_core_functions.py`); imports from `features.segmenter`
- [x] 6.2 Create `tests/test_vectorizer.py` with token_to_vec, token_to_obs, mean_pool_obs, compute_entropy tests (from `test_core_functions.py`); imports from `features.vectorizer`
- [x] 6.3 Create `tests/test_jsonl_utils.py` with sample_prefix and row_group_key tests (from `test_core_functions.py`)
- [x] 6.4 Create `tests/test_mil_training.py` with collate_rows, smoothness_loss, top-k MIL instance loss tests (from `test_core_functions.py`); imports from `mil.training`, `mil.model`
- [x] 6.5 Delete `tests/test_core_functions.py` (content migrated to 6.1–6.4)
- [x] 6.6 Update `tests/test_mil_model.py` (was `test_mil_forward.py`): imports from `mil.model`
- [x] 6.7 Update `tests/test_ppo_model.py` (was `test_ppo_shapes.py`): imports from `ppo.model`
- [x] 6.8 Update `tests/test_metrics.py`: imports from `mil.eval` (metric functions) and `utils.math` (safe_div)

## 7. Update shell scripts

- [x] 7.1 Update `scripts/run_pipeline.sh`:
  - Remove `GPU_COUNT`, `USE_DDP`, `TORCHRUN_MASTER_PORT`, `count_gpus()` function
  - Remove `launch_train_mil()` function
  - Remove torchrun preflight check
  - Change MIL invocation to `python -m mil.training --config "$CONFIG" ...`
  - Change PPO invocation to `python -m ppo.training --config "$CONFIG" ...`
  - Change online eval invocation to `python -m ppo.eval --config "$CONFIG" ...`
  - Change offline eval invocation to `python -m mil.eval` (or remove entirely if each stage evaluates itself)
- [x] 7.2 Verify `scripts/stage2_mil.sh`, `scripts/stage3_ppo.sh`, `scripts/stage4_eval.sh` still work (thin wrappers around run_pipeline.sh)

## 8. Verification

- [x] 8.1 Run full test suite: `python -m pytest tests/ -v` — all 80+ tests pass
- [x] 8.2 Verify `python -m mil.training --config configs/base.yaml` starts without import errors
- [x] 8.3 Verify each config parses: `python -c "import yaml; yaml.safe_load(open('configs/<name>.yaml'))"` for all 5
- [x] 8.4 Grep for old import paths and dead references:
  - `grep -r "from models\." *`
  - `grep -r "from training\." *`
  - `grep -r "from rl\." *`
  - `grep -r "from eval.mil_eval\|from eval.online_evaluate\|from eval.metrics\|from eval.dataset_eval\|from eval.evaluate" *`
  - `grep -r "use_ddp" *` (should return nothing)
  - `grep -r "ep_labels" *` (should return nothing)
  - `grep -r "from utils.distributed" *` (should return nothing)
