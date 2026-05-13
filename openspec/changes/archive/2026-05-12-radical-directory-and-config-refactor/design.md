## Context

The `tf-mil` project implements a 4-stage pipeline. Currently the code is organized by function type (`models/`, `training/`, `rl/`, `eval/`) rather than by stage, causing:

- **Config ambiguity**: `model.hidden_dim` and `ppo.policy_hidden_dim` both exist (both 256), but `online_evaluate.py` reads `model.hidden_dim` to create a PPO PolicyValueNet — a bug waiting to surface
- **Scattered MIL code**: MIL model in `models/`, training in `training/`, evaluation in `eval/mil_eval.py` — 3 directories for one logical module
- **Duplicate code**: `token_to_vec()` identical in `training/train_mil.py` and `rl/ppo_trainer.py`; `compute_entropy` duplicated between `rl/ppo_trainer.py` and `eval/online_evaluate.py`; `compute_attention_metrics` defined in `eval/metrics.py` but `mil_eval.py` computes the same thing inline
- **Misleading naming**: `ep_labels` in PPO uses 1=correct but project convention is 0=correct
- **Dead config keys**: `ppo.use_ddp` in all 5 configs — PPO code never reads it; `training.use_ddp` enables DDP for a ~500K param model where overhead exceeds benefit
- **Misplaced files**: `features/answer_verifier.py` is not a feature extraction utility; `features/segmenter.py` is being used as a general-purpose utils dump
- **Dead directory**: after splitting `eval/metrics.py`, the `eval/` directory would be empty

## Goals / Non-Goals

**Goals:**
- MIL and PPO each live in a self-contained top-level directory (`mil/`, `ppo/`)
- `features/` has clean boundaries: `schema.py`, `segmenter.py` (segmentation only), `vectorizer.py` (token feature construction), `dataset_eval.py`, `build_dataset.py`
- `utils/` expanded with properly-scoped shared infrastructure
- `eval/` directory deleted — content distributed to its consumers
- Config sections map 1:1 to module boundaries; dead keys removed
- All DDP code removed (~150 lines across Python + shell)
- All code duplication eliminated
- All existing tests pass

**Non-Goals:**
- Changing model architecture, loss functions, or training algorithms
- Changing checkpoint format or serialization
- Adding new features beyond the structural cleanup

## Decisions

### D1: Final directory layout

```
features/                       # Stage 1: data pipeline
├── schema.py                   # TokenFeature, Segment, BagSample
├── segmenter.py                # Segmentation strategies + segment_pooling (mean, concat)
├── vectorizer.py               # token_to_vec, token_to_obs, mean_pool_obs, compute_entropy
├── dataset_eval.py             # evaluate_dataset(), load_temperature_labels()
│                               #   (moved from eval/dataset_eval.py)
└── build_dataset.py            # Stage 1 main entry point

mil/                            # Stage 2: MIL error localization
├── __init__.py
├── model.py                    # MILModel, InstanceEncoder, SinusoidalPositionalEncoding,
│                               #   AttentionAggregator, GlobalTempHead, DynamicTempHead,
│                               #   smoothness_loss (merged from models/temp_predictor.py)
├── training.py                 # BagDataset, RowTensor, collate_rows, train_mil()
└── eval.py                     # evaluate_mil() + all MIL metric functions
│                               #   (metrics merged from eval/metrics.py)

ppo/                            # Stage 3: Online PPO training
├── __init__.py
├── model.py                    # PolicyValueNet, compute_gae, sample_action,
│                               #   load_mil_encoder_for_warmstart
├── training.py                 # train_ppo() + online feature extraction helpers
└── eval.py                     # OnlineTemperatureEvaluator

utils/                          # Cross-stage shared infrastructure
├── __init__.py
├── distributed.py              # ★ DELETED (DDP unused)
├── exp_logger.py               # unchanged
├── answer_verifier.py          # moved from features/ (not a feature extraction concern)
└── math.py                     # safe_div (new — one-liner used by 3 modules)

inference/                      # LLM backends (unchanged)
scripts/                        # Shell orchestration (DDP logic removed)
```

**Key moves:**

| Old | New | Reason |
|---|---|---|
| `models/mil.py` | `mil/model.py` | MIL self-containment |
| `models/temp_predictor.py` | merged into `mil/model.py` | Only 30 lines of auxiliary heads |
| `training/train_mil.py` | `mil/training.py` | MIL self-containment |
| `eval/mil_eval.py` | `mil/eval.py` | Tightly coupled to MIL model |
| `rl/ppo_trainer.py` | `ppo/model.py` + `ppo/training.py` | Split: model vs training |
| `eval/online_evaluate.py` | `ppo/eval.py` | Tightly coupled to PPO model |
| `eval/dataset_eval.py` | `features/dataset_eval.py` | Stage 1 data analysis |
| `eval/evaluate.py` | **deleted** | Each stage has its own eval |
| `eval/metrics.py` | split into consumers | No shared eval directory needed |
| `features/answer_verifier.py` | `utils/answer_verifier.py` | Cross-stage infrastructure |
| `utils/distributed.py` | **deleted** | DDP removed entirely |
| — | `utils/math.py` | New: `safe_div` shared by 3 modules |
| — | `features/vectorizer.py` | New: extracted from duplicated code |

### D2: `eval/metrics.py` split

| Symbol | Destination | Reason |
|---|---|---|
| `safe_div` | `utils/math.py` | One-liner shared by `features/dataset_eval.py`, `mil/eval.py`, `ppo/eval.py` |
| `compute_bag_metrics` | `mil/eval.py` | Only used by MIL eval |
| `compute_calibration` | `mil/eval.py` | Only used by MIL eval |
| `compute_multiclass_metrics` | `mil/eval.py` | Only used by MIL eval |
| `compute_confusion_matrix` | `mil/eval.py` | Only used by `compute_multiclass_metrics` |
| `compute_auc` | `mil/eval.py` | Only used by `compute_bag_metrics` |
| `compute_attention_metrics` | `mil/eval.py` | Previously dead code — `mil/eval.py` will call it instead of computing inline |

After the split, `eval/` is empty and deleted. `tests/test_metrics.py` imports from `mil.eval` and `utils.math`.

### D3: Config structure

```yaml
data:             # shared
  instance_dim: 64
  temp_bins: [...]

mil:
  model:          # MIL architecture
    hidden_dim: 256
    aggregator: attention
    use_position: true
    use_gru: true
  training:       # MIL hyperparams (use_ddp REMOVED)
    batch_size: 32
    lr: 2.0e-4
    ...

ppo:
  model:          # PPO architecture
    hidden_dim: 256   # was policy_hidden_dim
  training:       # PPO hyperparams (use_ddp REMOVED)
    ...
```

Top-level `model` and `training` sections removed. `use_ddp` removed from both `mil.training` and `ppo`.

### D4: DDP removal

| Component | Action |
|---|---|
| `mil/training.py` | Remove `DistributedDataParallel`, `DistributedSampler`, `setup_distributed`, `barrier`, `cleanup_distributed`, `emit_startup_self_check`. Device selection: `torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")`. Plain `shuffle=True` in DataLoader. |
| `utils/distributed.py` | Delete entire file |
| `scripts/run_pipeline.sh` | Remove `GPU_COUNT`, `USE_DDP`, `TORCHRUN_MASTER_PORT`, `count_gpus()`, `launch_train_mil()` function. `torchrun` preflight check removed. MIL training becomes direct `python -m mil.training` |
| All 5 configs | Remove `use_ddp` from both `mil.training` and `ppo` sections |

### D5: Code duplication eliminated

| Duplicate | Found in | Unified to |
|---|---|---|
| `token_to_vec` | `training/train_mil.py`, `rl/ppo_trainer.py` | `features/vectorizer.py` |
| `_compute_entropy` / `compute_entropy_from_logprobs` | `rl/ppo_trainer.py`, `eval/online_evaluate.py` | `features/vectorizer.py` |
| `token_to_obs` | `eval/online_evaluate.py` | `features/vectorizer.py` |
| `mean_pool_obs` | `eval/online_evaluate.py` | `features/vectorizer.py` |
| `compute_attention_metrics` (inline in mil_eval.py) | `eval/mil_eval.py` + `eval/metrics.py` | `mil/eval.py` (function called, not duplicated) |
| `safe_div` | 3 importers from `eval.metrics` | `utils/math.py` (single source) |

### D6: Bug fixes included

| Bug | Action |
|---|---|
| Bug 1 (dynamic head train/eval mismatch) | Already fixed — preserved in `mil/training.py` |
| Bug 2 (`ep_labels` semantics) | Rename `ep_labels` → `ep_correct` in `ppo/training.py` |
| Bug 3 (config key for PPO hidden_dim) | `ppo/eval.py` reads `ppo.model.hidden_dim` instead of `model.hidden_dim` |
| Bug 4 (dead repeat_interleave truncation) | Already fixed — preserved in `mil/eval.py` |
| Bug 5 (first-segment dummy values) | Comment added in `ppo/training.py` |

### D7: Config file scope

All 5 configs receive identical structural transformation. Experimental diffs preserved:

| Config | Preserved diff vs base |
|---|---|
| `base.yaml` | Baseline |
| `arch_mlp_only.yaml` | `mil.model.{use_position: false, use_gru: false}` |
| `baseline_fixed_window.yaml` | `data.{segment_mode: fixed_window, segment_size: 32}` |
| `pool_concat.yaml` | `data.{instance_dim: 2048, segment_pooling: concat}` |
| `temp_heads.yaml` | `mil.training.alpha_temp: 0.1` |

### D8: Test file reorganization

| Old | New |
|---|---|
| `test_schema.py` | `test_schema.py` (unchanged) |
| `test_core_functions.py` | split into `test_segmenter.py`, `test_vectorizer.py`, `test_jsonl_utils.py`, `test_mil_training.py` |
| `test_mil_forward.py` | `test_mil_model.py` (expanded) |
| `test_ppo_shapes.py` | `test_ppo_model.py` (expanded) |
| `test_metrics.py` | `test_metrics.py` (imports updated to `mil.eval` and `utils.math`) |

## Risks / Trade-offs

- **[Import breakage]** Moving 15+ files → **Mitigation**: create new files → update imports → delete old → run tests
- **[Shell script breakage]** Module paths change → **Mitigation**: audit all `.sh` files
- **[Checkpoint compatibility]** `torch.save`/`torch.load` with `state_dict` is path-agnostic → **No risk**
- **[Large diff]** ~25 files touched → **Mitigation**: logical commits per step

## Migration Plan

1. Create `mil/`, `ppo/` with `__init__.py`; create `utils/math.py`
2. Create `features/vectorizer.py` with unified `token_to_vec`, `token_to_obs`, `mean_pool_obs`, `compute_entropy`
3. Move `eval/dataset_eval.py` → `features/dataset_eval.py`; move `features/answer_verifier.py` → `utils/answer_verifier.py`
4. Merge `models/mil.py` + `models/temp_predictor.py` → `mil/model.py`; merge `eval/metrics.py` content into `mil/eval.py` + `utils/math.py`
5. Move `training/train_mil.py` → `mil/training.py` (de-DDP, de-duplicate, update config keys and imports)
6. Move `eval/mil_eval.py` → `mil/eval.py` (update imports, use merged metric functions)
7. Extract `rl/ppo_trainer.py` → `ppo/model.py` + `ppo/training.py` (de-duplicate, rename `ep_labels`, fix Bug 5)
8. Move `eval/online_evaluate.py` → `ppo/eval.py` (de-duplicate, fix Bug 3 config key)
9. Rewrite all 5 config files
10. Delete legacy files: `eval/` (entire directory), `models/`, `training/`, `rl/`, `utils/distributed.py`
11. Reorganize tests
12. Update `scripts/run_pipeline.sh` (de-DDP + new module paths)
13. Run full test suite (target: all 80+ tests pass)
