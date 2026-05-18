# tf-mil — Temperature Framework with Multiple Instance Learning

Dynamic temperature selection for LLM math reasoning. MIL learns to localise errors in reasoning chains; PPO learns to pick the right temperature per step to maximise majority-vote correctness. Full pipeline docs: [PIPELINE.md](PIPELINE.md).

## Directory structure

| Path | Role |
|---|---|
| `features/schema.py` | `Segment` dataclass |
| `features/segmenter.py` | Segmentation strategies + `segment_pooling` + `build_segment_obs_from_lp` |
| `features/vectorizer.py` | `token_to_vec`, `token_to_obs`, `mean_pool_obs`, `compute_entropy` |
| `features/dataset_eval.py` | `evaluate_dataset()`, `load_temperature_labels()` |
| `inference/vllm_runner.py` | `VLLMFeatureExporter` — generation + `extract_from_ids` + `generate_with_features` |
| `mil/model.py` | `MILModel`, temp heads, `smoothness_loss` |
| `mil/utils.py` | `BagDataset`, `TokenBatchSampler`, `make_collate_fn` — shared MIL data utilities |
| `mil/training.py` | `train()` — Stage 2 training loop |
| `mil/eval.py` | `evaluate_mil()` — Stage 2 evaluation |
| `ppo/model.py` | `PolicyValueNet`, `compute_gae`, `sample_action`, warm-start |
| `ppo/training.py` | `train_ppo()` — Stage 3 training |
| `ppo/eval.py` | `OnlineTemperatureEvaluator` — Stage 3 online evaluation |
| `utils/math.py` | `safe_div` |
| `utils/answer_verifier.py` | Math-Verify wrapper (`verify_answer`, `extract_final_answer`, `self_consistency_correct`) |
| `utils/exp_logger.py` | File + stream logging |
| `utils/jsonl.py` | JSONL helpers (`load_jsonl`, `write_jsonl`, `split_by_group`) |
| `scripts/build_dataset.py` | Stage 1: raw vLLM gen, majority voting → JSONL with train/val/test splits |
| `scripts/run_pipeline.sh` | Pipeline orchestrator (`STAGES` env var) |
| `configs/dataset/` | Dataset generation configs (paths, inference, split) |
| `configs/training/` | MIL + PPO training configs (data, mil, ppo, inference) |

## Key data flow

```
Stage 1: JSONL prompts → raw vLLM multi-temp gen → majority voting → JSONL
         JSONL row = {sample_id, prompt, response, label, temperature,
                      token_ids, tokens, metadata}
              ↓
Stage 2: BagDataset loads JSONL as list[dict].  make_collate_fn always
         does online extraction via extract_from_ids → build_segment_obs_from_lp
         → segment_pooling → [K, 4098] instance matrix.
         MILModel(instances) → {bag_logit, inst_logit, attn_w, encoder_out}
         Loss = bag_bce + β×instance_bce + α×temp_ce + γ×smoothness
              ↓  warm-start backbone + inst_logit as shaping reward
Stage 3: PolicyValueNet(segment_obs) → temperature action
         vLLM generates next segment at chosen temp
         Terminal reward = majority-vote correctness (±1)
         Shaping reward = shaping_coef × (1 − sigmoid(inst_logit))
         GAE + PPO clip update
```

## Coding conventions

- `from __future__ import annotations` in every `.py` file
- Type annotations on all function signatures
- Config keys use new schema: `data.{instance_dim,temp_bins}`, `mil.{model,training}`, `ppo.{model,training}`
- Tests are CPU-only; run with `python -m pytest tests/ -v`
- No DDP — MIL model is ~500K params, single-GPU training sufficient

## Common pitfalls

- **Label semantics**: `label=0` means correct (negative bag), `label=1` means error (positive bag). This is flipped from standard MIL convention. The `ep_correct` variable in PPO training is the opposite (1=correct). Do NOT mix them.
- **segment_obs is None on first segment**: In PPO training/eval, the first segment has no prior token features to observe. A dummy action (temp=0.7, action=0) is recorded but safely skipped during batch construction (`range(1, n_steps)`).
- **Even vote ties**: `num_votes=8` in config. Majority threshold `(V+1)//2=4` means 4-4 ties are labeled "correct". Use odd vote counts to avoid this.
- **Config section scoping**: `mil.model.hidden_dim` and `ppo.model.hidden_dim` are separate keys. The evaluator reads `ppo.model.hidden_dim` — do not use `model.hidden_dim` (that section was deleted).
- **`inst_repr = out["encoder_out"]`**: In `mil/training.py`, the dynamic temp head MUST receive `encoder_out` (pos+GRU processed), not `mil.encoder(x)` (raw). The latter was Bug 1 — fixed but easily reintroduced.
- **MIL DataLoader `num_workers=0`**: Feature extraction happens in collate_fn via VLLMFeatureExporter (cannot be pickled). Do NOT set num_workers > 0 or DataLoader will hang/fail.
- **BagDataset is metadata-only**: Rows are raw dicts from JSONL, not pre-built tensors. Instance tensor construction happens in `make_collate_fn()`. The old `collate_rows` and `RowTensor` are deleted.

## Common tasks

**Run a full pipeline:**
```bash
GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

**Run specific stages:**
```bash
STAGES=build,split GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

**Run an ablation experiment:**
```bash
CONFIG=configs/training/arch_mlp_only.yaml GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

**Add a new ablation:** Create `configs/training/my_ablation.yaml` from `base.yaml`, change the parameter you want to test, then run with `CONFIG=configs/training/my_ablation.yaml`.

**Run only MIL training after rebuilding data:**
```bash
STAGES=mil GPU_DEVICES=0 bash scripts/run_pipeline.sh
```

**Run tests:**
```bash
python -m pytest tests/ -v
```
