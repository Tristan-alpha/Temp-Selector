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
| `mil/model.py` | `MILModel`, `InstanceEncoder`, `AttentionAggregator` |
| `mil/utils.py` | `BagDataset`, `make_collate_fn`, `SegmentCacheDataset`, `make_cached_collate_fn` — shared MIL data utilities |
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
         MILModel(instances) → {bag_logit, attn_w, bag_repr}
         Loss = bag_bce only (Ilse et al. 2018)
              ↓  warm-start backbone encoder weights
Stage 3: PolicyValueNet(segment_obs) → temperature action
         vLLM generates next segment at chosen temp
         Terminal reward = majority-vote correctness (±1)
         Reward = terminal_reward × (attn_w / attn_w.sum())  distributed to all steps
         GAE + PPO clip update
```

## Coding conventions

- `from __future__ import annotations` in every `.py` file
- Type annotations on all function signatures
- Config keys use new schema: `data.{instance_dim,temp_bins}`, `mil.{model,training}`, `ppo.{model,training}`
- Tests are CPU-only; run with `python -m pytest tests/ -v`
- No DDP — MIL model is ~500K params, single-GPU training sufficient

## Common pitfalls

- **Label fields**: `individual_label` (0=correct, 1=error — per-response correctness, used by MIL). `voting_label` (0=correct, 1=error — majority-vote result, used by PPO temperature bias). MIL code has inline comments (`# label=1: positive bag (contains errors)`) at every branch. `ep_correct` in PPO is 1=correct.
- **segment_obs is None on first segment**: In PPO training/eval, the first segment has no prior token features to observe. A dummy action (temp=0.7, action=0) is recorded but safely skipped during batch construction (`range(1, n_steps)`).
- **Self-consistency is mode-based, not majority-threshold**: `self_consistency_correct()` uses `Counter.most_common(1)` — the most frequent extracted answer is compared to gold. There is no `(V+1)//2` threshold. Frequency ties among distinct answers are broken by first-occurrence order (deterministic but arbitrary). Even vs odd `num_votes` makes no structural difference for mode-based voting.
- **Config section scoping**: `mil.model.hidden_dim` and `ppo.model.hidden_dim` are separate keys. The evaluator reads `ppo.model.hidden_dim` — do not use `model.hidden_dim` (that section was deleted).
- **`inst_repr = out["encoder_out"]`**: In `mil/training.py`, the dynamic temp head MUST receive `encoder_out` (pos+GRU processed), not `mil.encoder(x)` (raw). The latter was Bug 1 — fixed but easily reintroduced.
- **MIL DataLoader `num_workers=0`**: Feature extraction happens in collate_fn via VLLMFeatureExporter (cannot be pickled). Do NOT set num_workers > 0 or DataLoader will hang/fail.
- **BagDataset is metadata-only**: Rows are raw dicts from JSONL, not pre-built tensors. Instance tensor construction happens in `make_collate_fn()`. The old `collate_rows` and `RowTensor` are deleted.
- **Concat pooling + instance_dim**: When `segment_pooling: concat`, `instance_dim` is per-token feature dim (not per-instance). Per-instance dim = `instance_dim × segment_size`. The `top_k_logprobs: 4096` is for entropy accuracy only — top-k values are not intended as model features. The truncation from 4098→64 when `instance_dim: 64` is intentional.

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

**Plot training metrics:**
```bash
python scripts/plot_training.py --metrics logs/<run_name>_mil_metrics.jsonl
python scripts/plot_training.py --metrics logs/<run_name>_ppo_metrics.jsonl
```
Outputs a multi-subfigure PNG at `logs/<run_name>_{mil,ppo}_training.png`.
