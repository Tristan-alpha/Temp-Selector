# tf-mil — Temperature Framework with Multiple Instance Learning

Dynamic temperature selection for LLM math reasoning. MIL learns to localise errors in reasoning chains; PPO learns to pick the right temperature per step to maximise majority-vote correctness. Full pipeline docs: [PIPELINE.md](PIPELINE.md).

## Directory structure

| Path | Role |
|---|---|
| `features/schema.py` | `TokenFeature`, `Segment`, `BagSample` — core data structures |
| `features/segmenter.py` | Segmentation strategies (`fixed_window`, `step`, `punctuation`) + `segment_pooling` |
| `features/vectorizer.py` | `token_to_vec`, `token_to_obs`, `mean_pool_obs`, `compute_entropy` — feature construction |
| `features/dataset_eval.py` | `evaluate_dataset()`, `load_temperature_labels()` — Stage 1 analysis |
| `inference/sglang_runner.py` | `SGLangRunner` — **default** backend; single engine with `generate()` + `extract()` |
| `inference/vllm_runner.py` | `VLLMFeatureExporter` — legacy backend (`--backend vllm`); raises ValueError for hidden_states mode |
| `mil/model.py` | `MILModel`, temp heads, smoothness_loss — all MIL model definitions |
| `mil/training.py` | `BagDataset`, `collate_rows`, `train_mil()` — Stage 2 training |
| `mil/eval.py` | `evaluate_mil()` + all MIL metric functions — Stage 2 evaluation |
| `ppo/model.py` | `PolicyValueNet`, `compute_gae`, `sample_action`, warm-start — PPO model |
| `ppo/training.py` | `train_ppo()` + online feature extraction — Stage 3 training |
| `ppo/eval.py` | `OnlineTemperatureEvaluator` — Stage 3 online evaluation |
| `utils/math.py` | `safe_div` — shared one-liner |
| `utils/answer_verifier.py` | Math-Verify wrapper for answer correctness checking |
| `utils/exp_logger.py` | File + stream logging setup |
| `utils/dataset_io.py` | Hybrid JSONL + safetensors I/O — **deleted**; SGLangRunner.extract() replaces safetensors sidecar |
| `scripts/run_pipeline.sh` | Full pipeline orchestrator (`STAGES` env var controls which stages run) |
| `scripts/build_dataset.py` | Stage 1 entry: vLLM multi-temp gen, majority voting. For hidden_states/all mode: merged build+split writes train/val/test JSONL+safetensors directly |
| `scripts/split_jsonl.py` | Group-aware train/eval split; propagates safetensors sidecar |
| `utils/jsonl.py` | Shared JSONL helpers (`sample_prefix`, `row_group_key`, `load_jsonl`, `write_jsonl`, `split_by_group`) |
| `configs/` | 5 YAML configs: `base.yaml` + 4 ablation variants |

## Key data flow

```
Stage 1: JSONL prompts → multi-temp generation → BagSample (per prompt×temp×vote)
         BagSample = {token_features: [TokenFeature], segment_spans: [Segment], label: 0|1, ...}
         feature_mode=hidden_states/all → merged build+split → train/val/test JSONL (no sidecar)
              ↓
Stage 2: BagDataset loads JSONL; for hidden_states/all: SGLang engine batch prefill
         extracts per-token hidden states → segment_pooling → [K, 64] instance matrix
         MILModel(instances) → {bag_logit, inst_logit, attn_w, encoder_out}
         Loss = bag_bce + β×top_k_instance_bce + α×temp_ce + γ×smoothness
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
CONFIG=configs/arch_mlp_only.yaml GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

**Add a new ablation:** Create `configs/my_ablation.yaml` from `base.yaml`, change the parameter you want to test, then run with `CONFIG=configs/my_ablation.yaml`.

**Run only MIL training after rebuilding data:**
```bash
STAGES=mil GPU_DEVICES=0 bash scripts/run_pipeline.sh
```

**Run tests:**
```bash
python -m pytest tests/ -v
```
