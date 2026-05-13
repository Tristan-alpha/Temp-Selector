# tf-mil — Temperature Framework with Multiple Instance Learning

Dynamic temperature selection for LLM math reasoning.  MIL learns to localise errors in reasoning chains; PPO learns to pick the right temperature per step to maximise majority-vote correctness.

Detailed methodology: **[PIPELINE.md](PIPELINE.md)**.

## Directory structure

```
tf-mil/
├── configs/                   # YAML configs (base + 4 ablation variants)
├── data/                      # Raw prompt JSONL
├── datasets/                  # Processed datasets + cache
├── checkpoints/               # Model weights
├── features/                  # Stage 1 — dataset construction
│   ├── build_dataset.py       #   entry point
│   ├── schema.py              #   BagSample, TokenFeature, Segment
│   ├── segmenter.py           #   segmentation strategies + segment_pooling
│   ├── vectorizer.py          #   token_to_vec, token_to_obs, mean_pool_obs, compute_entropy
│   └── dataset_eval.py        #   per-temperature accuracy, majority-voting analysis
├── inference/                 # LLM backends
│   ├── vllm_runner.py         #   local GPU (APC multi-temp batch)
│   └── api_runner.py          #   Bailian DashScope API
├── mil/                       # Stage 2 — MIL error localization
│   ├── model.py               #   MILModel + auxiliary temperature heads
│   ├── training.py            #   BagDataset, top-k MIL loss, train_mil()
│   └── eval.py                #   evaluate_mil() + all MIL metric functions
├── ppo/                       # Stage 3 — online PPO training
│   ├── model.py               #   PolicyValueNet, GAE, MIL warm-start
│   ├── training.py            #   train_ppo() + online feature extraction
│   └── eval.py                #   OnlineTemperatureEvaluator
├── utils/                     # shared infrastructure
│   ├── math.py                #   safe_div
│   ├── answer_verifier.py     #   math-verify wrapper
│   ├── jsonl.py                #   shared JSONL helpers (load/write/split/sample)
│   └── exp_logger.py          #   file + stream logging
├── scripts/
│   ├── run_pipeline.sh        #   orchestrator (STAGES env var)
│   ├── stage{1..4}_*.sh       #   per-stage convenience scripts
│   ├── split_jsonl.py         #   train/val/test split (group-aware)
│   └── subsample_jsonl.py     #   dataset subsampling
├── tests/                     # 8 files, all CPU-only (~80 tests)
└── logs/                      # run logs
```

## Quick start

```bash
# 1. One-time data preparation (only needed once — all configs share the same dataset)
CUDA_VISIBLE_DEVICES=0,1 python scripts/build_dataset.py --config configs/dataset.yaml
python scripts/split_jsonl.py --config configs/dataset.yaml
python -m features.dataset_eval --config configs/dataset.yaml

# 2. Training + evaluation pipeline (loop this for each ablation)
GPU_DEVICES=0,1 bash scripts/run_pipeline.sh

# Run a specific ablation config
CONFIG=configs/arch_mlp_only.yaml GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

## Config variants

| Config | What it tests |
|---|---|
| `base.yaml` | step segmentation, mean pooling, GRU + pos encoding, pure MIL (k=1) |
| `baseline_fixed_window.yaml` | fixed-window 32-token (classic baseline) |
| `pool_concat.yaml` | concatenation pooling (2048-dim, no information loss) |
| `arch_mlp_only.yaml` | pure MLP — no position encoding, no GRU |
| `temp_heads.yaml` | temperature classification heads enabled (alpha = 0.1) |
| `instance_soft_pseudo_label.yaml` | soft pseudo-label instance loss (SeLa-MIL) |
| `instance_contrastive.yaml` | contrastive instance loss (NDI-MIL) |
| `ppo_control.yaml` | PPO terminal reward only (shaping_coef=0) |
| `hidden_states.yaml` | Qwen3-8B hidden states as features (4096-dim) |

## Running individual stages

```bash
# Data prep (one-time, uses configs/dataset.yaml)
CUDA_VISIBLE_DEVICES=0,1 python scripts/build_dataset.py --config configs/dataset.yaml
python scripts/split_jsonl.py --config configs/dataset.yaml
python -m features.dataset_eval --config configs/dataset.yaml

# Stage 2 — train MIL
CUDA_VISIBLE_DEVICES=0 python -m mil.training --config configs/base.yaml

# Stage 2 — evaluate MIL
python -m mil.eval --config configs/base.yaml

# Stage 3 — online PPO
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.training --config configs/base.yaml

# Stage 3 — online evaluation (PPO vs best-fixed vs random, majority vote)
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.eval --config configs/base.yaml
```

## Pipeline stages

```
build → split → mil → eval → ppo → eval_ol
                  ↑
            MIL assessment before PPO
```

- **Stage 1** — vLLM generates all responses at 15 temperatures in one batch (APC shares prompt KV-cache).  `num_votes` completions per prompt x temp, majority-vote label.  Labels are flipped: 0 = correct, 1 = error.
- **Stage 2** — MIL learns to localise errors in reasoning chains.  Top-k instance loss: only the most suspicious segments in wrong answers are pushed toward "error".  `inst_head` output becomes PPO shaping reward; encoder weights warm-start PPO backbone.
- **Stage 3** — Policy truly controls vLLM generation temperature segment-by-segment.  Majority-vote terminal reward + MIL shaping reward.  Multi-epoch mini-batch PPO with overfitting diagnostic.
- **Stage 4** — Offline metrics (bag accuracy, AUC, ECE, attention entropy, instance separation).  Online comparison (PPO vs best-fixed vs random) under majority voting.

## Key design decisions

1. **Flipped labels** — 0 = correct (negative bag), 1 = error (positive bag).  Attention naturally focuses on error-like segments.
2. **Top-k instance loss** — matches MIL assumption: at least one segment is wrong in an incorrect chain.
3. **Position encoding + BiGRU** — captures where errors occur and how they propagate across reasoning steps.
4. **Online PPO** — offline data cannot learn causal action to reward; the policy must truly control generation.
5. **Majority voting end-to-end** — Stage 1 labels, PPO terminal reward, and online evaluation accuracy all use the same criterion.

## Tests

```bash
python -m pytest tests/ -v
```

Eight test files (~80 tests, all CPU-only): MIL model, MIL training, PPO model, segmenter, vectorizer, JSONL utils, metrics, and schema.
