# tf-mil — Temperature Framework with Multiple Instance Learning

Dynamic temperature selection for LLM math reasoning.  MIL learns to localise errors in reasoning chains; PPO learns to pick the right temperature per step to maximise majority-vote correctness.

Detailed methodology: **[PIPELINE.md](PIPELINE.md)**.

## Directory structure

```
tf-mil/
├── configs/
│   ├── dataset/                # Dataset generation configs (paths, inference, split)
│   └── training/               # MIL + PPO training configs (data, mil, ppo, inference)
├── data/                       # Raw prompt JSONL
├── datasets/                   # Processed datasets
├── checkpoints/                # Model weights
├── features/                   # Segmentation + feature construction
│   ├── schema.py               # Segment dataclass
│   ├── segmenter.py            # Segmentation strategies, segment_pooling, build_segment_obs_from_lp
│   ├── vectorizer.py           # token_to_vec, token_to_obs, mean_pool_obs, compute_entropy
│   └── dataset_eval.py         # per-temperature accuracy, majority-voting analysis
├── inference/
│   └── vllm_runner.py          # VLLMFeatureExporter: generation + online feature extraction
├── mil/                        # Stage 2 — MIL error localization
│   ├── model.py                # MILModel + auxiliary temperature heads
│   ├── utils.py                # BagDataset, TokenBatchSampler, make_collate_fn
│   ├── training.py             # train() + training loop
│   └── eval.py                 # evaluate_mil() + all MIL metric functions
├── ppo/                        # Stage 3 — online PPO training
│   ├── model.py                # PolicyValueNet, GAE, MIL warm-start
│   ├── training.py             # train_ppo() + online feature extraction
│   └── eval.py                 # OnlineTemperatureEvaluator
├── utils/                      # Shared infrastructure
│   ├── math.py                 # safe_div
│   ├── answer_verifier.py      # math-verify wrapper
│   ├── jsonl.py                # JSONL helpers (load/write/split/sample)
│   └── exp_logger.py           # file + stream logging
├── scripts/
│   ├── run_pipeline.sh         # Orchestrator (STAGES env var)
│   └── build_dataset.py        # Stage 1 entry: vLLM generation + majority voting
├── tests/                      # CPU-only tests
└── logs/                       # run logs
```

## Quick start

```bash
# 1. Generate dataset (raw vLLM, no speculative decode)
CUDA_VISIBLE_DEVICES=0,1 python scripts/build_dataset.py --config configs/dataset/full.yaml

# 2. Training + evaluation pipeline
GPU_DEVICES=0,1 bash scripts/run_pipeline.sh

# Run a specific ablation config
CONFIG=configs/training/arch_mlp_only.yaml GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

## Config variants

| Config | What it tests |
|---|---|
| `base.yaml` | fixed_window 512, mean pooling, GRU + pos encoding, pure MIL (k=1) |
| `step_control.yaml` | step segmentation (double-newline boundaries) vs fixed_window |
| `pool_concat.yaml` | concatenation pooling (2048-dim, no information loss) |
| `arch_mlp_only.yaml` | pure MLP — no position encoding, no GRU |
| `temp_heads.yaml` | temperature classification heads enabled (alpha = 0.1) |
| `instance_soft_pseudo_label.yaml` | soft pseudo-label instance loss (SeLa-MIL) |
| `instance_contrastive.yaml` | contrastive instance loss (NDI-MIL) |
| `ppo_control.yaml` | PPO terminal reward only (shaping_coef=0) |
| `hidden_states.yaml` | Qwen3-8B hidden states as features (4096-dim) |

## Running individual stages

```bash
# Stage 1 — build dataset
CUDA_VISIBLE_DEVICES=0,1 python scripts/build_dataset.py --config configs/dataset/full.yaml

# Stage 2 — train MIL
CUDA_VISIBLE_DEVICES=0 python -m mil.training --config configs/training/base.yaml

# Stage 2 — evaluate MIL
python -m mil.eval --config configs/training/base.yaml

# Stage 3 — online PPO
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.training --config configs/training/base.yaml

# Stage 3 — online evaluation (PPO vs best-fixed vs random)
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.eval --config configs/training/base.yaml
```

## Pipeline stages

```
build → mil → eval → ppo → eval_ol
          ↑
    MIL assessment before PPO
```

- **Stage 1** — Raw vLLM generates all responses at 15 temperatures (APC shares prompt KV-cache). Majority-vote label. Labels: 0 = correct (negative bag), 1 = error (positive bag).
- **Stage 2** — MIL learns to localise errors via always-on online feature extraction. `inst_head` output becomes PPO shaping reward; encoder weights warm-start PPO backbone.
- **Stage 3** — Policy controls vLLM generation temperature segment-by-segment. Majority-vote terminal reward + MIL shaping reward. PPO clip update.

## Key design decisions

1. **Flipped labels** — 0 = correct (negative bag), 1 = error (positive bag). Attention naturally focuses on error-like segments.
2. **Online feature extraction** — Logprobs and hidden states extracted during training via `extract_from_ids`, never stored in JSONL.
3. **Always-on speculative decode** — `_lazy_init` always configures extract_hidden_states for vLLM.
4. **Online PPO** — Offline data cannot learn causal action→reward; the policy must truly control generation.
5. **Majority voting end-to-end** — Stage 1 labels, PPO terminal reward, and online evaluation all use the same criterion.

## Tests

```bash
python -m pytest tests/ -v
```

CPU-only, 133 tests.
