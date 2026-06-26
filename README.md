# prefix-value-pipeline branch

This branch extends the main `tf-mil` pipeline with a Prefix Value Model (PVM)
for causal, prefix-level temperature control. It is meant to document the delta
from the main branch, not to repeat the baseline project overview.

For the original project structure and baseline dynamic-temperature workflow,
see the main branch documentation and `PIPELINE.md`.

## What changed from `main`

The main branch trains MIL to localize error-prone reasoning segments, then uses
PPO to choose generation temperatures online. This branch keeps that baseline
available, but adds a prefix-value path:

1. Build prefix examples from partial reasoning chains.
2. Label each prefix with continuation outcomes across candidate temperatures.
3. Train a causal Prefix Value Model over segment-level hidden states.
4. Use the PVM state and calibrated value differences to guide PPO temperature
   choices during online rollout.

The key difference is the training target. The main branch learns from completed
responses and segment-level error localization. This branch additionally learns
the expected downstream value of a partial prefix, so temperature selection can
condition on the current reasoning state before the answer is finished.

## Prefix Value Model

The PVM is implemented as a mask-aware causal recurrent model. It consumes
prefix segment features and predicts whether continuing from the current prefix
is likely to lead to a correct majority-vote answer.

Important entry points:

| Area | Files |
|---|---|
| Prefix dataset construction | `mil/prefix_data.py` |
| PVM architecture | `mil/prefix_value.py` |
| PVM training and calibration | `mil/value_training.py` |
| PVM evaluation | `mil/value_eval.py` |
| Prefix PPO training | `ppo/prefix_training.py` |
| Prefix online rollout | `ppo/prefix_rollout.py` |
| Prefix PPO evaluation | `ppo/prefix_eval.py` |

## Experiment configs

The branch includes configs for both the full prefix-value path and smaller
single-seed evidence runs.

| Config | Purpose |
|---|---|
| `configs/training/full_prefix_value_500.yaml` | Full 500-problem PVM + PPO experiment |
| `configs/training/full_prefix_value_500_ppo_smoke_mb128.yaml` | PPO smoke variant for the full config |
| `configs/training/min_pvm_ppo_500_seed42.yaml` | Minimal single-seed PVM + PPO evidence path |
| `configs/training/min_pvm_q_500_seed42.yaml` | Prefix-Q selector extension |

Local timestamped configs may exist for follow-up runs. They are experiment
snapshots, not the canonical branch interface.

## Typical commands

Train the Prefix Value Model:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mil.value_training \
  --config configs/training/full_prefix_value_500.yaml
```

Run prefix PPO:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.prefix_training \
  --config configs/training/full_prefix_value_500.yaml
```

Evaluate prefix PPO online:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.prefix_eval \
  --config configs/training/full_prefix_value_500.yaml
```

Run the legacy-vs-prefix comparison helper when the required artifacts are
already prepared:

```bash
GPU_DEVICES=0,1 bash scripts/run_legacy_full_comparison.sh
```

## How to compare with `main`

Use the main branch as the baseline for the MIL warm-start and online PPO
workflow. Use this branch when evaluating whether prefix-derived values improve
temperature decisions.

For fair comparisons:

- Keep the same prompt split and vote count.
- Compare against fixed-temperature and legacy PPO baselines under the same
  code revision when possible.
- Report prompt-level majority accuracy separately from per-vote individual
  accuracy.
- Treat single-seed runs as implementation evidence; use the 42/43/44 seed
  flow before making stability claims.
