## Why

MIL and PPO training currently produce text-only logs, making it difficult to monitor convergence, detect mode collapse, or diagnose hyperparameter issues without manually reading numbers. Adding structured metrics export and matplotlib-based plots gives a visual dashboard for every training run.

## What Changes

- MIL training loop records per-epoch metrics (`train_acc`, `grad_norm`, `attn_entropy`, `per_class_val_acc`) to a `{run_name}_{stage}_metrics.jsonl`
- PPO training loop records per-iteration temperature distribution, segment statistics, advantage stats, and clip fraction to the same JSONL format
- New `scripts/plot_training.py` reads one JSONL file and produces a single multi-subfigure PNG (`{run_name}_{stage}_training.png`)
- **No breaking changes** to existing training APIs or config schemas

## Capabilities

### New Capabilities
- `training-metrics-export`: structured per-epoch/iteration metrics written to JSONL during MIL and PPO training
- `training-plotting`: standalone script that reads a metrics JSONL and renders a multi-subfigure PNG with matplotlib

### Modified Capabilities
<!-- None -->

## Impact

- Affected files: `mil/training.py`, `ppo/training.py` (metrics recording), new `scripts/plot_training.py`
- Dependencies: `matplotlib` (new), `json` (stdlib), no vLLM or GPU needed for plotting
