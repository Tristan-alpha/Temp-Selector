## Context

MIL and PPO training loops currently log scalar metrics via `logger.info()` but do not persist them in a structured format. To enable visual monitoring, we need to:

1. Record metrics per-epoch (MIL) or per-iteration (PPO) to a JSONL file
2. Provide a plotting script that reads this file and renders a multi-subfigure PNG

The project already uses JSONL for datasets (`utils/jsonl.py`). We follow the same pattern.

## Goals / Non-Goals

**Goals:**
- Structured JSONL output: one JSON object per row, appendable during training
- One PNG per stage per run, not separate files per metric
- Pure matplotlib (no extra Python dependencies beyond what's commonly available)
- Covers both MIL and PPO stages
- Plotting runs offline (after or during training), no GPU needed

**Non-Goals:**
- Real-time watch mode (can be added later, `--watch` flag out of scope)
- Multi-run comparison on the same chart
- seaborn integration
- Web dashboard, TensorBoard, or W&B export
- GPU metrics (vLLM throughput, memory, etc.)

## Decisions

### 1. JSONL naming: `{run_name}_{stage}_metrics.jsonl`

The `setup_experiment_logger()` already produces a `final_run_name`. We derive the JSONL path by suffixing the stage:

```
<log_dir>/<run_name>_mil_metrics.jsonl
<log_dir>/<run_name>_ppo_metrics.jsonl
```

**Alternatives considered:**
- Single `metrics.jsonl` with a `stage` field → harder to locate for a specific stage, and the two stages have different schemas.
- Embedding the path in config → unnecessary complexity; derived from `run_name` is sufficient.

### 2. PNG naming and layout: `{base}_training.png` with `plt.subplots`

The plotting script reads a single JSONL and outputs a single PNG with the same base name:

```
scripts/plot_training.py --metrics logs/exp_20260619_mil_metrics.jsonl
  → logs/exp_20260619_mil_training.png
```

Subfigure grids are fixed per stage:
- MIL: 3×2 (5 plots, 1 empty)
- PPO: 4×3 (11 plots, 1 empty)

Each subfigure is self-contained (title, axis labels, legend if applicable).

**Alternatives considered:**
- Separate PNGs per metric → too many files, harder to scroll through.
- `make_subplots` with `figsize` adjusted by `--rows`/`--cols` → unnecessary; the grid is fixed per stage and known at design time.

### 3. Metrics recording: inline helpers, not a callback system

Add small `_record_*()` helpers or explicit dict construction at the point of measurement. The training loop already has all the data — we just need to serialize it.

**Alternatives considered:**
- A `MetricsCollector` class → over-engineering for two training loops.
- A `@log_metric` decorator → opaque, harder to trace data flow.

### 4. Matplotlib only, no seaborn

The project has no plotting dependencies today. Adding `matplotlib` is acceptable (nearly universal in the ecosystem). Adding `seaborn` adds a transitive dependency with no critical benefit — our plots are simple line/bar/heatmap charts.

### 5. JSONL schema: flat dict per row, stage-specific keys

Each row is a self-contained metrics snapshot. MIL and PPO have different keys because they measure different things.

**MIL row:**
```json
{"epoch": 1, "loss": 0.452, "train_acc": 0.81, "val_acc": 0.72,
 "val_acc_pos": 0.65, "val_acc_neg": 0.88,
 "grad_norm": 1.23, "attn_entropy": 0.95}
```

**PPO row:**
```json
{"iter": 5, "total_loss": 0.82, "policy_loss": 0.51, "value_loss": 0.20,
 "entropy": 1.23, "reward_mean": -0.12, "reward_pos_ratio": 0.47,
 "train_acc": 0.55, "val_acc": 0.58,
 "temp_dist": {"0.1": 2, "0.3": 5, ..., "1.5": 1},
 "temp_mean": 0.72, "temp_std": 0.31,
 "segments_mean": 34.2, "segments_min": 12, "segments_max": 64,
 "advantage_mean": 0.01, "advantage_std": 0.85,
 "clip_fraction": 0.23, "total_steps": 480}
```

### 6. Plotting script auto-detects stage from JSONL keys

The script detects MIL vs PPO by checking for `"epoch"` or `"iter"` in the first row. No `--stage` flag needed.

## Risks / Trade-offs

- **[Risk]** JSONL file grows unbounded with long training runs (e.g., 200 MIL epochs × 5 fields = negligible; 200 PPO iters × 16 fields = ~10KB). Mitigation: Not a real problem — total file size is under 100KB even for the longest run.
- **[Risk]** Plotting while training writes to the same JSONL may produce torn rows. Mitigation: Training loop writes one complete `json.dumps()` + `\n` per write (atomic on POSIX for small writes). For safety, the plotting script ignores the last line if it fails to parse.
- **[Risk]** If `matplotlib` is not installed, the plotting script fails with a clear import error. Users who only train don't need matplotlib installed.
