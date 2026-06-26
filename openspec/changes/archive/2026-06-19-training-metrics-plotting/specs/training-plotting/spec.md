## ADDED Requirements

### Requirement: Plotting script renders a multi-subfigure PNG

The script `scripts/plot_training.py` SHALL read a single metrics JSONL file and
output a single PNG file at the same location with the same base name and
`_training.png` suffix.

```
python scripts/plot_training.py --metrics logs/exp_mil_metrics.jsonl
  → logs/exp_mil_training.png
```

The script SHALL auto-detect the training stage (MIL or PPO) from the first row:
- Row has `"epoch"` → MIL layout (3×2 grid, 5 plots)
- Row has `"iter"` → PPO layout (4×3 grid, 11 plots)

#### Scenario: MIL JSONL produces MIL layout

- **WHEN** `scripts/plot_training.py --metrics logs/exp_mil_metrics.jsonl` is invoked
- **AND** the first row contains `"epoch": 1`
- **THEN** a 3×2 subfigure PNG is written to `logs/exp_mil_training.png`
- **AND** the subfigure grid is 3 rows × 2 columns

#### Scenario: PPO JSONL produces PPO layout

- **WHEN** `scripts/plot_training.py --metrics logs/exp_ppo_metrics.jsonl` is invoked
- **AND** the first row contains `"iter": 1`
- **THEN** a 4×3 subfigure PNG is written to `logs/exp_ppo_training.png`
- **AND** the subfigure grid is 4 rows × 3 columns

### Requirement: MIL subfigure layout

The MIL PNG SHALL contain the following subfigures in a 3×2 grid:

| Position | Content |
|----------|---------|
| (0, 0) | `loss.png-equivalent`: single Y-axis line plot of BCE loss by epoch |
| (0, 1) | `accuracy.png-equivalent`: two-line plot of `train_acc` and `val_acc` (0–1 scale) |
| (1, 0) | `grad_norm.png-equivalent`: line plot of `grad_norm` by epoch |
| (1, 1) | `attention_entropy.png-equivalent`: line plot of `attn_entropy` by epoch |
| (2, 0) | `per_class_accuracy.png-equivalent`: two-line plot of `val_acc_pos` and `val_acc_neg` (0–1 scale) |
| (2, 1) | empty (no subfigure) |

#### Scenario: MIL PNG has correct titles and labels

- **WHEN** the MIL PNG is rendered
- **THEN** each subfigure has a descriptive title
- **AND** X-axis is labeled "Epoch"
- **AND** multi-line subfigures include a legend

### Requirement: PPO subfigure layout

The PPO PNG SHALL contain the following subfigures in a 4×3 grid:

| Position | Content |
|----------|---------|
| (0, 0) | `accuracy.png-equivalent`: `train_acc` and `val_acc` lines |
| (0, 1) | `policy_loss.png-equivalent`: `policy_loss` line |
| (0, 2) | `value_loss.png-equivalent`: `value_loss` line |
| (1, 0) | `entropy.png-equivalent`: `entropy` line |
| (1, 1) | `reward_pos_ratio.png-equivalent`: `reward_pos_ratio` line (0–1 scale) |
| (1, 2) | `temperature_distribution.png-equivalent`: heatmap or stacked bar of `temp_dist` |
| (2, 0) | `temperature_stats.png-equivalent`: `temp_mean` and `temp_std` lines |
| (2, 1) | `segment_length.png-equivalent`: `segments_mean`, `segments_min`, `segments_max` lines |
| (2, 2) | `advantage.png-equivalent`: `advantage_mean` and `advantage_std` lines |
| (3, 0) | `clip_fraction.png-equivalent`: `clip_fraction` line |
| (3, 1) | `total_steps.png-equivalent`: `total_steps` line |
| (3, 2) | empty (no subfigure) |

#### Scenario: PPO PNG includes temperature heatmap

- **WHEN** the PPO PNG is rendered
- **AND** `temp_dist` contains counts for 15 temperature bins across all iterations
- **THEN** the `temperature_distribution` subfigure renders as a heatmap or stacked bar chart with temperature values on one axis and iteration on the other

#### Scenario: PPO PNG has correct layout

- **WHEN** the PPO PNG is rendered
- **THEN** each subfigure has a descriptive title
- **AND** X-axis is labeled "Iteration"
- **AND** multi-line subfigures include a legend

### Requirement: Missing or partial data is handled gracefully

If a metric key is absent from all rows in the JSONL, the corresponding subfigure
SHALL display "No data" text instead of crashing.

#### Scenario: Missing key does not crash plotting

- **WHEN** a MIL JSONL is missing the `grad_norm` key in all rows
- **THEN** the `grad_norm` subfigure displays "No data" text
- **AND** all other subfigures render normally

### Requirement: Plotting has no GPU dependency

`scripts/plot_training.py` SHALL NOT import torch, vllm, or any other
GPU-accelerated library. It SHALL run on a CPU-only machine with only
matplotlib and stdlib as dependencies.

#### Scenario: Plotting script runs without GPU

- **WHEN** invoked on a machine with no CUDA devices and matplotlib>=3.5 installed
- **THEN** the script renders a PNG successfully
