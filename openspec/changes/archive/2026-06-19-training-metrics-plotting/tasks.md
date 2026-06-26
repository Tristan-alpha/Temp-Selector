## 1. MIL metrics recording

- [x] 1.1 Modify `mil/training.py` `train()`: add `train_acc`, `grad_norm`, `attn_entropy`, `val_acc_pos`, `val_acc_neg` measurement during training and validation
- [x] 1.2 Open `{log_dir}/{run_name}_mil_metrics.jsonl` at training start, append one JSON line per epoch
- [x] 1.3 Include `final_run_name` from `setup_experiment_logger` so JSONL filename matches the log filename convention

## 2. PPO metrics recording

- [x] 2.1 Modify `ppo/training.py` `train_ppo()`: compute `reward_pos_ratio`, `temp_dist` dict, `temp_mean`, `temp_std`, `segments_mean/min/max`, `advantage_mean`, `advantage_std`, `clip_fraction` per iteration
- [x] 2.2 Open `{log_dir}/{run_name}_ppo_metrics.jsonl` at training start, append one JSON line per iteration
- [x] 2.3 Flush after each write to protect against partial writes on crash

## 3. Plotting script

- [x] 3.1 Create `scripts/plot_training.py` — entry point: `--metrics <path>` arg, `--output <path>` optional (default: derive from metrics path)
- [x] 3.2 Auto-detect stage: `"epoch"` in first row → MIL layout; `"iter"` → PPO layout
- [x] 3.3 Implement MIL 3×2 grid: loss, accuracy (train+val), grad_norm, attn_entropy, per_class_acc (pos+neg), one empty
- [x] 3.4 Implement PPO 4×3 grid: accuracy, policy_loss, value_loss, entropy, reward_pos_ratio, temp_dist heatmap, temp_stats, segment_length, advantage, clip_fraction, total_steps, one empty
- [x] 3.5 Handle missing keys: render "No data" in the subfigure instead of crashing
- [x] 3.6 Handle truncated JSONL: skip last line if `json.loads` fails

## 4. Tests

- [x] 4.1 Add CPU-only tests in `tests/test_plotting.py` for: JSONL parsing, stage detection, missing-key handling, truncated-line handling
- [x] 4.2 Verify MIL metrics JSONL format with a 2-epoch synthetic run
- [x] 4.3 Verify PPO metrics JSONL format with a 2-iteration synthetic run

## 5. Documentation

- [x] 5.1 Update `CLAUDE.md` Common Tasks: add plotting invocation example
- [x] 5.2 Update `PIPELINE.md` if pipeline stages diagram or scripts list references any new file

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests pass (including new plotting tests)
- [x] 6.2 Run `python -m compileall -q scripts/plot_training.py mil/training.py ppo/training.py`
- [x] 6.3 Check CLAUDE.md and PIPELINE.md for stale references to metrics or logging conventions
