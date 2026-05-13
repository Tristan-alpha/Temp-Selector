## Why

1. Current ablation configs use separate dataset paths (e.g., `datasets/all_spl.jsonl`) even though the Stage 1 product is identical. The root cause: `segment_mode` and `segment_size` determine segmentation, which happens in Stage 1. Moving segmentation to Stage 2 (`BagDataset`) makes ALL configs share a single Stage 1 output, eliminating redundant vLLM re-rollouts.
2. `feature_mode` naming is inconsistent: `combined` equals `logits_topk` (neither includes hidden states), and no explicit name exists for "basic" (logprob + entropy only).
3. No config uses hidden states as features.

## What Changes

- **Move segmentation to BagDataset**: `build_segments()` called in `BagDataset.__init__` instead of `build_dataset.py`. All configs now share `datasets/all.jsonl` (always step-segmented in Stage 1 for consistency with system prompt format, but BagDataset can override). This makes `baseline_fixed_window.yaml` and `pool_concat.yaml` also share base Stage 1 data.
- **All 8 configs use shared dataset paths**: `all_dataset`/`train_dataset`/`val_dataset`/`test_dataset` are identical across all configs. Only `mil_ckpt` and `ppo_ckpt` paths differ.
- **Rename feature_mode values**: empty/无 → `basic`; both `combined` and `logits_topk` → `topk_logits`; `hidden_states` unchanged. Remove `"combined"` branch from runner code.
- **Add `configs/hidden_states.yaml`**: `feature_mode: hidden_states`, `instance_dim: 4096` (Qwen3-8B hidden_size)
- **Add `configs/ppo_control.yaml`**: `shaping_coef: 0.0` for terminal-reward-only baseline
- **PPO training reads from `train_dataset`**: `ppo/training.py` extracts prompts from labeled train JSONL instead of `raw_input`, symmetric with the `ppo/eval.py` fix

## Capabilities

### New Capabilities

- `shared-datasets`: All configs share a single Stage 1 output; segmentation deferred to BagDataset
- `hidden-states-config`: New config using Qwen3-8B hidden states as MIL features
- `feature-mode-cleanup`: Renamed to `basic`/`topk_logits`/`hidden_states`; dead `combined` alias removed
- `ppo-train-prompts`: PPO training reads from `train_dataset` instead of `raw_input`

## Impact

- **Configs**: All 9 files updated (shared paths, renamed feature_mode, 2 new configs)
- **Code**: `BagDataset` gains `segment_mode`/`segment_size` params to call `build_segments`; `vllm_runner.py` + `api_runner.py` dispatch simplified
- **Data**: Stage 1 always uses step segmentation (for consistent multi-purpose reuse); BagDataset can override
- **Docs**: README, PIPELINE.md, CLAUDE.md directory structure
