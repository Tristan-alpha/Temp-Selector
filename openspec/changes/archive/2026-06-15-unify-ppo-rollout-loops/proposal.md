## Why

`ppo/training.py` 中的训练回滚和验证回滚是同一核心逻辑的两个副本，但验证副本在创建时即有三个 bug（始终用 T=0.7、忽略 feature_mode、不构建 segment observation），且后续对训练路径的修复从未同步到验证路径。统一两者可以消除整类"训练修复了但验证没修"的 bug，并让验证回滚产生有意义的结果。

## What Changes

- 提取 `_decide_temperature(obs, policy, temp_bins, device, deterministic)` — 温度决策逻辑（训练用 sample，验证用 argmax）
- 提取 `_process_generated_features(feats, segment_size, instance_dim, ...)` — 生成后处理逻辑（构建 segment_obs、检测终止）
- 训练回滚和验证回滚均调用这两个共享函数
- 修复验证回滚的三个 bug：T=0.7 死循环、`return_logprobs=False`/`return_hidden=False` 硬编码、缺少 `build_segment_obs_from_lp`
- 验证回滚的 policy 决策改为 argmax（确定性），训练回滚保持 sample（随机探索）

## Capabilities

### New Capabilities

- `unified-rollout-loop`: PPO 训练中的共享回滚循环骨架，同时服务于训练和验证。`_decide_temperature` 和 `_process_generated_features` 两个函数封装所有差异点，确保训练和验证的生成→特征提取→状态更新路径始终一致。

### Modified Capabilities

- `ppo-online-generation`: 强化 PPO validation 的要求——验证回滚 SHALL 使用与训练回滚相同的 `generate_with_features` 参数（`return_logprobs=True, return_hidden=hs_needed`），SHALL 调用 `build_segment_obs_from_lp` 构建下一轮的 segment observation，SHALL 通过 policy 的 argmax 选择温度（而非固定 T=0.7）。

## Impact

- 影响文件：`ppo/training.py`（仅此一个文件）
- 不涉及 API 变更、配置变更或外部依赖
- 验证回滚的行为变更：`val_acc` 将从"固定 T=0.7 baseline"变为"policy 的确定性推理结果"
- 训练回滚的行为应保持不变（仅重构，无语义变更）
