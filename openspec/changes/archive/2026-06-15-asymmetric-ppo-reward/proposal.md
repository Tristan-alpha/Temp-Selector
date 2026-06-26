## Why

当前 PPO reward 对正确和错误链使用同一套 MIL attention 权重分布。MIL 学的是"哪个 segment 最像错"，它的 attention 在错误链上有语义（定位可疑 segment），但在正确链上没有对应语义——正确链的 attention 只是"哪个 segment 最能证明这链没问题"。把正负奖励对称地按 attention 分配，隐含了一个 MIL 不保证的前提。

正确链上均匀分布正奖励更自然：每个 segment 都对正确结果有平等贡献。

## What Changes

- PPO reward 改为不对称：**错误链**按 MIL attention 权重分布负奖励，**正确链**均匀分布正奖励。
- 不影响无 MIL 时的均匀回退（已存在）。

## Capabilities

### New Capabilities
- `asymmetric-ppo-reward`: PPO terminal reward 按链的正确性不对称分配：错误链用 attention 权重，正确链用均匀权重。

### Modified Capabilities
- `ppo-online-generation`: MODIFIED requirement — reward distribution 在正确/错误链之间不对称。

## Impact

- `ppo/training.py` — reward 构造循环（约 3 行变更）
- `specs/ppo-online-generation/spec.md` — 更新 reward 行为描述
