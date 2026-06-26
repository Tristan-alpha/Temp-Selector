## Context

当前 MIL 模型训练使用四个 loss 组件，其中 instance_bce 和 temp_ce 的监督信号来自模型自身输出（自循环），无法用真实标签评估。Ilse et al. (2018) 证明纯 bag_bce 训练的 attention 即可学到有意义的 instance importance。同时，PPO 中 `inst_logit` shaping reward 的可靠性从未被验证。本次重构将 MIL 精简为 bag_bce-only，用 attention weight 替代 inst_logit 在 PPO 中做 credit assignment。

## Goals / Non-Goals

**Goals:**
- MIL 模型只保留 `bag_head` + `AttentionAggregator`，训练只使用 `bag_bce`
- 删除 `inst_head`、`DynamicTempHead`、`GlobalTempHead`、`smoothness_loss` 及所有 instance loss 逻辑
- MIL 评估使用 bag AUC、calibration；删除 instance AUC 和 temperature accuracy
- PPO shaping reward 改为 `terminal_reward × attention_weight`，累积全 chain segment 后一次调用 MIL
- 对应 config key 清理

**Non-Goals:**
- 不修改 vLLM 生成管线
- 不修改 segmenter 或特征提取
- 不修改 PPO 模型结构（PolicyValueNet 不变）
- 不修改 warm-start 逻辑（encoder 仍然可以 warm-start）

## Decisions

### Decision 1: 删除 inst_head，attention 由 bag_bce 驱动

**选择**: 删除 `inst_head` 和所有 instance loss。MILModel.forward() 不输出 `inst_logit`。

**替代方案**: 保留 inst_head 但不参与训练——但这样 forward 里还有死代码，不如直接删掉。

**理由**: Ilse et al. 的原始 MIL 论文只用 bag_bce 训 attention。当前 instance_bce 的伪标签来自 `inst_logit` 自身的 top-k 排名，不是真实实例标签，相当于模型给自己出题。去掉它不会损失有效监督信号。

### Decision 2: 删除 temp_head 和 smoothness_loss

**选择**: 删除 `DynamicTempHead`、`GlobalTempHead` 及 temp_ce、smoothness_loss。

**理由**: 
- temp_head 需要 `encoder_out` 作为输入，但该特征已与 attention 耦合，其输出的 temp 预测缺乏独立的评估标准。
- smoothness_loss 旨在防止 attention 坍缩，但 Ilse et al. 论文中没有此正则化且 attention 表现良好。坍缩风险可以通过 attention 可视化监测。

### Decision 3: PPO 使用 terminal_reward × attention_weight

**选择**: 在 PPO batch 构建时，将每 chain 的所有 segment 拼成 full bag（`[K, obs_dim]`），一次调用 MIL 获取 `attn_w`。intermediate step 的 reward = `shaping_coef × terminal_reward × attn_w[step_idx]`。

**理由**: attention 的职责从"判断错误"变为"分配责任"。terminal_reward 是唯一可靠信号，attention 决定它在各 step 间如何分配。

## Risks / Trade-offs

- **[风险] attention 可能坍缩到少数 segment** → 缓解：训练完成后在 eval 中检查 attention 分布（top3_mass、effective_n）。如果坍缩，可以考虑重新引入 smoothness_loss 或 gated attention（论文中的门控版本）。
- **[风险] 旧 MIL checkpoint 完全不可用** → 缓解：这是 BREAKING change，需要重新训练 MIL。需要更新 pipeline 文档说明。
- **[权衡] 失去了 inst_logit 提供的 per-segment 绝对值** → 可以接受。attention 的职责更清晰（归因），terminal_reward 的职责更清晰（判断对错）。
