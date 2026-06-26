## Why

当前 MIL 训练使用了四个 loss 组件（bag_bce + instance_bce + temp_ce + smoothness），其中 instance_bce 依赖模型自身的 inst_logit 排名作为伪标签（"top-k 最高分 = 错误实例"），是一个自循环的不可靠信号。Ilse et al. (2018) "Attention-based Deep MIL" 证明：只用 bag-level BCE 训练的 attention 就能学到有意义的 instance importance。此外，`inst_head` 和 `temp_head` 的输出在 PPO shaping 中不可评估且不可靠。简化为纯 bag_bce + attention 可以消除这些没有理论支撑的组件，同时让 MIL 的 attention weight 成为 PPO credit assignment 的可解释信号。

## What Changes

- **删除** instance loss（topk/pure/soft_pseudo_label/contrastive）及相关代码
- **删除** `DynamicTempHead` 和 `GlobalTempHead` 及相关 temp_ce loss
- **删除** `smoothness_loss`（attention 坍缩未被证实是问题，论文中无不必要正则化）
- **删除** `inst_head` — MIL model 只输出 `bag_logit` 和 `attn_w`
- MIL 评估简化为 bag-level 指标（AUC、calibration），移除 instance-level 指标
- PPO shaping reward 从 `shaping_coef × (1 - sigmoid(inst_logit))` 改为 **`shaping_coef × terminal_reward × attention_weight`**
- PPO 构建 batch 时累积全 chain 的 segment，调用 MIL 一次获得 attention 权重后分配

## Capabilities

### New Capabilities

- `mil-bag-only`: MIL 模型只通过 bag_bce 训练，使用 attention aggregation（对齐 Ilse et al. 2018）。评估仅使用 bag-level 指标。PPO 通过 `terminal_reward × attention_weight` 使用 MIL attention 进行 credit assignment。

### Modified Capabilities

- `ppo-online-generation`: PPO shaping reward 的计算方式从 instance-logit 改为 attention-weight-based credit assignment。

## Impact

- 影响文件：`mil/model.py`（删除 inst_head/temp_heads/smoothness）、`mil/training.py`（简化 loss）、`mil/eval.py`（简化评估指标）、`ppo/training.py`（PPO batch 构建改用 attention weight）
- **BREAKING**: 旧 MIL checkpoint 不再兼容（模型结构变更）
- 配置变更：删除 `mil.training.instance_loss`、`mil.training.alpha_temp`、`mil.training.beta_inst`、`mil.training.gamma_smooth` 等 key
- 不涉及 vLLM 或 segmenter 变更
