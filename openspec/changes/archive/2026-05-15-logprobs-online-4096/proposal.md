## Why

1. `instance_dim=64` 导致 hidden states (4096-dim) 被 `token_to_vec` 截断丢弃 98%。上调到 4098 保留完整 hidden，同时让 top-4096 logprobs 提供分布形状信息逼近 hidden 的表征能力。
2. `topk_logits` 历史命名误导——实际存储的始终是 log-probabilities（来自 `return_logprob` API），不是 logits。
3. logprobs（每 token 4096 floats）和 hidden states 一样占空间，同样应该在线计算、用完即弃。

## What Changes

- **BREAKING**：`data.instance_dim` 从 64 改为 4098；`inference.top_k_logits` 从 16 改为 4096
- 全局重命名：`topk_logits` → `topk_logprobs`（配置、schema、runner、PPO、测试）
- `build_dataset` 不再写入 logprobs 到 JSONL — `TokenFeature.topk_logprobs` 字段始终为 `None`
- `SGLangRunner.extract()` 改为返回 `(hidden_tensors, logprob_tensors)` — 一次 prefill 同时拿两样
- `BagDataset` 在线提取时，extract 同时返回 hidden + logprobs，pool 为 instance

## Capabilities

### Modified Capabilities

- `sglang-hidden-extraction`: `extract()` 增加 logprobs 返回值，engine prefill 同时获取 hidden states + top-k logprobs
- `mil-online-hidden-extract`: 语义扩展为在线获取 hidden + logprobs

## Impact

- 修改 ~15 个文件（重命名触及面广）
- `instance_dim=4098` → MIL 模型参数膨胀 ~64x（525K，仍然很小）
- JSONL 大小减小（不再存 logprobs）
- 训练启动多 ~10s（prefill 同时拿 hidden + logprobs vs 只拿 hidden）
