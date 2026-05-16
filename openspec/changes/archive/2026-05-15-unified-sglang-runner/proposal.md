## Why

当前 inference 层碎片化严重——api_runner、vllm_hidden_extractor、sglang_hidden_extractor、sglang_runner 各管一块，切换 hidden state 提取方案需要在不同类之间跳转。ppo/training.py 甚至绕过 runner 裸用 Engine。统一为单一 SGLangRunner（generate + extract），删除三个文件，所有阶段通过 runner 统一入口。

## What Changes

- **BREAKING**：删除 `inference/api_runner.py`、`inference/vllm_hidden_extractor.py`、`inference/sglang_hidden_extractor.py`
- `inference/sglang_runner.py` → 合并 extract 能力：`SGLangRunner.generate()` 和 `SGLangRunner.extract(prompts, responses)`，根据 `self.feature_mode` 控制 hidden state 导出
- 修复 hidden state 切片偏移：`hs[prompt_len:]` → `hs[prompt_len - 1:]`（h[i] 对应 token[i+1]）
- `scripts/build_dataset.py`：移除 api/vllm extractor 路径，统一用 runner
- `ppo/training.py`：用 runner 替代裸 `sglang.Engine`
- `mil/training.py`、`mil/eval.py`：用 `runner.extract()` 替代 `SGLangHiddenStateExtractor`
- `configs/`：移除 api 相关配置段

## Capabilities

### New Capabilities

- `unified-sglang-runner`: `SGLangRunner` 统一提供 generate（token 特征）和 extract（hidden states）两种能力，根据 `feature_mode` 控制行为

### Modified Capabilities

- `sglang-hidden-extraction`: 功能合并入 `SGLangRunner.extract()`，独立 extractor 类不再存在

### Removed Capabilities

- `mil-online-hidden-extraction` 的 `SGLangHiddenStateExtractor` 独立类

## Impact

- 删除 3 文件，修改 5 文件
- 内聚性提升：所有 SGLang 交互通过 `SGLangRunner` 单一入口
- PPO 训练不再裸用 `Engine`，feature_mode 在 runner 初始化时固定
