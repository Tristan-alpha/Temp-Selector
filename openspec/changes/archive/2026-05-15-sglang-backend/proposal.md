## Why

vLLM 0.18 的 `extract_hidden_states` 需要第二个 LLM 实例（带 speculative config），与生成 LLM 在 GPU 和 EngineCore 多进程上冲突。多次尝试 sleep/wake_up、destroy/recreate 均不稳定。SGLang 原生支持 `return_hidden_states=True`，一个 engine 同时完成生成和 hidden state 提取，无需双实例 hack。

## What Changes

- 新增 `inference/sglang_runner.py`：SGLang 后端，封装 `SGLangFeatureExporter`，支持多温生成 + `return_hidden_states`
- 新增 `inference/sglang_hidden_extractor.py`（或合并入 runner）：从 SGLang 输出中提取 per-token hidden states，返回 `List[torch.Tensor]`（与 vLLM extractor 同接口）
- **BREAKING**：`scripts/build_dataset.py` 和 `ppo/training.py` 默认后端从 vLLM 切换到 SGLang，vLLM 降级为可选
- 新增 `configs/dataset_sglang.yaml`：SGLang 专用配置
- 新增 `requirements.txt` 添加 `sglang>=0.4`
- 删除 PPO 训练中的 LLM destroy/recreate/sleep 逻辑

## Capabilities

### New Capabilities

- `sglang-hidden-extraction`: SGLang engine 通过 `return_hidden_states=True` 在生成时直接获取 per-token hidden states，不需要第二个实例

### Modified Capabilities

- `binary-hidden-storage`: 数据格式不变（JSONL + safetensors sidecar），但 hidden state 来源从 vLLM speculative decoding 改为 SGLang 原生 API

## Impact

- 新增 3-4 文件，修改 4-5 文件
- 新增依赖 `sglang`
- 移除 PPO 中所有 LLM 双实例相关 hack 代码
- vLLM 后端保留但降级为 `--backend vllm` 选项
