## Why

per-token hidden states（4096-dim bf16）的 safetensors 侧车文件随数据集线性膨胀——500 题 15 温 8 票即可达 ~120GB，完全阻塞规模化。但 MIL 训练/评估中 raw hidden states 仅在 `segment_pooling` 的一次 mmap 遍历中被消费、压缩为 64-dim 后再不使用。将 hidden states 的提取从 build 阶段移到 MIL 训练阶段，用 SGLang engine 批量 prefill 按需提取并立即压缩，可消除磁盘存储瓶颈，同时保留换 segmentation 的自由度。

## What Changes

- **BREAKING**：`scripts/build_dataset.py` 在 `feature_mode=hidden_states/all` 时不再写 `.hidden.safetensors` 文件，JSONL 行也不含 `_hidden_offset/_hidden_count`
- **BREAKING**：`mil/training.py` 的 `BagDataset` 不再从侧车文件 mmap 加载 hidden states，改为在 `__init__` 中使用 SGLang engine 批量 prefill 所有 sample 的 `prompt+response`，对每个 response 提取 hidden states 并立即做 `segment_pooling` → 64-dim instance tensor，存储于 CPU 内存
- `mil/eval.py` 同理改为按需提取
- `feature_mode` 依然控制 `token_to_vec` 的行为（basic vs. hidden_states 特征），但对 hidden states 的获取方式从"读文件"变为"engine prefill"
- 删除 `utils/dataset_io.py` 中 hidden 侧车相关函数（`write_hidden_sidecar`、`read_hidden_offsets`、`split_hidden_sidecar`），仅保留 JSONL 读写工具

## Capabilities

### New Capabilities

- `mil-online-hidden-extract`: MIL 训练/评估时用 SGLang engine 批量 prefill 在线提取 per-token hidden states，立即 pooling 为 segment 向量，不落盘

### Modified Capabilities

- `binary-hidden-storage`: 移除 safetensors 侧车存储，hidden states 不再作为 dataset artifact 持久化
- `sglang-hidden-extraction`: 继承 SGLang engine 的 hidden 提取能力，在 MIL 训练上下文中复用

## Impact

- 磁盘占用从 ~120GB (500 题) 降为 <1MB (JSONL only)
- 训练启动多 ~10-30 分钟（一次性批量 prefill 提取）
- 删除 ~100 行侧车 I/O 代码
- 换 segmentation 只需改 seg 参数、重新 run MIL，不需要重新 build dataset
