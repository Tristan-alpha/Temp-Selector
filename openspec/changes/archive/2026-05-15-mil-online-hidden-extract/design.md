## Context

当前 safetensors 侧车存储 4096-dim per-token hidden states，500 题即可达 ~120GB。但实际上 hidden states 在 MIL 训练中只在 `segment_pooling` 中被消费一次，随即被压缩为 64-dim segment 向量。将提取从 build 阶段移至训练阶段可完全消除磁盘瓶颈。

## Goals / Non-Goals

**Goals:**
- 消除 safetensors 侧车文件，dataset 仅包含 JSONL
- MIL 训练/评估启动时用 SGLang engine 批量 prefill 提取 hidden states
- 换 segmentation 策略无需重新 build
- engine 由训练脚本管理，BagDataset 不负责创建/销毁

**Non-Goals:**
- 改变 JSONL 格式中 token 级特征（logprobs / entropy / top-k 保持不变）
- 改变 `segment_pooling` 或 `token_to_vec` 的逻辑
- 支持 vLLM 后端（仅 SGLang 支持单 engine 在线提取）

## Decisions

### Decision 1: engine prefill 提取 hidden states，不存盘

**Chosen**: `BagDataset.__init__` 接收外部 `sglang.Engine` 对象，对 JSONL 中所有 `(prompt, response)` 对做批量 prefill，提取 hidden states 并立即 pooling。

**Rationale**: 消除了 TB 级的磁盘需求。SGLang engine 的 `return_hidden_states=True` 在一次 prefill 中同时完成 tokenization 和 hidden state 提取。token 级结果只活在内存中，pooling 完毕即释放。

```
For each row in JSONL:
  engine.generate(prompt+response, max_new_tokens=1, return_hidden_states=True)
  → hs = output["meta_info"]["hidden_states"][prompt_len:]   # [n_tokens, 4096]
  → segment_pooling(token_vecs, spans, ...)                    # [n_segments, 64]
  → self.rows.append((instances, label, temp_idx))
  → hs = None  # released
```

### Decision 2: engine 外部注入，BagDataset 不管理生命周期

```python
class BagDataset(Dataset):
    def __init__(self, data_path, ..., engine=None):
        if engine is not None and feature_mode in {"hidden_states","all"}:
            # Use engine for hidden extraction
        else:
            # Logprob-only features (existing path)
```

`train_mil()` 创建 engine，传给 train + val BagDataset，训练结束 shutdown。

### Decision 3: 保留 JSONL 中 token 级特征

JSONL 中 `token_features` 依然包含 `logprob`、`entropy`、`topk_logits`、`text`。这些是生成过程的副产品（vLLM/SGLang 的 logprob API），体积极小（~16KB/sample），不构成存储瓶颈。`hidden` 字段统一为 `null`。

### Decision 4: `SGLangHiddenStateExtractor` 薄封装

**Chosen**: 创建 `inference/sglang_hidden_extractor.py`，提供 `SGLangHiddenStateExtractor` 类。接收外部 `sglang.Engine`，暴露 `extract(prompts, responses) -> List[torch.Tensor]` 接口。内部仅做参数转发，不处理分批——SGLang engine 自身的 scheduler 负责 batch 调度。

**Rationale**: 与 vLLM extractor 同名同接口，降低切换成本。分批是调用方的内存管理需求，不属于 extractor 的职责。

### Decision 5: 调用方（BagDataset / train_mil）负责分批

**Chosen**: `BagDataset.__init__` 或 `train_mil()` 按 `batch_size` 将 JSONL 行分组，每组调用一次 `extractor.extract()`，立即 pool → instance tensor → 释放 hidden states。`batch_size` 通过 config（`mil.training.hidden_batch_size`，默认 256）控制。

**Rationale**: 60,000 samples × 4096-dim 的中间结果可达数 GB，由调用方按内存预算分批。extractor 保持简单。

```python
for batch_rows in chunks(all_rows, batch_size):
    hs_tensors = extractor.extract(batch_prompts, batch_responses)
    for hs, row in zip(hs_tensors, batch_rows):
        instances = segment_pooling(...)  # pool → [n_segments, 64]
        self.rows.append((instances, label, temp_idx))
    # hs_tensors freed after batch
```

## Risks / Trade-offs

- **训练启动延迟**: 全量提取 60,000 samples × 4096-dim ≈ 分批处理。`batch_size` 默认 256，每批 ~30s，全量 ~30min。可接受的一次性开销。
- **内存控制**: 通过 `batch_size` 控制 GPU/CPU 峰值。默认 256 时每批 hidden states ≈ 256 × 256 tokens × 4096 × 2 bytes ≈ 500MB bf16，pooling 后只有 256 × 4 segments × 64 × 4 bytes ≈ 256KB，可以忽略。
- **多 epoch**: instance tensor 存于 `self.rows` 的 CPU 内存中，跨 epoch 复用，不重复提取。
- **GPU 占用**: engine 在训练全过程中占用 GPU。MIL model 本身 ~500K 参数，GPU 大部分空闲，无冲突。
- **SGLang 依赖**: 此方案依赖 SGLang。vLLM 后端不再支持 hidden states（需双实例），但 basic/topk_logits 模式无需 engine。
