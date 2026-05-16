## Context

当前 `inference/` 目录有 5 个文件处理不同 backend，实际只有 SGLang 和 vLLM 两个在用。隐藏状态提取逻辑分散在 sglang_hidden_extractor 和 vllm_hidden_extractor 中，各阶段的 SGLang 调用也绕过 runner 裸用 Engine。

## Goals / Non-Goals

**Goals:**
- 合并 `SGLangFeatureExporter` + `SGLangHiddenStateExtractor` → 单一 `SGLangRunner`
- 删除 api_runner、vllm_hidden_extractor、sglang_hidden_extractor
- PPO/MIL/Build 全阶段通过 runner 接入
- 修复 hidden state 切片偏移

**Non-Goals:**
- 改变 vllm_runner 接口
- 修改 JSONL 格式或 MIL/PPO 算法

## Decisions

### Decision 1: unified SGLangRunner with generate() + extract()

`SGLangRunner` 合并 exporter + extractor 的功能：

```
SGLangRunner(model_path, max_new_tokens, tp_size, gpu_mem, feature_mode)
  .generate(prompts, temperatures, top_k_logits, num_votes)
  .extract(prompts, responses)
```

内部使用单一 `sglang.Engine`。`feature_mode` 控制 `enable_return_hidden_states`。

### Decision 2: extract 切片 offset = prompt_len - 1

`h[i]` 是产生 `token[i+1]` 的 hidden state。要获取 response token 的 hidden states，需要从 `prompt_len - 1` 开始切片。

### Decision 3: 所有阶段通过 runner，不裸用 Engine

PPO training 直接创建 `SGLangRunner`，其 `generate()` 方法处理段级生成（带 `return_hidden_states` 内联提取）。MIL training 用 `runner.extract()` 做在线 hidden 提取。

### Decision 4: vLLM runner 拒绝 hidden_states 模式

`VLLMFeatureExporter.__init__` 收到 `feature_mode in {"hidden_states", "all"}` 时直接 `raise ValueError`。vLLM 单 engine 不支持 hidden state 提取，尽早报错避免运行时崩溃。

### Decision 5: `tensor_parallel_size` → `parallel_size`

所有 config 和代码中的 `tensor_parallel_size` 改为 `parallel_size`。SGLang 当前使用 data parallel（`dp_size`），vLLM 使用 tensor parallel（`tp_size`）——统一语义为 `parallel_size`，底层由各 backend 按需要映射。

## Files

| 操作 | 文件 |
|---|---|
| DELETE | `inference/api_runner.py` |
| DELETE | `inference/vllm_hidden_extractor.py` |
| DELETE | `inference/sglang_hidden_extractor.py` |
| REWRITE | `inference/sglang_runner.py` → 合并 extract() |
| MODIFY | `scripts/build_dataset.py` → 移除 api/vllm-extractor |
| MODIFY | `ppo/training.py` → 用 runner 替代裸 Engine |
| MODIFY | `mil/training.py` → runner.extract() |
| MODIFY | `mil/eval.py` → runner.extract() |
