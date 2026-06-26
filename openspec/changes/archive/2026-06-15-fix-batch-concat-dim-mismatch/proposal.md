## Why

`batch_build_segment_obs_from_lp` 的 concat 快速路径用 `max_tok × obs_dim` 作为输出维度，但逐链路径（`segment_pooling` concat）用 `segment_size × obs_dim`。两条路径仅在 `max_tok == segment_size` 时等价，这个隐式假设没有被任何断言或注释保护。一旦上游过滤逻辑变化或出现边界情况，两条路径的输出维度不一致，下游模型会收到错误形状的输入。

## What Changes

- **修复 concat 快速路径**：在 `reshape` 之前将 `tok_vecs` pad/truncate 到 `segment_size` 个 token，使输出维度恒为 `segment_size × obs_dim`，与逐链 `segment_pooling` concat 路径完全一致

## Capabilities

### New Capabilities
<!-- No new capabilities — this is a bug fix within existing behavior -->
<!-- Leave empty -->

### Modified Capabilities

- `vectorized-pooling`: 修订 `batch_build_segment_obs_from_lp` concat 快速路径的输出维度要求，从隐式 `max_tok × obs_dim` 改为显式 `segment_size × obs_dim`
- `gpu-batch-segment-obs`: 修订 concat 快速路径的内部实现约束，确保与 `segment_pooling` concat 语义一致

## Impact

- `features/segmenter.py` — `batch_build_segment_obs_from_lp` concat 快速路径（~5 行改动）
- 不影响 mean pooling 路径、step/punctuation 路径、逐链 CPU 路径
- 不影响任何调用方（输出形状不变时行为不变；边界情况下从错误变为正确）
