## Context

`batch_build_segment_obs_from_lp` 的 GPU 快速路径有三条分支：

| 分支 | 输出维度 |
|---|---|
| `fixed_window` + `mean` | `[1, obs_dim]` — `mean(dim=1)` 天然与 token 数无关 |
| `fixed_window` + `concat` | `[1, max_tok × obs_dim]` — **依赖 max_tok** |
| `step` / `punctuation` | 逐链 CPU，调 `segment_pooling` |

concat 快速路径目前用 `tok_vecs.reshape(B, -1)` 产出 `max_tok × obs_dim`。而 `segment_pooling(mode="concat")` 产出 `segment_size × obs_dim`（pad/truncate 到 segment_size）。两者仅在 `max_tok == segment_size` 时一致。

当前代码中该假设成立——PPO eval 每轮生成 `segment_size` 个 token，且 EOS/stop 链在上游被 `active[i][v] = False` 过滤。但这依赖上游行为，没有显式保证。

## Goals / Non-Goals

**Goals:**
- concat 快速路径输出维度恒为 `segment_size × obs_dim`，与 `segment_pooling` concat 路径完全一致
- 不引入性能退化（pad/truncate 是零拷贝或极小开销的 GPU 操作）

**Non-Goals:**
- 不修改 mean 快速路径（`mean(dim=1)` 天然正确）
- 不修改 step/punctuation 路径（逐链 CPU，已正确）
- 不修改 `segment_pooling` 本身

## Decisions

### Decision 1: 在 reshape 之前 pad/truncate 到 segment_size

**选择**：在 GPU 上将 `tok_vecs` 沿 token 维度 pad 或 truncate 到 `segment_size`，然后 `reshape(B, segment_size * obs_dim)`。

**替代方案 A**：加 assert 文档化假设。简单但脆弱——上游变化时静默行为不确定。

**替代方案 B**：删除快速路径，统一走逐链 CPU。彻底消除不一致，但 eval 场景下 batch 可达 1500+ 链，GPU 加速有意义。

**选择理由**：pad/truncate 在 GPU 上几乎是零开销（`torch.cat` 分配少量显存，`[:, :segment_size, :]` 是 view），保持了 GPU 加速优势，同时消除了隐式假设。

### Decision 2: 两端都处理（< segment_size pad，> segment_size truncate）

**选择**：`T < segment_size` 时 zero-pad，`T > segment_size` 时截断前 segment_size 个 token。`T == segment_size` 时无操作。

这与 `segment_pooling` concat 的行为精确匹配：短 segment zero-pad，长 segment truncate。

## Risks / Trade-offs

- **GPU 显存**：pad 操作可能临时分配 `[B, segment_size - T, obs_dim]` 的零张量。当 `B=1500`、`segment_size - T=200`、`obs_dim=64` 时约 7.7 MB，可忽略。
- **正确性**：pad/truncate 后的行为与逐链 `segment_pooling` concat 完全一致，已有 `test_segment_pooling_concat_padding` 和 `test_segment_pooling_concat_truncation` 覆盖 concat 语义。
