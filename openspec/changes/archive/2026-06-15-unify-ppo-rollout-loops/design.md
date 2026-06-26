## Context

`ppo/training.py` 的 `train_ppo` 函数包含两个 60 行的回滚循环：训练回滚（lines 201-270）和验证回滚（lines 380-420）。验证回滚是训练回滚的早期副本，创建后训练回滚经历了多次修复（添加 `build_segment_obs_from_lp`、修复 `return_logprobs`/`return_hidden`），但验证回滚从未同步更新，导致三个 bug（见 proposal）。

两个循环共享相同骨架：
```
for each round:
    1. 遍历活跃链 → 决定温度
    2. 批量 vLLM 生成
    3. 处理结果 → 更新状态
```

差异仅在于：(a) 温度决策用 sample 还是 argmax，(b) 训练需记录 PPO 轨迹数据，(c) 训练需构建 `segment_obs` 而验证目前缺失此步骤。

## Goals / Non-Goals

**Goals:**
- 将温度决策逻辑提取为单一函数，参数化 deterministic/sampling 模式
- 将生成后处理逻辑提取为单一函数（构建 segment_obs、检测终止）
- 训练和验证均调用这两个函数
- 修复验证回滚的三个 bug
- 零配置变更、零 API 变更

**Non-Goals:**
- 不提取 PPO batch 构建逻辑（lines 272-321，纯训练逻辑）
- 不改造 `ppo/eval.py` 的 `OnlineTemperatureEvaluator`（那是评估脚本，不是训练验证）
- 不添加新测试（现有 `tests/test_ppo_model.py` 覆盖 PPO 模型单元；回滚集成测试需要 GPU，超出范围）

## Decisions

### Decision 1: 两个独立函数而非一个统一循环

**选择**: 提取 `_decide_temperature()` 和 `_process_generated_features()` 两个函数，保持训练和验证的循环在 `train_ppo` 中内联。

**替代方案**:
- *单一 `_run_rollout_round()` 统一函数*：需要传入 callback 或 mode 参数来处理 PPO 轨迹记录、argmax vs sample、是否返回 segment_obs 等差异点。参数爆炸，且训练特有的 PPO batch 构建（lines 272-321）仍需要留在外部，统一收益有限。
- *继承（BaseRollout → TrainRollout, ValRollout）*：两个 60 行的循环不值得引入类层次结构。函数粒度刚好。

**理由**: 两个小函数封装了最容易出错的"生成→特征提取→状态更新"路径。保持训练和验证循环独立可以清楚地看到各自特有的逻辑（训练记录 PPO buffer，验证用 argmax），避免复杂的 mode flag。

### Decision 2: `_decide_temperature` 的参数化方式

**选择**: 用 `deterministic: bool` 参数区分 sample（训练）和 argmax（验证）。

```python
def _decide_temperature(
    segment_obs: torch.Tensor | None,
    policy: PolicyValueNet,
    temp_bins: List[float],
    device: torch.device,
    deterministic: bool,
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
```

**替代方案**:
- *传入 `action_fn: Callable`*：更灵活但过度抽象。当前只有 sample 和 argmax 两种模式，且不太可能有第三种。

**理由**: `deterministic=True` 时 argmax，`deterministic=False` 时 `sample_action()`。返回 `(temp, action, logp, value)` 四个值，训练使用全部四个（记录 PPO buffer），验证只使用前两个。

### Decision 3: `_process_generated_features` 封装构建 segment_obs

**选择**: 该函数处理单个 chain 的生成结果，返回 `(text_delta, is_done, next_segment_obs | None)`。

**理由**: `build_segment_obs_from_lp` 的调用逻辑（包括 extra_parts 的 None 处理、pooling_mode、include_topk 推导）是当前验证回滚缺失的核心步骤。封装进共享函数确保训练和验证永远使用相同的特征构建路径。

## Risks / Trade-offs

- **[风险] 训练回滚行为回归** → 缓解：训练回滚的核心逻辑不变，重构只是将代码移入函数。改动后立即运行 `GPU_DEVICES=0,1 STAGES=ppo bash scripts/run_pipeline.sh` 验证。
- **[风险] 验证 `val_acc` 语义变更** → 缓解：这是预期行为。val_acc 从"固定 T=0.7"变为"policy argmax"，更准确地反映 policy 质量。需要在变更日志中说明这一点。
- **[权衡] 训练回滚的 `ep_obs/actions/logprobs/values` 记录仍在循环内** → 可以接受，因为这些 PPO 专有数据结构不应污染共享函数。
