# tf-mil 完整 Pipeline 文档

## 项目概述

`tf-mil`（Temperature Framework -- Multiple Instance Learning）研究**LLM 推理时动态温度选择**。核心假设是：

1. 数学推理链由若干步骤（segment）组成，不同步骤可能需要不同的生成温度
2. 错误推理链中**至少存在一个错误 segment**（正 bag），正确推理链中所有 segment 都正确（负 bag）
3. 可以用 MIL 定位错误 segment，再用 PPO 学习"根据 segment 特征选择温度以避免错误"

### 核心思想

```
                     ┌──────────────────────────┐
                     │  Stage 1: 多温度数据采集   │
                     │  15 temps × N prompts     │
                     │  × num_votes 投票         │
                     └────────────┬─────────────┘
                                  │ BagSample (含 segment 特征 + label)
                                  ▼
                     ┌──────────────────────────┐
                     │  Stage 2: MIL 错误定位    │
                     │  InstanceEncoder          │
                     │  + PositionEncoding       │
                     │  + BiGRU (错误传播建模)    │
                     │  + AttentionAggregator    │
                     │  → bag_logit (错误概率)    │
                     │  → inst_logit (每段错误分)  │
                     └────────────┬─────────────┘
                                  │ warm-start backbone
                                  │ inst_logit → shaping reward
                                  ▼
                     ┌──────────────────────────┐
                     │  Stage 3: 在线 PPO 训练    │
                     │  vLLM 逐段生成             │
                     │  PPO policy 选温度         │
                     │  Majority Voting reward    │
                     └────────────┬─────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │  Stage 4: 在线评估         │
                     │  PPO vs best-fixed vs random│
                     │  三种策略同等条件下对比     │
                     └──────────────────────────┘
```

### 目录结构

```
tf-mil/
├── configs/                    # YAML 配置（base + 4 份对比实验变体）
├── data/                       # 原始输入数据
│   ├── prompts.example.jsonl   # 输入数据格式示例
│   └── math.jsonl              # 实际输入数据
├── datasets/                   # 处理后的数据集
│   ├── all.jsonl               # Stage 1 输出的完整数据集
│   ├── train.jsonl             # Stage 1b 训练集
│   ├── eval.jsonl              # Stage 1b 评估集
│   └── cache/                  # 缓存目录
├── checkpoints/                # 模型权重
│   ├── mil_ckpt.pt             # Stage 2 输出的 MIL checkpoint
│   └── ppo_ckpt.pt             # Stage 3 输出的 PPO checkpoint
├── features/                   # Stage 1：数据集构建
│   ├── build_dataset.py        # 主入口：特征提取 + 多数投票 + BagSample 构造
│   ├── schema.py               # TokenFeature, Segment, BagSample 数据结构
│   ├── segmenter.py            # 分段策略 + segment_pooling（mean、concat）
│   ├── vectorizer.py           # token_to_vec, token_to_obs, mean_pool_obs, compute_entropy
│   └── dataset_eval.py         # evaluate_dataset(), load_temperature_labels()
├── inference/                  # LLM 推理后端
│   ├── vllm_runner.py          # VLLMFeatureExporter：vLLM 批量生成 + 多温 APC
│   └── api_runner.py           # APIFeatureExporter：百炼 API 后端
├── mil/                        # Stage 2：MIL 错误定位
│   ├── model.py                # MILModel + temp heads + smoothness_loss（全部 MIL 模型定义）
│   ├── training.py             # BagDataset, top-k MIL loss, train_mil()
│   └── eval.py                 # evaluate_mil() + 所有 MIL 指标函数
├── ppo/                        # Stage 3：在线 PPO 训练
│   ├── model.py                # PolicyValueNet, compute_gae, MIL warm-start
│   ├── training.py             # train_ppo() + 在线特征提取
│   └── eval.py                 # OnlineTemperatureEvaluator（PPO vs best-fixed vs random）
├── utils/                      # 跨阶段共享基础设施
│   ├── math.py                 # safe_div
│   ├── answer_verifier.py      # Math-Verify 答案验证包装器
│   └── exp_logger.py           # 文件 + 流日志
├── scripts/
│   ├── run_pipeline.sh         # 全流程编排（支持 STAGES 选择）
│   ├── stage1_build.sh         # 单独 Stage 1
│   ├── stage2_mil.sh           # 单独 Stage 2
│   ├── stage3_ppo.sh           # 单独 Stage 3
│   ├── stage4_eval.sh          # 单独 Stage 4
│   ├── split_jsonl.py          # train/eval 数据集划分（group-aware）
│   └── subsample_jsonl.py      # 数据集下采样
├── tests/                      # 单元测试（8 个文件，~80 用例，全部 CPU-only）
├── logs/                       # 运行日志
└── PIPELINE.md                 # 本文档
```

---

## 环境准备

### 依赖

```
torch>=2.2.0
pyyaml>=6.0.1
numpy>=1.26.0
vllm>=0.5.0
math-verify[antlr4_13_2]>=0.8.0
openai>=1.0.0          # API 后端需要
```

### 输入数据格式

`data/math.jsonl`，每行一道数学题：

```json
{"unique_id":"q1","problem":"If x + 2 = 5, what is x?","answer":"3","source":"math"}
```

| 字段 | 必需 | 备注 |
|---|---|---|
| `unique_id` 或 `sample_id` | 是 | 唯一标识符 |
| `problem` 或 `question` 或 `prompt` | 是 | 题目文本 |
| `answer` | 是 | 标准答案（纯表达式，不含 `\boxed{}`） |
| `source`, `subject`, `level` | 否 | 元信息 |

---

## 核心概念：标签语义

整个项目的标签采用**错误定位视角**：

| label | 含义 | MIL 术语 |
|---|---|---|
| `0` | 回答**正确**，所有 segment 都无错误 | **负 bag**（无正 instance） |
| `1` | 回答**错误**，至少一个 segment 出错 | **正 bag**（至少一个正 instance） |

这与传统 MIL（正 = 正常、负 = 异常）相反。翻转是为了让 MIL 的 attention 机制自然地聚焦于"最像错误的 segment"，直接服务于**错误定位**目标。

### Majority Voting

每个 (prompt, temperature) 组合生成 `num_votes` 路回答。通过 **self-consistency** 多数投票决定 label：

1. 用 `math_verify.parse()` 从每票回答中提取数学答案
2. 取出现次数最多的众数答案（plurality）
3. 众数答案等同于标准答案 → label=0（正确），否则 label=1（错误）

```
4 票提取答案: ["3", "3", "7", "3"] → 众数 "3" = gold "3" → label=0
4 票提取答案: ["7", "3", "3", "7"] → 众数 "7" ≠ gold "3" → label=1
```

所有 vote 的 BagSample 共享同一个 majority label，但 metadata 中保留各票的 `individual_correct`（per-vote `verify_answer` 结果，辅助参考，不参与 label 决策）。

---

## Stage 1：特征提取与数据集构建

### 入口

```bash
# vLLM 后端（默认）
CUDA_VISIBLE_DEVICES=0 python -m features.build_dataset --config configs/base.yaml

# API 后端（百炼 DashScope）
python -m features.build_dataset --config configs/base.yaml --backend api
```

路径从 config 的 `paths` 段读取；CLI 参数 `--input` / `--output` 可覆盖。

### 流程

```
输入 JSONL (N 条 prompt)
        │
        ▼
┌─ vLLM 后端：一次性提交全部 N × 15 条请求 ──────────────────┐
│  提交顺序: prompt₀@T₀, prompt₀@T₁, ..., prompt₀@T₁₄,       │
│            prompt₁@T₀, ...                                  │
│  APC 自动共享同一 prompt 在不同温度间的 KV-cache             │
│  每请求: SamplingParams(n=num_votes, temperature=T, ...)    │
└────────────────────────────────────────────────────────────┘
        │
        ▼
对每条 prompt × 温度 × vote:
  ├── 1. Math-Verify (parse + verify) 验证答案
  ├── 2. 提取 per-token 特征: logprob, entropy, top-16 logits
  ├── 3. fixed_window_segment(32 token/段) 切分 token 序列
  │
  └── 4. 同 (prompt, temperature) 的 num_votes 票: 多数投票
         │  label = 0 if ≥ ceil(k/2) 票正确 else 1
         │  每票生成一个 BagSample，共享 majority label
         └── metadata: vote_id, num_votes, individual_correct, votes_correct
```

### 输出

`datasets/all.jsonl`：

```
行数 = N × 15 × num_votes
sample_id 格式: {base}_t{temp}             (num_votes=1)
                {base}_t{temp}_v{v}        (num_votes>1)
```

### BagSample 数据结构

```python
@dataclass
class BagSample:
    sample_id: str                        # 唯一标识
    prompt: str                           # 原始题目
    response: str                         # LLM 生成的回答文本
    label: int                            # 0=正确, 1=错误 (majority vote)
    temperature: float                    # 生成时使用的温度
    token_features: List[TokenFeature]    # per-token 特征
    segment_spans: List[Segment]          # 分段边界 [{start, end}, ...]
    metadata: Dict[str, Any]              # vote_id, individual_correct, ...

@dataclass
class TokenFeature:
    token_id: int                         # vLLM token id
    text: str                             # 解码后的 token 文本
    logprob: float                        # 该 token 的 log 概率
    entropy: float                        # 从 top-k logprobs 计算的熵
    topk_logits: Optional[List[float]]    # top-16 logprobs

@dataclass
class Segment:
    segment_id: int
    start: int                            # token 起始位置
    end: int                              # token 结束位置
```

### 特征向量构造

每个 token 的特征向量为：

```
[logprob, entropy, top16_logprob₁, ..., top16_logprob₁₆]
  → padding 或截断到 instance_dim=64
```

每个 segment 的 instance 向量 = segment 内所有 token 向量的 **mean-pool**。

### vLLM 优化

- **多温度一次性提交**: N×15 条请求在单次 `llm.generate()` 中提交，同一 prompt 的 15 个温度变体通过 APC 共享 KV-cache
- **`max_model_len = max_new_tokens + 2048`**: 限制 KV-cache 预分配范围
- **`enable_thinking=False`**: 通过 chat template 关闭 Qwen3 思考模式，避免超 token
- **`use_tqdm`**: Stage 1 保留进度条（单批生成），在线 PPO/评估禁用

---

## Stage 1b：数据集划分

### 入口

```bash
python scripts/split_jsonl.py --config configs/base.yaml
```

路径从 config `paths` 段读取；默认 `--val-ratio 0.1 --test-ratio 0.1`（两个 ratio 均以全量数据为基准），三路 group-aware 划分。

### 逻辑

`--group-by sample_prefix`：同一 prompt 的所有温度变体和所有票**作为一个 group**整体分入同一 split，防止数据泄漏。`sample_prefix()` 函数从 `q1_t0.2_v0` 提取 `q1` 作为分组键。

### 输出

- `datasets/train.jsonl`：~80% 的 group（训练）
- `datasets/val.jsonl`：~10% 的 group（early stopping + best-fixed temp 选择）
- `datasets/test.jsonl`：~10% 的 group（最终评估指标）

---

## Stage 2：MIL 模型训练 — 错误定位

### 入口

```bash
python -m mil.training --config configs/base.yaml
```

路径从 config 读取，无需额外参数。

### 数据预处理

```
BagDataset 加载 JSONL:
  对每个 BagSample:
    1. token_to_vec(): [logprob, entropy] + topk_logits → padding 到 64 维
    2. segment_pooling(): 每个 segment 内 mean-pool → [K, 64] 的 instance 矩阵
    3. 构造 (instances [K, 64], label {0=正确,1=错误}, temp_idx {0..14})

DataLoader collate:
  batch 内 zero-padding 到最大 segment 数 + mask 标记有效 instance
```

### 模型架构

```
MILModel (错误定位器)
│
├── InstanceEncoder
│   Linear(64→256) → ReLU → Linear(256→256) → ReLU
│   输出: [B, K, 256]
│
├── SinusoidalPositionalEncoding (可选)
│   可学习的位置编码，让模型感知 segment 在推理链中的位置
│
├── BiGRU (可选, bidirectional)
│   GRU(256→256, bidirectional) → Linear(512→256)
│   建模错误在 segment 间的传播模式
│
├── AttentionAggregator
│   score = Linear(256→1) → softmax → w
│   bag_repr = Σ wᵢ · hᵢ
│   attention 权重 wᵢ 指示「哪个 segment 最像错误」
│
├── bag_head: Linear(256→1)        → bag_logit (整个回答是错的概率)
└── inst_head: Linear(256→1)       → inst_logit (每个 segment 是错的分数)

参数量:
  基础 (no pos, no GRU): ~150K
  + position encoding:     ~150K (buffer, 不训练)
  + BiGRU:                ~500K (GRU weights + projection)
```

### 损失函数

```
Total Loss = bag_bce                                         # ① 主任务
           + α × (global_temp_ce + dynamic_temp_ce) / 2     # ② 温度辅助
           + β × instance_bce                                # ③ 实例辅助
           + γ × smoothness_loss                              # ④ 正则项
```

**① bag_bce — 预测回答是否为错误（主任务）**

```
BCEWithLogitsLoss(bag_logit, label)
    pos_weight = sqrt(n_correct / n_wrong)   # 平衡正负 bag（错误 ~33%）
```

**② 温度分类（辅助，α=0.1 低权重）**

- `global_temp_ce`: 从 bag_repr 预测 15 分类温度 bin
- `dynamic_temp_ce`: 从 per-instance GRU 输出预测温度
- `smoothness_loss`: 惩罚相邻 segment 温度预测剧烈变化

**③ instance_bce — 实例辅助 loss（可配置）**

由 `mil.training.instance_loss` 控制，四种方法可选：

| 方法 | 正 bag | 负 bag | 参考文献 |
|---|---|---|---|
| `pure` (默认) | k=1，最高分 segment→target=1 | 全→target=0 | [FocusMIL 2024](https://arxiv.org/abs/2408.09449) |
| `topk` | k=n_valid//3，最高 k 个→target=1 | 全→target=0 | 原有实现 |
| `soft_pseudo_label` | target=sigmoid(inst_logit).detach()，反退化 clamp | 全→target=0 | [SeLa-MIL 2024](https://arxiv.org/abs/2408.04813) |
| `contrastive` | logsumexp(scores)-max(scores) | scores².mean() | [NDI-MIL 2025](https://ieeexplore.ieee.org) |

对于负 bag：无论哪种方法，所有 instance target=0。正确回答中每个 segment 都应该是"无错"的。

对于正 bag：核心挑战是我们只有 bag 级标签（整个回答对/错），不知道具体哪个 segment 出错。不同方法的差异在于如何分配 instance 级训练信号。

**④ 正则项（γ=0.05）**

`smoothness_loss = mean((logit_{t+1} - logit_t)²)` 鼓励 GRU 输出平滑的温度梯度。

### 训练超参数

```yaml
mil:
  model:
    hidden_dim: 256
    aggregator: attention
    use_position: true         # 正弦位置编码
    use_gru: true              # 双向 GRU 错误传播建模
  training:
    instance_loss: pure        # pure | topk | soft_pseudo_label | contrastive
    batch_size: 32
    lr: 2.0e-4
    max_epochs: 50
    early_stop_patience: 5     # stop if val inst_logit_separation doesn't improve for N epochs
    alpha_temp: 0.1            # 温度分类权重 (压低，因温度区分困难)
    beta_inst_aux: 0.2         # instance 辅助 loss 权重
    gamma_smooth: 0.05         # 平滑正则权重
```

### 输出

`checkpoints/mil_ckpt.pt`：

```python
{
    "mil": {                       # MILModel state_dict
        "encoder.net.0.weight",    # [256, 64]
        "encoder.net.0.bias",      # [256]
        "encoder.net.2.weight",    # [256, 256]
        "encoder.net.2.bias",      # [256]
        "pos_encoder.pe",          # [1, 512, 256] (buffer)
        "gru.weight_ih_l0", ...    # BiGRU params
        "gru_proj.weight", ...     # [256, 512] projection
        "attn_agg.attn.weight",    # [1, 256]
        "bag_head.weight",         # [1, 256]
        "inst_head.weight",        # [1, 256]
        ...
    },
    "global_head": {...},          # GlobalTempHead state_dict
    "dynamic_head": {...},         # DynamicTempHead state_dict
    "config": {...},
}
```

---

## Stage 3：在线 PPO 策略训练

### 入口

```bash
# 单卡（vLLM tensor parallelism 占满单卡）
CUDA_VISIBLE_DEVICES=0 python -m ppo.training --config configs/base.yaml

# 双卡（vLLM tensor parallelism 占满双卡，单进程）
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.training --config configs/base.yaml
```

路径和 MIL checkpoint 从 config 读取。**单进程运行**——PolicyValueNet 只有 ~150K 参数；vLLM 通过 tensor parallelism 利用多卡。

### 在线生成流程

```
每 iteration（80 轮）:
  1. 从训练集中随机采样 128 条 prompt
  2. 所有 prompt 同时逐 segment 生成:
     Round 0: [prompt] → PPO 选温度 T₀ → 生成 segment₀ (×num_votes)
     Round 1: [prompt+seg₀] → PPO 选温度 T₁ → 生成 segment₁ (×num_votes)  (APC✓)
     ...
     直到 EOS 或 max_tokens
  3. 用第一票的特征做 PPO 决策，所有票共享同一温度序列
  4. 终局: 多数投票决定 terminal reward (+1 正确 / -1 错误)
  5. 中间步: MIL inst_logit 作为 shaping reward
      reward = shaping_coef × (1 - sigmoid(inst_logit))
      (inst_logit 越高 = 越像错误 = 惩罚越大)
  6. GAE + PPO multi-epoch mini-batch 更新
```

### 模型

```
PolicyValueNet (3 层 MLP, ~150K 参数)
├── backbone: Linear(64→256)→ReLU→Linear(256→256)→ReLU→Linear(256→256)→ReLU
│   (前 2 层从 MIL InstanceEncoder warm-start 权重)
├── pi: Linear(256→15)         # 策略头，输出 15 个温度 bin 的 logits
│   (bias 初始化: best-fixed 温度=+5, 其余=-5 → 初始策略倾向于最优固定温度)
└── v:  Linear(256→1)          # 价值头，估计状态价值
```

### MIL 提供的三项

| 用途 | 来源 | 说明 |
|---|---|---|
| Backbone warm-start | `InstanceEncoder` → `backbone.0` & `backbone.2` | 1:1 映射前两层权重 |
| Shaping reward | `inst_head(segment)` → sigmoid → 1-sigmoid | segment 越不像错误，reward 越高 |
| pi head bias | 数据集统计 | 初始策略倾向 best-fixed 温度 |

### 过拟合诊断

每轮 PPO 更新后输出 `value`（训练集 80% MSE）和 `val_value`（验证集 20% MSE）：

| 现象 | 诊断 |
|---|---|
| `value ↓` 但 `val_value ↑` | 过拟合，策略死记当前 rollout |
| 两者都持续 `↑` | value 发散，降低 LR 或缩小 shaping_coef |
| 两者都缓慢 `↓` | 健康 |

### 训练超参数

```yaml
ppo:
  model:
    hidden_dim: 256              # PolicyValueNet backbone 维度
  training:
    max_iterations: 200           # 在线 rollout 总轮数上限
    early_stop_patience: 10      # val_value 不提升 N 轮即停
    online_rollout_size: 128     # 每轮 prompt 数
    ppo_epochs: 8                # 每轮 PPO 更新 epoch 数
    mini_batch_size: 32          # 每次梯度更新的 batch 大小
    val_ratio: 0.2               # rollout 保留 20% 验证过拟合
    gamma: 0.99                  # GAE 折扣
    gae_lambda: 0.95             # GAE λ
    clip_eps: 0.2                # PPO clip ε
    value_coef: 0.5              # value loss 系数
    entropy_coef: 0.005          # 熵正则系数
    lr: 1.0e-5                   # 学习率 (小模型+在线数据，保守)
    shaping_coef: 0.15           # MIL 塑形奖励权重
```

### PPO 日志格式

```
iter=N loss=X policy=Y value=Z val_value=V entropy=E reward=R acc=A steps=S updates=U
```

| 字段 | 含义 | 健康状态 |
|---|---|---|
| `loss` | 总 loss | 参考意义不大 |
| `policy` | PPO clipped policy loss | 偶尔非零 (±0.01~0.05) 表示策略在更新 |
| `value` | 训练集 value MSE | < 3，趋势下降或稳定 |
| `val_value` | 验证集 value MSE | 与 value 接近 |
| `entropy` | 策略熵 (max=2.71) | 前期缓慢升到 0.3~0.5 后稳定 |
| `reward` | 平均 step reward | 趋势上升 |
| `acc` | 本 iteration 正确率 (majority vote) | 趋势上升 (±4% 噪声正常) |
| `steps` | 总 PPO 步数 | ~2000 |
| `updates` | 总梯度更新次数 | ~400 |

### 输出

`checkpoints/ppo_ckpt.pt`：

```python
{
    "policy_value": {
        "backbone.0.weight",    # [256, 64]
        "backbone.0.bias",      # [256]
        "backbone.2.weight",    # [256, 256]
        "backbone.2.bias",      # [256]
        "backbone.4.weight",    # [256, 256]
        "backbone.4.bias",      # [256]
        "pi.weight",            # [15, 256]
        "pi.bias",              # [15]
        "v.weight",             # [1, 256]
        "v.bias",               # [1]
    },
    "config": {...},
}
```

---

## Stage 4：评估

### 离线评估（数据集统计 + MIL 模型）

```bash
# MIL 离线评估
python -m mil.eval --config configs/base.yaml
```

#### 数据集级指标

| 指标 | 说明 |
|---|---|
| `n_samples` | 总样本数 |
| `positive_ratio` | 正确例比例（label=0） |
| `per_temperature_breakdown` | 每个温度下的 majority/individual 正确率 |
| `majority_voting` | num_votes, n_groups, majority/individual 准确率, best_temperature |

#### MIL 模型指标

| 类别 | 指标 | 说明 |
|---|---|---|
| Bag 分类 | `bag_accuracy / precision / recall / f1` | 预测"回答是否为错误"的性能 |
| | `bag_auc` | ROC-AUC（梯形法则） |
| | `bag_tp / tn / fp / fn` | 混淆矩阵（tp=正确预测为错误） |
| 校准 | `ece` | Expected Calibration Error（10 分箱） |
| | `brier_score` | Brier Score |
| 温度分类 | `temp_accuracy` (global / dynamic) | 温度 bin 分类准确率 (基线 ~1/15=0.067) |
| | `temp_per_class` | 每温度 precision/recall/f1/support |
| | `temp_confusion_matrix` | 15×15 混淆矩阵 |
| | `dynamic_head_per_instance` | 逐 instance 温度分类指标 |
| | `dynamic_head_smoothness` | 逐样本 smoothness min/mean/max |
| | `dynamic_head_correctness_temp_distribution` | 正确/错误 bag 中各温度被预测的频率 |
| Instance | `inst_logit_mean_error_bags` | 错误 bag 的 instance error score 均值 |
| | `inst_logit_mean_correct_bags` | 正确 bag 的 instance error score 均值 |
| | `inst_logit_separation` | 两者差值（>0 说明模型能区分错误/正确 segment） |
| | `attn_entropy` | 注意力熵（越高越分散） |
| | `attn_top3_mass` | 前 3 instance 注意力质量占比 |
| | `attn_effective_n` | 有效 instance 数 `1/Σw²` |
| 每温度 | `per_temperature_bag_accuracy` | 每温度下 MIL 判断准确率 |

### 在线评估（PPO 策略真实生成对比）

```bash
python -m ppo.eval --config configs/base.yaml
```

#### 工作原理

```
对每条 prompt，num_votes 路同时生成:
  Round 0: 提交 [prompt]→选温 T₀→生成 segment₀×num_votes
  Round 1: 提交 [prompt+seg₀]→选温 T₁→生成 segment₁×num_votes  (APC✓)
  ...直到 EOS 或 max_tokens
  
  第一票的特征驱动温度决策，所有票共享同一温度序列
  最终 majority voting 决定正确性
```

#### 三种策略（同 prompt，同随机种子）

| 策略 | 温度来源 |
|---|---|
| PPO dynamic | PPO 策略每 segment 动态选择 |
| Best fixed | 始终用数据集最优温度（从 eval JSONL 统计） |
| Random | 每 segment 均匀随机选温度 |

#### 输出

| 指标 | 说明 |
|---|---|
| `accuracy` | Majority Voting 正确率 |
| `mean_temperature` | 平均选中温度 |
| `std_temperature` | 温度标准差 |
| `mean_segments` | 平均生成 segment 数 |
| `improvement_over_random` | PPO vs random 的 accuracy 差（正数 = PPO 更好） |
| `improvement_over_best_fixed` | PPO vs 最佳固定温度的 accuracy 差（终极指标） |

#### 输出示例

```
ONLINE EVALUATION RESULTS  (vLLM + APC, per-segment temperature)
======================================================================
Prompts evaluated: 500
Majority voting:   4 votes per prompt
Segment size: 32 tokens
Best fixed temp (from dataset): T=0.3

  PPO dynamic temperature:
    accuracy=0.6600  correct=330/500
    mean_temp=0.52 ± 0.31    avg_segments=16.3

  Best fixed temperature (T=0.3):
    accuracy=0.6420  correct=321/500
    mean_temp=0.30 ± 0.00    avg_segments=17.1

  Random temperature:
    accuracy=0.6280  correct=314/500
    mean_temp=0.81 ± 0.42    avg_segments=17.5

  Improvement over random:     +0.0320
  Improvement over best fixed:  +0.0180
======================================================================
```

---

## 一键运行

### 完整 Pipeline

```bash
# 所有路径从 config 读取，只需指定 GPU
GPU_DEVICES=0,1 bash scripts/run_pipeline.sh

# 在线评估默认开启，跳过:
STAGES=build,split,mil,ppo GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
```

### 分阶段执行

```bash
# 逐 stage 运行
bash scripts/stage1_build.sh
bash scripts/stage2_mil.sh
bash scripts/stage3_ppo.sh
bash scripts/stage4_eval.sh

# 或直接用 STAGES 变量
STAGES=build,split GPU_DEVICES=0,1 bash scripts/run_pipeline.sh    # 只跑 Stage 1
STAGES=mil GPU_DEVICES=0,1 bash scripts/run_pipeline.sh            # 只跑 MIL
STAGES=ppo GPU_DEVICES=0 bash scripts/run_pipeline.sh              # 只跑 PPO
```

### Pipeline 执行顺序

```
1. 自检 (ENABLE_STARTUP_SELF_CHECK=1)
2. build_dataset     Stage 1: 多温多数投票 → JSONL 数据集
3. split_jsonl       Stage 1b: group-aware train/eval 划分
4. train_mil         Stage 2: MIL 错误定位模型训练
5. evaluate          Stage 4a: MIL 评估（PPO 前检查 MIL 质量）
6. train_ppo         Stage 3: 在线 PPO 训练（vLLM + majority voting）
7. online_evaluate   Stage 4b: 在线 PPO 评估
```

### Stage 名称

| stage 名 | 对应阶段 |
|---|---|
| `build` | Stage 1 build_dataset |
| `split` | Stage 1b split_jsonl |
| `mil` | Stage 2 train_mil |
| `eval` | Stage 4a 离线评估 (dataset + MIL) |
| `ppo` | Stage 3 在线 PPO 训练 |
| `eval_ol` | Stage 4b 在线 PPO 评估 |

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPU_DEVICES` | 空 | 逗号分隔 GPU 编号 |
| `BACKEND` | `vllm` | Stage 1 后端 (`vllm` 或 `api`) |
| `STAGES` | `build,split,mil,eval,ppo,eval_ol` | 要执行的阶段 |
| `RUN_NAME` | `exp_时间戳` | 日志文件名 |
| `LOG_DIR` | `logs` | 日志目录 |
| `VAL_RATIO` | `0.1` | val 集比例（训练/val 池中的占比） |
| `TEST_RATIO` | `0.1` | test 集比例（全量数据中的占比） |
| `ENABLE_STARTUP_SELF_CHECK` | `1` | GPU 自检 |

路径从 `configs/base.yaml` 的 `paths` 段读取，也可通过环境变量覆盖（如 `RAW_INPUT`、`ALL_DATASET` 等）。

---

## 配置参考

完整配置位于 `configs/base.yaml`：

```yaml
seed: 42

paths:
  raw_input: data/math.jsonl
  all_dataset: datasets/all.jsonl
  train_dataset: datasets/train.jsonl
  val_dataset: datasets/val.jsonl
  test_dataset: datasets/test.jsonl
  mil_ckpt: checkpoints/mil_ckpt.pt
  ppo_ckpt: checkpoints/ppo_ckpt.pt

data:
  max_length: 2048
  instance_dim: 64
  temp_bins: [0.1, 0.2, ..., 1.5]
  segment_mode: step
  segment_size: 256
  segment_pooling: mean

inference:
  model_name_or_path: /home/xuezhe/models/Qwen3-8B
  tensor_parallel_size: auto
  max_new_tokens: 8192
  temperature_grid: [0.1, 0.2, ..., 1.5]   # 15 bins
  num_votes: 8
  top_k_logits: 16
  feature_mode: combined
  api:
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    max_concurrent: 16
    max_retries: 3

mil:
  model:
    hidden_dim: 256
    aggregator: attention
    use_position: true       # 位置编码
    use_gru: true            # 双向 GRU 错误传播建模
  training:
    instance_loss: pure      # pure | topk | soft_pseudo_label | contrastive
    batch_size: 32
    lr: 2.0e-4
    max_epochs: 50           # 训练轮数上限
    early_stop_patience: 5   # val inst_logit_separation 不提升 N 轮即停
    alpha_temp: 0.1          # 温度分类权重 (已压低)
    beta_inst_aux: 0.2       # instance 辅助 loss 权重
    gamma_smooth: 0.05       # 平滑正则权重

ppo:
  model:
    hidden_dim: 256
  training:
    max_iterations: 200      # 在线 rollout 总轮数上限
    early_stop_patience: 10  # val_value 不降低 N 轮即停
    online_rollout_size: 128
    ppo_epochs: 8
    mini_batch_size: 32
    val_ratio: 0.2
    gamma: 0.99
    gae_lambda: 0.95
    clip_eps: 0.2
    value_coef: 0.5
    entropy_coef: 0.005
    lr: 1.0e-5
    shaping_coef: 0.15
```

---

## 模型汇总

| 模型 | 参数量 | 位置 | 用途 |
|---|---|---|---|---|
| Qwen3-8B (vLLM) | ~8B | 外部加载 | 数学回答生成 |
| MILModel (含 pos + GRU) | ~500K | `mil/model.py` | 错误定位：预测回答是否错 + 每个 segment 的错误分数 |
| GlobalTempHead | ~4K | `mil/model.py` | 从 bag 表示预测温度 bin |
| DynamicTempHead | ~200K | `mil/model.py` | 逐 segment 温度预测 + 平滑正则 |
| PolicyValueNet (3-layer 256) | ~150K | `ppo/model.py` | PPO 策略 + 价值函数 |

---

## 关键架构决策

### 1. 标签翻转：错误定位视角

`label=0` 表示正确（负 bag），`label=1` 表示错误（正 bag）。这使 MIL attention 自然聚焦于"最像错误的 segment"，直接服务于错误定位目标。整个 pipeline（MIL → PPO shaping）在此语义下统一。

### 2. Top-k MIL instance loss

错误回答中并非所有 segment 都有错。对正 bag 只惩罚 top-k 最高分 instance，其余维持 target=0。这比平铺 label 更符合 MIL 的"至少一个正 instance"假设，减少了 label 噪声对 backbone 特征的污染。

### 3. 位置编码 + BiGRU

推理链中 segment 具有时序依赖和错误传播特性。位置编码让模型知道 segment 在链中的位置，BiGRU 捕捉"错误出现后 hidden state 的系统性偏移"，提升 error localization 精度。

### 4. 在线 PPO 而非离线

离线数据中 action 不影响 reward（预采集的 segment 特征已固定），PPO 无法学习因果策略。在线模式下策略真正控制 vLLM 生成温度，action→reward 因果链完整。

### 5. Majority Voting 全链路统一

Stage 1 用多数投票决定 label → Stage 2 MIL 学习 majority-vote 正确性 → Stage 3 PPO terminal reward 基于 majority vote → Stage 4 在线评估测量 majority-vote accuracy。全链路优化同一目标。

### 6. vLLM APC 跨温度共享

Stage 1 将 N×15 条请求在一次 `llm.generate()` 中提交，同一 prompt 的 15 个温度变体通过 APC 共享 prompt KV-cache，prefill 计算量从 15× 降到 1×。在线 PPO/评估中每 segment 轮次也通过 APC 共享前缀。
