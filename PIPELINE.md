# tf-mil 完整 Pipeline 文档

## 项目概述

`tf-mil`（Temperature Framework -- Multiple Instance Learning）研究**LLM 推理时动态温度选择**。核心假设：

1. 数学推理链由若干步骤（segment）组成，不同步骤可能需要不同的生成温度
2. 错误推理链中**至少存在一个错误 segment**（正 bag），正确推理链中所有 segment 都正确（负 bag）
3. 用 MIL 定位错误 segment，再用 PPO 学习"根据 segment 特征选择温度以避免错误"

### 核心流程

```
Stage 1: 多温度数据采集 → 多数投票 → JSONL 数据集
Stage 2: MIL 在线特征提取 + 训练 → 错误定位模型
Stage 3: PPO 在线逐段生成 + 强化学习 → 动态温度策略
Stage 4: 在线评估 (PPO vs best-fixed vs random)
```

### 目录结构

```
tf-mil/
├── configs/
│   ├── dataset/                # 数据集生成配置
│   └── training/               # MIL + PPO 训练配置
├── inference/
│   └── vllm_runner.py          # VLLMFeatureExporter: 生成 + 在线特征提取
├── features/
│   ├── schema.py               # Segment 数据结构
│   ├── segmenter.py            # 分段策略 + segment_pooling + build_segment_obs_from_lp
│   ├── vectorizer.py           # token_to_vec, token_to_obs, compute_entropy
│   └── dataset_eval.py         # 数据集统计分析
├── mil/
│   ├── model.py                # MILModel + temp heads
│   ├── utils.py                # BagDataset, TokenBatchSampler, make_collate_fn
│   ├── training.py             # MIL 训练
│   └── eval.py                 # MIL 评估
├── ppo/
│   ├── model.py                # PolicyValueNet, GAE, warm-start
│   ├── training.py             # PPO 训练
│   └── eval.py                 # 在线评估
├── utils/
│   ├── answer_verifier.py      # Math-Verify 包装器
│   ├── jsonl.py                # JSONL 工具
│   └── exp_logger.py           # 日志
├── scripts/
│   ├── run_pipeline.sh         # 全流程编排
│   └── build_dataset.py        # Stage 1 数据构建
└── tests/                      # CPU-only 测试
```

---

## 核心概念：标签语义

每个样本有两个标签字段：

| 字段 | 含义 | 使用者 |
|---|---|---|
| `individual_label` | 0=该条回答正确, 1=该条回答错误 (per-response) | MIL 训练/评估 |
| `voting_label` | 0=多数投票正确, 1=多数投票错误 (majority vote) | PPO 温度偏置初始化 |

### Majority Voting

每个 (prompt, temperature) 组合生成 `num_votes` 路回答。通过 self-consistency 多数投票决定 voting_label：

1. 从每票回答中提取最后一个 `\boxed{...}` 内容
2. 取出现次数最多的众数答案
3. 众数答案等同于标准答案 → voting_label=0，否则 voting_label=1
4. individual_label 由每条回答自身是否正确决定，与多数投票结果无关

---

## Stage 1：数据集构建

### 入口

```bash
python scripts/build_dataset.py --config configs/dataset/full.yaml
```

### 流程

```
输入 JSONL (N 条 prompt)
    ↓
raw vLLM (无 speculative decode): N × 15 temps 一次性提交
  APC 自动共享同 prompt KV-cache
    ↓
每条 (prompt, temp, vote):
  ├── extract_final_answer: 提取 \boxed{...} 答案
  ├── verify_answer: vs gold → individual_correct
  ├── tokenize: 记录 token_ids + tokens
  └── minority/majority voting → label
```

### 输出

```
datasets/train.jsonl, val.jsonl, test.jsonl
```

JSONL 行格式：

```json
{
  "sample_id": "q1_t0.5_v0",
  "prompt": "original question",
  "response": "generated answer text",
  "individual_label": 0,
  "voting_label": 0,
  "temperature": 0.5,
  "token_ids": [123, 456, ...],
  "tokens": ["Hello", " world", ...],
  "metadata": {
    "gold_answer": "42",
    "rendered_prompt": "...",
    "individual_correct": true,
    "extracted_answer": "42",
    "vote_id": 0,
    "num_votes": 8,
    "votes_correct": 6,
    "votes_total": 8
  }
}
```

### 配置

```yaml
# configs/dataset/full.yaml
seed: 42
paths:
  raw_input: data/math.jsonl
  train_dataset: datasets/train_full.jsonl
  ...
inference:
  model_name_or_path: /home/xuezhe/models/Qwen3-8B
  max_new_tokens: 8192
  temperature_grid: [0.1, ..., 1.5]
  num_votes: 8
split:
  val_ratio: 0.1
  test_ratio: 0.1
```

---

## Stage 2：MIL 训练 — 在线特征提取 + 错误定位

### 入口

```bash
python -m mil.training --config configs/training/base.yaml
```

### 在线特征提取

MIL 训练**不在 JSONL 中存储特征**。`make_collate_fn` 在每个 batch 中：

```
for each batch:
    full_ids = pids + token_ids      (预分词)
    extractor.extract_from_ids(full_ids, prompt_lens, temperatures, ...)
      → llm.generate(full_ids, max_tokens=1)  ← speculative decode 产出 hidden states
      → apply_model(_LogprobsComputeFn) per chunk ← 计算 top-k logprobs
    segment_pooling(tok_vecs, spans) → [K, instance_dim] instance 矩阵
```

两种 feature_mode：

| mode | 在线提取 | instance_dim |
|------|---------|-------------|
| `topk_logprobs` | logprob + entropy + top-4096 logprobs | 4098 |
| `hidden_states` | hidden states (Qwen3-8B last layer) | 4096 |

### 模型架构

```
instances [B, K, 4098]
    ↓
InstanceEncoder: Linear(4098→1024)→ReLU→Linear(1024→1024)→ReLU
    ↓ [B, K, 1024]
SinusoidalPositionalEncoding (可选)
    ↓
BiGRU (可选, bidirectional)
    ↓
AttentionAggregator
    ├── bag_repr → bag_head → bag_logit
    └── inst_logit (per-segment error score)
```

### 损失函数

```
Total = bag_bce + β×instance_bce + α×temp_ce + γ×smoothness
```

Instance loss 四种方法：`pure` (k=1, 默认), `topk`, `soft_pseudo_label`, `contrastive`

### 训练超参数 (base.yaml)

```yaml
mil:
  model:
    hidden_dim: 1024
    aggregator: attention
    use_position: true
    use_gru: true
  training:
    instance_loss: pure
    max_tokens_per_batch: 131072
    lr: 2.0e-4
    max_epochs: 50
    early_stop_patience: 5
    alpha_temp: 0.0
    beta_inst_aux: 0.2
    gamma_smooth: 0.05
```

---

## Stage 3：在线 PPO 训练

### 入口

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m ppo.training --config configs/training/base.yaml
```

### 在线生成流程

```
每 iteration (200 轮 max):
  1. 采样 128 prompts
  2. 逐 segment 生成 (segment_size=512 tokens, fixed_window):
     Round 0: prompt → T=0.7 → generate 512 tokens
     Round 1: prompt+seg₀ → policy(obs₁)→temperature → generate 512 tokens
     ...直到 EOS 或 max_tokens
  3. 终局: majority voting → terminal reward (±1)
  4. 中间步: MIL shaping reward = shaping_coef × (1-sigmoid(inst_logit))
  5. GAE + PPO clip update
```

### 模型

```
PolicyValueNet (~150K params)
  backbone: 3-layer MLP (4098→1024→1024→1024)
  pi head:  Linear(1024→15)   # 15 temperature bins
  v head:   Linear(1024→1)    # value
```

MIL warm-start: backbone 前 2 层从 MIL encoder 拷贝

### 训练超参数

```yaml
ppo:
  model:
    hidden_dim: 1024
  training:
    max_iterations: 200
    online_rollout_size: 128
    ppo_epochs: 8
    mini_batch_size: 32
    lr: 1.0e-5
    clip_eps: 0.2
    shaping_coef: 0.15
```

---

## 一键运行

```bash
GPU_DEVICES=0,1 bash scripts/run_pipeline.sh

# 指定 stage
STAGES=build,split,mil GPU_DEVICES=0,1 bash scripts/run_pipeline.sh
STAGES=mil GPU_DEVICES=0 bash scripts/run_pipeline.sh
```

### Pipeline 执行顺序

```
1. build_dataset    Stage 1: 多温生成 + 多数投票 → JSONL
2. train_mil        Stage 2: MIL 在线特征提取 + 训练
3. evaluate         MIL 评估
4. train_ppo        Stage 3: PPO 在线训练
5. online_evaluate  PPO vs best-fixed vs random
```

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GPU_DEVICES` | 空 | GPU 编号 |
| `CONFIG` | `configs/training/base.yaml` | 训练配置 |
| `DATASET_CONFIG` | `configs/dataset/full.yaml` | 数据集配置 |
| `STAGES` | `build,split,mil,eval,ppo,eval_ol` | 执行阶段 |
| `RUN_NAME` | `exp_时间戳` | 日志文件名 |

---

## 配置参考 (base.yaml)

```yaml
seed: 42

paths:
  train_dataset: datasets/train.jsonl
  val_dataset: datasets/val.jsonl
  test_dataset: datasets/test.jsonl
  mil_ckpt: checkpoints/mil_ckpt.pt
  ppo_ckpt: checkpoints/ppo_ckpt.pt

data:
  instance_dim: 4098
  temp_bins: [0.1, ..., 1.5]
  segment_mode: fixed_window
  segment_size: 512
  segment_pooling: mean

inference:
  model_name_or_path: /home/xuezhe/models/Qwen3-8B
  gpu_memory_utilization: 0.90
  max_new_tokens: 8192
  num_votes: 8
  top_k_logprobs: 4096
  feature_mode: topk_logprobs

mil:
  model:
    hidden_dim: 1024
    aggregator: attention
    use_position: true
    use_gru: true
  training:
    instance_loss: pure
    max_tokens_per_batch: 131072
    lr: 2.0e-4
    max_epochs: 50
    early_stop_patience: 5

ppo:
  model:
    hidden_dim: 1024
  training:
    max_iterations: 200
    early_stop_patience: 10
    online_rollout_size: 128
    ppo_epochs: 8
    mini_batch_size: 32
    lr: 1.0e-5
    clip_eps: 0.2
    shaping_coef: 0.15
```

---

## GPU 分配

```
N >= 2 GPUs:
  VLLMFeatureExporter(reserve_training_gpu=True)
    → vLLM 占用 GPU 0..N-2, 训练独占 GPU N-1

1 GPU:
  全部共享 single GPU
```
