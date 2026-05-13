# MIL Module Design

## Problem formulation

Given a math reasoning chain split into segments, MIL answers two questions:

1. **Bag-level**: Is this whole answer wrong? (`bag_logit`)
2. **Instance-level**: Which specific segments are likely wrong? (`inst_logit`)

The labels are flipped from standard MIL convention:

| label | meaning | MIL term |
|---|---|---|
| 0 | answer correct, all segments correct | negative bag |
| 1 | answer wrong, at least one segment wrong | positive bag |

**Why flip?** Standard MIL (positive = anomaly) would make attention focus on "normal" segments. Flipping makes attention naturally focus on error-like segments, directly serving the error-localization goal.

Labels come from majority voting over `num_votes` completions per (prompt, temperature). A 4:0 or 3:1 correct:wrong split → label=0 (correct bag); 2:2 or worse → label=1 (error bag).

## Model architecture

```
instances [B, K, 64]
     │
     ▼
InstanceEncoder           Linear(64→256)→ReLU→Linear(256→256)→ReLU
     │  [B, K, 256]
     ▼
SinusoidalPositionalEncoding   (optional) learnable PE buffers
     │  [B, K, 256]
     ▼
BiGRU                     bidirectional GRU(256→256) + Linear(512→256)
     │  [B, K, 256]   captures error propagation patterns across segments
     ▼
AttentionAggregator       Linear(256→1)→softmax → weighted sum
     │
     ├── bag_repr [B, 256]  →  bag_head  → bag_logit [B]   (whole-answer error prob)
     └── inst_logit [B, K]  ←  inst_head applied per-segment  (per-segment error score)
```

**Design rationale for key components:**

- **Position encoding**: Reasoning chains have sequential structure. Without PE, the model cannot distinguish "error at start" from "error at end", which matters for PPO temperature selection (early steps benefit more from low temperature).

- **BiGRU**: Errors propagate — a mistake at step 3 often causes downstream steps to also produce wrong results. Bidirectional GRU lets the model see both "what led to this step" and "what followed from it".

- **Attention aggregator**: Instead of mean/max pooling, learned attention weights let the model decide how much each segment contributes to the bag-level prediction. These weights are directly interpretable as "where the model thinks the error is".

## Loss function design

```
Total = bag_bce + β × instance_bce + α × temp_ce + γ × smoothness
         ①          ②              ③            ④
```

### 1. Bag BCE loss

`BCEWithLogitsLoss(bag_logit, label)` with `pos_weight = sqrt(n_correct / n_wrong)`.

The sqrt (rather than direct ratio) is a compromise: correct answers are typically ~67% of data (majority), so the raw ratio ~2.0 would over-weight the minority class. Sqrt gives ~1.4, providing moderate rebalancing without dominating the loss.

### 2. Instance auxiliary loss — β=0.2

This is the key MIL-specific term. **Without it, the model has no signal about WHICH segments are wrong** — only that the bag is wrong. The method is configurable via `mil.training.instance_loss`:

| Method | Pos bag strategy | Reference |
|---|---|---|
| `pure` (default) | k=1: only the single highest-scoring instance → target=1 | [FocusMIL 2024](https://arxiv.org/abs/2408.09449) |
| `topk` | k=n_valid//3: top third → target=1 | Legacy |
| `soft_pseudo_label` | target = sigmoid(inst_logit).detach() with anti-degeneration clamp | [SeLa-MIL 2024](https://arxiv.org/abs/2408.04813) |
| `contrastive` | logsumexp(scores) - max(scores) | [NDI-MIL 2025](https://ieeexplore.ieee.org) |

All methods share the same negative bag treatment: all instances target=0.
The default `pure` is the simplest theoretically-grounded approach per the MIL assumption: at least one positive instance makes a positive bag.

### 3. Temperature auxiliary loss — α=0.1 (typically 0.0)

Predicts which of the 15 temperature bins was used from (a) bag representation (GlobalTempHead) and (b) per-instance features (DynamicTempHead). Weight is kept low because temperature bin is a weak signal — many temperatures can produce correct answers.

### 4. Smoothness regularization — γ=0.05

`mean((logit_{t+1} - logit_t)²)` penalizes adjacent segments having wildly different temperature predictions. Encourages smooth, gradual temperature changes across a reasoning chain.

## DynamicTempHead and the Bug 1 lesson

The DynamicTempHead receives per-segment features and predicts per-segment temperatures. During training, it must receive `out["encoder_out"]` (post-encoder, post-position, post-GRU), NOT `mil.encoder(x)` (raw encoder output).

**Historical Bug 1**: The training code originally used `mil_encoder(x)`, which passed raw encoder features to the dynamic head. But evaluation code used `out["encoder_out"]` — features with position encoding and GRU processing. The representations were fundamentally different, making the dynamic head's evaluation meaningless.

**Lesson**: When a feature path exists in two places (training + eval), verify they receive the same tensor. A simple assertion `assert torch.allclose(train_features, eval_features)` in a test (`test_dynamic_head_feature_consistency`) now guards against regression.
