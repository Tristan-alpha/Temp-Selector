# MIL Module Design

## Causal Prefix Value Model

The Full proposal replaces BiGRU instance pseudo-labeling with a mask-aware
unidirectional GRU estimating `P(final answer correct | generated prefix)`.
Training combines continuation binomial likelihood, terminal response BCE, and
same-problem paired ranking. Packed batch inference and streaming `step` share
the same parameters and positional encoding.

## Problem formulation

Given a math reasoning chain split into segments, MIL answers:

- **Bag-level**: Is this whole answer wrong? (`bag_logit`)
- **Attention**: Which segments contribute most to the prediction? (`attn_w`)

MIL reads `individual_label` (per-response correctness):

| individual_label | meaning | MIL term |
|---|---|---|
| 0 | answer correct | negative bag (no errors) |
| 1 | answer wrong, ≥1 segment wrong | positive bag (contains errors) |

Every `> 0.5` branch carries an inline comment (`# label=1: positive bag (contains errors)`) to make the convention explicit.

## Design philosophy: bag_bce only (Ilse et al. 2018)

Following Ilse et al. (2018) "Attention-based Deep Multiple Instance Learning", MIL is trained with **bag-level BCE only** — no instance loss, no auxiliary heads, no smoothness regularization. Attention weights are learned purely through backpropagation from the bag classification objective. The paper shows that this is sufficient: the model naturally learns to attend to discriminative instances.

## Segment feature pre-computation

Features are extracted once before training and cached in system RAM:

```
Pre-computation (once before epoch loop):
  BagDataset loads JSONL → pre-tokenize prompts
  make_collate_fn (with extractor):
    extractor.extract_from_ids(...)  ← vLLM prefill, expensive
    build_segment_obs_from_lp → segment_pooling → [K, instance_dim]
  → segment_cache[i] = {instances, label, temp_idx}  (system RAM, ~3-13 GB)

Training (every epoch):
  SegmentCacheDataset → make_cached_collate_fn:
    cache lookup → pad → stack  ← no vLLM calls
```

Two feature modes, both producing exactly 4098-dim per-token vectors via `build_segment_obs_from_lp`:

- `topk_logprobs`: `[sampled_logprob, entropy, topk_logprob_0..topk_logprob_4095]` (2+4096=4098, no zero-padding)
- `hidden_states`: `[sampled_logprob, entropy, hidden_dim_0..hidden_dim_4095]` (2+4096=4098, no zero-padding)

Both modes use logprobs (always extracted). `build_segment_obs_from_lp` accepts `include_topk` to
control whether the top-k array or hidden states fill the 4096 dims after logprob+entropy.

## Model architecture

```
instances [B, K, 4098]
     ↓
InstanceEncoder           Linear(4098→1024)→ReLU→Linear(1024→1024)→ReLU
     ↓  [B, K, 1024]
SinusoidalPositionalEncoding   (optional)
     ↓  [B, K, 1024]
BiGRU                     bidirectional GRU(1024→1024) + Linear(2048→1024)
     ↓  [B, K, 1024]
AttentionAggregator       Linear(1024→1)→softmax → weighted sum
     │
     ├── bag_repr [B, 1024]  →  bag_head  →  bag_logit [B]
     └── attn_w   [B, K]       (attention weights, interpretable)
```

~450K params with pos + GRU (removed inst_head).

## Loss function

```
Total = bag_bce
```

`BCEWithLogitsLoss` with `pos_weight = sqrt(n_correct / n_wrong)`. No auxiliary losses.

## Evaluation

Bag-level: AUC, accuracy, precision, recall, F1, calibration (ECE, Brier).
Attention interpretability: entropy, top3_mass, effective_n.

## PPO credit assignment

During PPO training, MIL attention weights are used for credit assignment:

```
batch construction (post-rollout):
  full_bag = torch.stack(all_round_segments)  # [K, obs_dim] — full response bag
  mil_model(full_bag) → attn_w  # [K] — attention weights
  reward[t] = terminal_reward × attn_w[t] / attn_w.sum()  # L1-normalized attention weights
```

The attention mechanism distributes the terminal reward (±1) across steps proportional to their importance. This replaces the previous `inst_logit`-based shaping which had no reliable ground truth for evaluation.
