# MIL Module Design

## Problem formulation

Given a math reasoning chain split into segments, MIL answers two questions:

1. **Bag-level**: Is this whole answer wrong? (`bag_logit`)
2. **Instance-level**: Which specific segments are likely wrong? (`inst_logit`)

MIL reads `individual_label` (per-response correctness):

| individual_label | meaning | MIL term |
|---|---|---|
| 0 | answer correct | negative bag (no errors) |
| 1 | answer wrong, ≥1 segment wrong | positive bag (contains errors) |

Every `> 0.5` branch carries an inline comment (`# label=1: positive bag (contains errors)`) to make the convention explicit.

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
SinusoidalPositionalEncoding   (optional) learnable PE buffers
     ↓  [B, K, 1024]
BiGRU                     bidirectional GRU(1024→1024) + Linear(2048→1024)
     ↓  [B, K, 1024]
AttentionAggregator       Linear(1024→1)→softmax → weighted sum
     │
     ├── bag_repr [B, 1024]  →  bag_head  → bag_logit [B]
     └── inst_logit [B, K]   ←  inst_head per-segment  (error score)
```

~500K params with pos + GRU.

## Loss function

```
Total = bag_bce + β×instance_bce + α×temp_ce + γ×smoothness
```

### Bag BCE

`BCEWithLogitsLoss` with `pos_weight = sqrt(n_correct / n_wrong)`.

### Instance auxiliary loss (β=0.2)

| Method | Pos bag strategy |
|---|---|
| `pure` (default) | k=1: highest-scoring instance → target=1 |
| `topk` | k=n_valid//3: top third → target=1 |
| `soft_pseudo_label` | sigmoid(inst_logit).detach() with anti-degeneration clamp |
| `contrastive` | logsumexp(scores) - max(scores) |

All methods: negative bag → all instances target=0.

### Temperature auxiliary loss (α, typically 0.0)

GlobalTempHead + DynamicTempHead predict temperature bin.

### Smoothness regularization (γ=0.05)

`mean((logit_{t+1} - logit_t)²)` — penalize jagged temperature predictions.

## DynamicTempHead and the Bug 1 lesson

Must receive `out["encoder_out"]` (post-encoder, post-position, post-GRU), NOT `mil.encoder(x)` (raw). These representations are fundamentally different — using raw encoder output would make the dynamic head's evaluation meaningless.
