# Features Module Design

## Data pipeline overview

```
JSONL prompts           Stage 1 entry: N math problems with gold answers
     │
     ▼
vLLM multi-temp gen     N × 15 temps × num_votes completions in one batch (APC shares KV)
     │
     ▼
Per-completion:         1. Math-Verify checks answer correctness
                        2. Extract per-token features (logprob, entropy, top-16 logprobs)
                        3. Segment tokens (step / fixed_window / punctuation)
     │
     ▼
Majority voting         Per (prompt, temp): majority_correct → label 0 (correct) or 1 (error)
     │
     ▼
BagSample               One per (prompt, temp, vote): token_features + segment_spans + label
     │
     ▼
datasets/all.jsonl      Serialized BagSamples (N × 15 × num_votes lines)
     │
     ▼
Stage 2 consumption:    segment_pooling(token_vecs, spans) → [K, 64] instance matrix
```

## Segmentation modes

Three strategies exist; `step` is the default and recommended mode.

### step (default)

Segments by double-newline (`\n\n`). The system prompt instructs the model to separate each reasoning step with exactly two newlines, making this a natural choice for mathematical derivations where each step is a logical unit.

```
Step 1: Define variables.
          ↓ \n\n
Step 2: Set up equation.
          ↓ \n\n
Step 3: Solve.
```

Falls back to `fixed_window` if no `\n\n` delimiters are found.

**Why not always step?** If the model ignores the formatting instruction, `step` falls back to `fixed_window`. The step strategy is preferred because semantic step boundaries align better with reasoning structure than arbitrary 32-token windows.

### fixed_window

Fixed `segment_size` tokens per segment. Simple, deterministic, independent of content. Used in `baseline_fixed_window.yaml` ablation to compare against step segmentation. Required for `concat` pooling mode.

### punctuation

Segments at sentence-ending punctuation (`.`, `!`, `?`, `;`, `\n`). More granular than step but less semantically meaningful. Not used in any active config.

## Segment pooling

Each segment contains an arbitrary number of token vectors (dim=64). Pooling collapses them to a single segment feature vector:

### mean pooling (default)

```
segment_feat[j] = mean(token_vec[j][start:end])  → [64]
```

Information-preserving but lossy: the distribution of token features within the segment is lost. Works with all segmentation modes.

### concat pooling

```
segment_feat = concat(token_vec[j][:segment_size]) → [segment_size × 64]
```

No information loss — each token position retains its own features. Downside: fixed input dimension (`segment_size × 64` = up to 2048 for segment_size=32). Only works with `fixed_window` segmentation. Used in `pool_concat.yaml` ablation.

## Vectorizer: feature construction strategy

`features/vectorizer.py` converts raw token features into fixed-dim observation vectors:

```
token_to_vec(token_feat) → [logprob, entropy, top16_logprob₁, ..., top16_logprob₁₆, padding]
                                  └────────── 2 ──────────┘  └──────── 16 ──────────┘  └─ to 64 ─┘
```

**Merge order**: logprob → entropy → top-k logprobs → hidden states. This order groups semantically related features. logprob and entropy capture individual token certainty; top-k logprobs capture the distribution shape.

**Defaults**: Missing logprob → -20.0 (≈log(2e-9), near-zero probability); missing entropy → 0.0. These defaults represent "maximum uncertainty" — a token we know nothing about.

**Padding**: Features shorter than `obs_dim` (64) are zero-padded. Features longer are truncated. This handles variable numbers of top-k logprobs and optional hidden states.

## Majority voting (self-consistency)

Each (prompt, temperature) combination produces `num_votes` completions. Self-consistency determines the bag label:

1. `extract_answer()` parses each response with `math_verify.parse()` to extract the mathematical answer
2. `Counter.most_common(1)` finds the modal (plurality) answer across all votes
3. The modal answer is compared to gold via `verify_answer_by_value()` — if equivalent, the bag is correct (label=0), otherwise error (label=1)

This label is shared by all `num_votes` BagSamples for that (prompt, temp). Individual per-vote correctness is preserved in `metadata.individual_correct` as an auxiliary reference statistic (computed via per-vote `verify_answer()`, independent of the self-consistency label).

**Why self-consistency?** Single-answer verification is noisy — LLMs can be correct via wrong reasoning or wrong via arithmetic error despite correct steps. Self-consistency provides a robust correctness signal by aggregating multiple independent samples, and the same criterion is used end-to-end: Stage 1 labeling, PPO terminal reward, and Stage 4 online evaluation all use the same self-consistency logic.

## Inter-stage data contract

```
Stage 1 → datasets/{train,eval}.jsonl
         Each line: BagSample with token_features, segment_spans, label, temperature

Stage 2 consumes:  BagDataset loads JSONL → segment_pooling(token_to_vec(tf)) → [K, 64]
         Produces:   mil_ckpt.pt (MIL weights + config)

Stage 3 consumes:  raw prompts JSONL (same as Stage 1 input)
                   mil_ckpt.pt (for backbone warm-start + shaping reward)
         Produces:   ppo_ckpt.pt (policy weights + config)

Stage 4 consumes:  eval.jsonl (for dataset stats)
                   mil_ckpt.pt (for MIL evaluation)
                   raw prompts + ppo_ckpt.pt (for online evaluation)
```

The key contract is the `BagSample` format. Stage 1 writes it; Stage 2 reads it. The `instance_dim` (64) is the shared dimensionality across all stages.
