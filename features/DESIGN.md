# Features Module Design

## Mask-aware concat path

The Full proposal retains incomplete final segments. Each segment has 64 token
slots and 64 probability features per token, producing `[K, 4096]` features and
a `[K, 64]` token mask. Padding is zeroed explicitly. The legacy concat path is
unchanged for reproducibility.

## Data pipeline overview (post-simplification)

```
JSONL prompts           Stage 1 entry: N math problems with gold answers
     ↓
raw vLLM multi-temp      N × 15 temps × num_votes completions (APC shares KV)
     ↓
Per-completion:          1. extract_final_answer → \boxed{...} extraction
                         2. verify_answer → individual_correct
                         3. tokenize → token_ids + tokens
     ↓
Majority voting          Per (prompt, temp): self_consistency → label
     ↓
JSONL rows               {sample_id, prompt, response, label, temperature,
                          token_ids, tokens, metadata}
     ↓
Stage 2 consumption:     make_collate_fn → online extraction via
                         extract_from_ids → build_segment_obs_from_lp →
                         segment_pooling → [K, instance_dim] instance matrix
```

**Key design**: Features are NOT stored in JSONL. Logprob + entropy + top-k logprobs are extracted online during MIL/PPO training via `extract_from_ids` and `generate_with_features`.

## Segmentation modes

### fixed_window (default)

Fixed `segment_size` tokens per segment. Simple, deterministic, independent of content.

### step

Segments by double-newline (`\n\n`). Falls back to `fixed_window` if no `\n\n` delimiters found. Used in `step_control.yaml` for comparison.

### punctuation

Segments at sentence-ending punctuation. Not used in any active config.

## Segment pooling

### mean pooling (default)

```
segment_feat[j] = mean(token_vec[j][start:end])  → [instance_dim]
```

### concat pooling

```
segment_feat = concat(token_vec[j][:segment_size]) → [segment_size × instance_dim]
```

No information loss. Only works with `fixed_window`. Used in `pool_concat.yaml`.

## build_segment_obs_from_lp

Shared helper in `segmenter.py` used by both MIL collate_fn and PPO. Converts `generate_with_features` output logprob tensor into segment observations:

```
lp_tensor [n_tok, top_k+1]
  → col 0: sampled logprob
  → cols 1:: top-k logprobs + exp → entropy
  → cat with optional hidden states
  → pad/trunc to instance_dim
  → build_segments + segment_pooling
```

Vectorized: all per-token entropy computed in one `exp(lp) * lp` sum — no Python for-loop.

## Online feature extraction (MIL/PPO)

```
extract_from_ids(full_ids, prompt_lens, temperatures, return_logprobs, return_hidden):
  llm.generate(full_ids, max_tokens=1)      ← speculative decode → hidden states
  for each sample:
    read safetensors (/dev/shm)
    slice response hidden states
    if return_logprobs: apply_model(_LogprobsComputeFn) per chunk
    if return_hidden: return raw hidden states
```

`_LogprobsComputeFn`: single-chunk, picklable callable for `llm.apply_model()`. Computes `model.norm(h) → compute_logits → compute_topk_logprobs → logprobs.cpu()`.

## Majority voting (self-consistency)

1. `extract_final_answer()`: extract last `\boxed{...}` (no fallback)
2. `Counter.most_common(1)`: modal answer across votes
3. `verify_answer_by_value()`: compare modal vs gold

Same criterion used end-to-end: Stage 1 labeling, PPO terminal reward, Stage 4 online evaluation.

## Inter-stage data contract

```
Stage 1 → datasets/{train,val,test}.jsonl
          Each line: {sample_id, prompt, response, label, temperature,
                      token_ids, tokens, metadata}

Stage 2:  BagDataset loads JSONL → pre-tokenize → online extract →
          segment_pooling → [K, instance_dim]
          Produces: mil_ckpt.pt

Stage 3:  Raw prompts + mil_ckpt → VLLMFeatureExporter →
          online PPO → ppo_ckpt.pt
```
