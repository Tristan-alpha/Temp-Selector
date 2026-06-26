## Context

`segment_pooling` in concat mode currently drops the last segment when `chunk.shape[0] < segment_size`. For fixed_window segmentation, this means the final segment of a reasoning chain (typically containing `\boxed{answer}`) is discarded. PPO is unaffected because it uses mean pooling, but MIL training with concat pooling would lose critical answer-segment features.

Separately, PPO's reward scheme assigns `terminal_reward = ±1` entirely to the final decision step, with intermediate steps receiving `shaping_coef × terminal_reward × attn_w[t]`. The shaping coefficient (`shaping_coef=0.15` in base.yaml) is an arbitrary hyperparameter that dilutes the credit signal. The MIL attention weights already score each instance's error likelihood — they should directly distribute the terminal reward across all steps without a scaling hyperparameter.

## Goals / Non-Goals

**Goals:**
- Concat mode in `segment_pooling` preserves tail segments via zero-padding
- Terminal reward is distributed across all PPO steps proportional to MIL attention weights
- Remove the `shaping_coef` hyperparameter from config and training code
- MIL attention weights are L1-normalized before reward distribution (sum to |1|)

**Non-Goals:**
- Changing mean-pooling behavior (already handles variable-length segments correctly)
- Changing how MIL attention weights are computed (only how they are consumed by PPO)
- Affecting PPO eval — eval uses terminal reward only for accuracy metrics

## Decisions

### 1. Zero-pad instead of overlap for concat tail segments

**Chosen**: Zero-pad the short chunk to `segment_size × obs_dim`.

**Alternatives considered**:
- *Overlap last window*: Adjust the final fixed_window boundary backward so every segment has exactly `segment_size` tokens. Rejected because it changes segmentation boundaries (violates the fixed_window contract) and would require modifying `fixed_window_segment`, a separate concern.
- *Drop and accept*: Rejected. Losing the `\boxed{answer}` segment degrades MIL error localization where it matters most.
- *Mean-pool the tail as a special case*: Mixes pooling modes, adds complexity for an edge case.

**Rationale**: Zero-padding introduces at most `segment_size - 1` rows of zeros per instance, which is <2% of total dimensions for typical configs (`64 × 4096 = 262144`). This is standard practice in sequence learning (cf. BERT attention masks) and the Linear layer can learn to ignore these dimensions.

### 2. Attention-weighted terminal reward distribution

**Chosen**: Distribute `terminal_reward` across all steps with `softmax(attn_w)` as weights: `reward[t] = terminal_reward × softmax(attn_w)[t]`.

**Alternatives considered**:
- *Keep current shaping scheme*: `shaping_coef × terminal_reward × attn_w` for intermediates, full `terminal_reward` on last step. Rejected because it splits reward semantics arbitrarily — the "shaping" vs "terminal" distinction is a hyperparameter artifact, not a principled decomposition.
- *Uniform distribution*: Rejected. Discards the MIL attention signal entirely.
- *L1-normalize instead of softmax*: `attn_w / attn_w.sum()`. Almost identical to softmax in practice (attention weights already softmax-normalized). Using the raw weights directly (L1-norm) preserves their magnitude, while softmax re-normalizes. Chosen L1-normalize because MIL attention is already a proper distribution (softmax output).

**Rationale**: Each step's contribution to the final correctness is scored by MIL attention. Summing attention-weighted reward across all steps yields `terminal_reward × Σ(w_i) = terminal_reward × 1`, matching the total budget. No hyperparameter needed — the MIL model provides the credit assignment signal directly.

### 3. Graceful fallback when MIL model is absent

**Chosen**: When `mil_model is None` or `attn_weights is None`, revert to uniform distribution: `reward[t] = terminal_reward / n_steps`.

**Rationale**: Without MIL attention, uniform distribution is the least-biased assumption. Still strictly better than the current scheme (full reward on last step, zero on intermediates), which biases the policy toward short chains.

## Risks / Trade-offs

- **Zero-padding may introduce spurious correlation**: The model might associate zeros with correct/incorrect labels. Mitigated by the small fraction of padded dimensions (<2%) and the fact that padded positions are always at the end of the last segment — a consistent structural pattern the model can learn to ignore.
- **Uniform fallback reduces signal for the last step**: When MIL is unavailable and the last step truly determines correctness, uniform distribution under-rewards it. In practice, all PPO configs that enable shaping load a MIL checkpoint.
