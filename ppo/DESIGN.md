# PPO Module Design

## Why online PPO (not offline)

In offline RL, the policy learns from pre-collected data where actions don't affect outcomes. If you train on Stage 1 data, the policy sees "temperature T was used, and the answer was correct/wrong." But it cannot learn "if I had chosen T' instead of T, would the answer have been correct?" — the counterfactual is missing.

Online PPO closes this loop:

```
Policy chooses temperature → vLLM generates segment at that temp → features extracted → reward computed
                                                                         ↓
                                                          next segment's observation
```

The policy's action **causally affects** the next state and eventual reward. This is why the PPO loop runs against a live vLLM engine, not a dataset.

## Per-segment generation loop

Each prompt is generated segment-by-segment (default 32 tokens per segment via `segment_size`):

```
Round 0: prompt → default T=0.7 → generate segment₀ → extract segment_obs₁
Round 1: prompt+seg₀ → PPO(T|obs₁) → generate segment₁ → extract segment_obs₂
Round 2: prompt+seg₀+seg₁ → PPO(T|obs₂) → generate segment₂ → extract segment_obs₃
...
Until EOS or max_tokens
```

**vLLM APC optimization**: The shared prompt prefix KV-cache is reused across rounds via vLLM's Automatic Prefix Caching. Only newly generated tokens incur computation — the prompt is not re-encoded each round.

**Multiple votes**: `num_votes` completions are generated per prompt. The first vote drives temperature decisions; all votes share the same temperature sequence. Majority voting at the end determines the terminal reward. This keeps the reward signal consistent with Stage 1 and Stage 4.

## Reward design

### Terminal reward (end of episode)

```
reward = +1.0 if majority_vote_correct else -1.0
```

Self-consistency majority voting: `self_consistency_correct()` extracts the math answer from each of the `num_votes` completions, finds the modal (plurality) answer via `Counter.most_common(1)`, and compares it to the gold answer. This is the primary optimization signal.

### Shaping reward (intermediate steps)

```
reward = shaping_coef × (1 − sigmoid(inst_logit))
```

`inst_logit` comes from the pre-trained MIL model. Higher inst_logit means "this segment looks like an error." The shaping reward gives the policy immediate feedback without waiting for the final answer — segments that look erroneous are penalized immediately.

The default `shaping_coef = 0.15` keeps the shaping signal gentle. Too high and the policy overfits to MIL's opinion of what "looks wrong"; too low and intermediate steps get no learning signal (most terminal rewards are -1, and the credit assignment problem makes individual step learning hard).

## GAE + PPO mechanics

### GAE (Generalized Advantage Estimation)

```python
for t in reversed(range(T)):
    delta = reward[t] + gamma × value[t+1] × (1−done[t]) − value[t]
    advantage[t] = delta + gamma × lambda × (1−done[t]) × advantage[t+1]
```

Key detail: `done[t]` uses `mask = 1 − done[t]`. When done=1 (terminal step), the mask zeros out future value bootstrapping, preventing advantage from propagating across episode boundaries. After computing all advantages, they are standardized to mean=0, std=1 for stable policy gradient updates.

### PPO clipped objective

```
ratio = exp(new_logprob − old_logprob)
L_clip = −min(ratio × advantage, clip(ratio, 1−ε, 1+ε) × advantage)
```

The clip (`ε=0.2`) prevents the policy from changing too much in a single update. Combined with multi-epoch mini-batch updates (8 epochs over the rollout data), this provides stable online learning.

## Overfitting diagnostic

Each rollout is split into training (80%) and validation (20%) indices. After each PPO update:

```
value      = MSE(ret, value_pred) on training split
val_value  = MSE(ret, value_pred) on validation split
```

| Pattern | Meaning |
|---|---|
| `value ↓`, `val_value ↓` | Healthy learning |
| `value ↓`, `val_value ↑` | Overfitting — policy memorizing this specific rollout |
| Both `↑` | Value divergence — reduce learning rate or shaping_coef |

## First-segment dummy values

The first segment has no prior observation (`segment_obs[i] is None`) because there are no previously generated tokens to extract features from. A default temperature (0.7) is used, and dummy values (zero observation, zero logprob, zero value) are appended to episode tracking lists. These are safely skipped during PPO batch construction by `range(1, n_steps)` — they exist only to keep array indices aligned.

## ep_correct vs MIL label convention

| Variable | 1 means | 0 means | Where |
|---|---|---|---|
| `label` (MIL) | error (positive bag) | correct (negative bag) | Stage 1 & 2 |
| `ep_correct` (PPO) | majority correct | majority wrong | Stage 3 |

These are **opposite conventions**. The naming `ep_correct` (not `ep_labels`) makes this explicit. Reading `if ep_correct[i] > 0` is unambiguously "was this episode's answer correct?"
