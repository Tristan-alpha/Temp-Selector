# PPO Module Design

## Prefix-potential PPO

The Full path keeps one frozen Prefix Value GRU state per vote. Policy and
critic heads consume its 1024-dimensional hidden state. Non-terminal rewards
use `lambda * (gamma * V_next - V_current)`; terminal transitions use
`majority_reward - lambda * V_current`. EOS segments update value before close.

## Why online PPO

Offline RL cannot learn causal policies — pre-collected data has no counterfactuals. Online PPO closes the loop:

```
Policy chooses temperature → vLLM generates segment → features extracted → reward
                                                                      ↓
                                                       next segment's observation
```

## Per-segment generation loop

Each prompt is generated segment-by-segment (segment_size=512 tokens, fixed_window):

```
Round 0: prompt → default T=0.7 → generate 512 tokens → extract segment_obs₁
Round 1: prompt+seg₀ → PPO(T|obs₁) → generate 512 tokens → extract segment_obs₂
...
Until EOS or max_tokens
```

vLLM APC reuses prompt KV-cache across rounds.

## Reward design

### Terminal reward

```
reward = +1.0 if majority_vote_correct else -1.0
```

Self-consistency majority voting over `num_votes` completions.

### Shaping reward (intermediate steps)

```
reward = shaping_coef × (1 − sigmoid(inst_logit))
```

`inst_logit` from pre-trained MIL model. Default `shaping_coef = 0.15`.

## PPO architecture

```
PolicyValueNet (~150K params)
  backbone: 3-layer MLP (4098→1024→1024→1024)
  pi head:  Linear(1024→15)   # 15 temperature bins
  v head:   Linear(1024→1)    # value

MIL warm-start: backbone first 2 layers from MIL encoder
pi bias init: best-fixed temp=+5, rest=-5
```

## GAE + PPO Clip

```python
for t in reversed(range(T)):
    delta = reward[t] + gamma × value[t+1] × (1−done[t]) − value[t]
    advantage[t] = delta + gamma × lambda × (1−done[t]) × advantage[t+1]

ratio = exp(new_logprob − old_logprob)
L_clip = −min(ratio × A, clip(ratio, 1−ε, 1+ε) × A)
```

Done flags prevent advantage propagation across episode boundaries. Advantages standardized to mean=0, std=1.

## Overfitting diagnostic

| Pattern | Meaning |
|---|---|
| `value ↓`, `val_value ↓` | Healthy |
| `value ↓`, `val_value ↑` | Overfitting |
| Both `↑` | Value divergence |

## First-segment dummy values

First segment has no prior observation. Default T=0.7, dummy values (zero obs, zero logprob) keep episode lists aligned. Skipped during PPO batch construction via `range(1, n_steps)`.

## Label conventions

| Variable | 1 means | 0 means | Used by |
|---|---|---|---|
| `individual_label` | error | correct | MIL training/eval |
| `voting_label` | error | correct | PPO temperature bias (via `load_temperature_labels`) |
| `ep_correct` | correct | wrong | PPO training/eval (runtime-computed) |
