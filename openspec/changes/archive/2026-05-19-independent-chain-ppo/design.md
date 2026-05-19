## Context

Current architecture shares a single temperature per prompt across all V=8 generation chains. Only chain 0's response drives subsequent prompts and policy observations. This breaks the majority-voting objective because the policy learns from chain 0's trajectory alone.

## Goals / Non-Goals

**Goals:**
- Each chain independently receives its own temperature from the policy
- Each chain accumulates its own text, observation, and episode steps
- Majority voting across chains determines terminal reward, propagated to ALL chains
- Each chain contributes independently to the PPO batch

**Non-Goals:**
- Changing majority voting logic
- Changing `generate_with_features` interface
- Changing config schema

## Decisions

### Decision 1: Per-chain active tracking

**Choice**: Replace `active: List[bool] * N` with `active: List[List[bool]] = [[True]*V for _ in range(N)]`. Each chain stops independently on EOS or stop.

**Why**: Chains may produce different length responses at the same temperature. When one chain stops, others should continue.

### Decision 2: Per-chain episode trajectories

**Choice**: `ep_obs[i][v]`, `ep_actions[i][v]`, `ep_logprobs[i][v]`, `ep_values[i][v]` become `List[List[List[Tensor]]]` indexed by `[prompt_idx][chain_idx][step]`.

**Why**: Each chain is now an independent episode that needs its own trajectory for PPO.

### Decision 3: Shared terminal reward

**Choice**: `ep_correct[i]` is still per-prompt (majority vote across chains). Terminal reward `±1` is applied to every chain's last step for prompt `i`.

**Why**: All chains contribute to the majority vote result. The credit/blame is shared equally. GAE + value function propagate this back through each chain's policy decisions.

## Risks / Trade-offs

- **PPO batch size grows V×**: N prompts × V chains = N×V episodes. This provides more training data but increases compute. Mitigated by reducing `online_rollout_size` if needed.
- **Independent chains may diverge early**: Different temperatures → different first segments → very different continuation prompts. This increases exploration diversity but may produce degenerate chains. Mitigated by entropy bonus in PPO loss.
