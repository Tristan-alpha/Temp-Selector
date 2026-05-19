## Why

Current PPO training runs V=8 generation chains per prompt but shares a single temperature decision across all chains. Only chain 0 drives the next prompt and observation; chains 1-7 contribute only to majority voting. This means the policy does not learn from the temperature-to-correctness relationship of individual chains — if chain 0 happens to produce wrong answer while 7/8 chains are correct, the policy still sees only chain 0's trajectory. The majority voting objective is only a terminal reward, not captured in the intermediate policy decisions across chains.

## What Changes

Each of the V chains becomes an independent episode with its own policy decisions, observations, and accumulated text. The terminal majority-vote reward (±1) is propagated to every chain's trajectory. This directly aligns the PPO objective with majority-vote correctness: the policy learns temperature decisions that make each individual chain more likely to contribute to a correct majority.

## Capabilities

### Modified Capabilities

- `ppo-online-generation`: Each chain per prompt is an independent episode with own (obs, action, reward) trajectory. Policy makes a temperature decision per chain per segment.

## Impact

- `ppo/training.py`: Major restructuring of episode bookkeeping — `segment_obs`, `ep_*`, `active`, `generated` all become `[N][V]` indexed. PPO batch construction iterates over all chains.
- `ppo/eval.py`: Same restructuring.
