## 1. Restructure data to per-chain indexing

- [x] 1.1 Change `active` from `List[bool]` (N) to `List[List[bool]]` (N × V), per-chain stop tracking
- [x] 1.2 Change `segment_obs` from `List[Optional[Tensor]]` to `List[List[Optional[Tensor]]]` (N × V)
- [x] 1.3 Change `ep_obs`, `ep_actions`, `ep_logprobs`, `ep_values` to `List[List[List[Tensor]]]` indexed `[i][v][t]`

## 2. Update generation loop

- [x] 2.1 Per-chain policy decisions: iterate over active chains `(i, v)` instead of just `i`
- [x] 2.2 Flatten `(i, v)` into round_prompts/round_temps for `generate_with_features`
- [x] 2.3 Zip results back into per-chain `generated[i][v]` and `segment_obs[i][v]`
- [x] 2.4 Remove `max_rounds +1` (unnecessary overflow guard)

## 3. Update PPO batch construction

- [x] 3.1 Flatten all chains into PPO batch: iterate `(i, v)` → `range(1, n_steps)`
- [x] 3.2 Apply shared terminal reward: same `±1` to all chains of prompt `i`

## 4. Update PPO eval

- [x] 4.1 Mirror per-chain indexing changes in `ppo/eval.py`

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 5.2 Run `python -m compileall -q ppo/training.py ppo/eval.py` to catch syntax errors
