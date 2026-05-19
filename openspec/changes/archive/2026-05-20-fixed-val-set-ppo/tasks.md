## 1. Load fixed val set

- [x] 1.1 Load val prompts from `paths.val_dataset` at training start
- [x] 1.2 Select a fixed random subset (e.g., 16 prompts) as `val_fixed`

## 2. Per-iteration val rollout

- [x] 2.1 After PPO update, run val rollout on `val_fixed` with current policy (argmax, no sampling)
- [x] 2.2 Compute `val_value` from val rollout trajectories

## 3. Remove old 80/20 split

- [x] 3.1 Use all training rollout samples for PPO update (no `val_idx` split)
- [x] 3.2 Keep `val_ratio` config key but ignore it (backward compat)

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 4.2 Run `python -m compileall -q ppo/training.py` — syntax OK
