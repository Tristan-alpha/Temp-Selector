## 1. Core change

- [x] 1.1 Modify reward construction in `ppo/training.py`: for incorrect chains (`terminal_reward < 0`), use attention-weighted distribution; for correct chains (`terminal_reward > 0`), use uniform distribution `terminal_reward / n_eff`.

## 2. Docs and specs

- [x] 2.1 Update `ppo/DESIGN.md` — document the asymmetric reward scheme.
- [x] 2.2 Update `memory/ppo-reward-attention-distributed.md` — note the asymmetry.

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — all tests must pass.
- [x] 3.2 Run `python -m compileall -q ppo/training.py` — no syntax errors.
