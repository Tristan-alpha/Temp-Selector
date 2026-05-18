## 1. Add CLI args and remove config reads

- [x] 1.1 `mil/training.py`: add `--parallel-size` CLI arg (type=int, default=None), pass `args.parallel_size` instead of `cfg["inference"].get("parallel_size")`
- [x] 1.2 `mil/eval.py`: same changes as mil/training.py
- [x] 1.3 `scripts/build_dataset.py`: same changes as mil/training.py
- [x] 1.4 `ppo/training.py`: remove `inf.get("parallel_size", "auto")` fallback, keep existing `--parallel-size` arg, pass directly

## 2. Verification

- [x] 2.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 2.2 Run `python -m compileall -q` on all modified files
- [x] 2.3 Check whether docs need updating (no config key changes, no file moves — skip)
