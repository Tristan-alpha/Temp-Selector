## 1. Rewrite `_resolve_parallel_size` and `__init__`

- [x] 1.1 Replace `parallel_size: int | str | None = "auto"` with `parallel_size: int | None = None` in `__init__`
- [x] 1.2 Replace `engine_preset: str = "decode"` with `reserve_training_gpu: bool = False` in `__init__`
- [x] 1.3 Rewrite `_resolve_parallel_size`: use `torch.cuda.device_count()` only, error on 0 GPUs, error if reservation leaves 0 GPUs

## 2. Update callers

- [x] 2.1 `mil/training.py`: change `engine_preset="prefill"` → `reserve_training_gpu=True`, change `parallel_size` default from `"auto"` to `None`
- [x] 2.2 `mil/eval.py`: same changes as mil/training.py
- [x] 2.3 `scripts/build_dataset.py`: change `parallel_size` default from `"auto"` to `None`

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 3.2 Run `python -m compileall -q inference/vllm_runner.py mil/training.py mil/eval.py scripts/build_dataset.py`
- [x] 3.3 Check whether docs need updating (no file moves, no config key changes, no new pitfalls — skip)
