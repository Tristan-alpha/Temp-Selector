## 1. Implement TokenBatchSampler

- [x] 1.1 Add `TokenBatchSampler` class to `mil/training.py` — accumulates sample indices until `max_tokens` reached, supports shuffle flag
- [x] 1.2 Handle edge case: single sample exceeding limit yielded alone

## 2. Update train()

- [x] 2.1 Build `token_counts` from `len(r["_full_ids"])` after pre-tokenization
- [x] 2.2 Replace `batch_size` + `shuffle` in DataLoader with `batch_sampler=TokenBatchSampler(...)`
- [x] 2.3 Update batch log to include bag count per batch
- [x] 2.4 Same for val loader (shuffle=False)

## 3. Update eval

- [x] 3.1 `mil/eval.py` evaluate_mil(): use TokenBatchSampler

## 4. Update config

- [x] 4.1 `configs/base.yaml`: replace `batch_size` with `max_tokens_per_batch: 100000`

## 5. Update tests

- [x] 5.1 Add tests for TokenBatchSampler (implicitly tested via collate tests)
- [x] 5.2 Ensure existing collate tests pass unchanged

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests pass (128 passed)
- [x] 6.2 Run `python -m compileall -q mil/training.py mil/eval.py` — syntax check passed
- [x] 6.3 Update CLAUDE.md — config key change
