## 1. Simplify _LogprobsComputeFn

- [x] 1.1 Remove internal CHUNK_SIZE loop, process one chunk only
- [x] 1.2 Remove `id_chunks` computation and return (unused by caller)
- [x] 1.3 Return only `result.logprobs.cpu()` — a single CPU tensor

## 2. Move chunking to extract_from_ids

- [x] 2.1 Add CHUNK_SIZE loop in `extract_from_ids` around `apply_model` calls
- [x] 2.2 Add `device` parameter, cat chunks on that device (or CPU if None)
- [x] 2.3 Clean up old internal chunking references

## 3. Update callers

- [x] 3.1 `mil/training.py`: pass `device=train_device` in collate_fn
- [x] 3.2 Verify `mil/eval.py` works via shared `make_collate_fn`

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 4.2 Run `python -m compileall -q` on all modified files
- [x] 4.3 Check whether docs need updating (no API changes visible to end users — skip)
