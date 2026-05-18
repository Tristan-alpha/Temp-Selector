## Why

`_LogprobsComputeFn` does internal chunking and returns unused `id_chunks`, wasting 50% of pickle bandwidth. Moving chunking to `extract_from_ids` simplifies the compute function to a single-chunk pure function and allows cat on the training GPU to avoid an extra `.to(device)` transfer.

## What Changes

- Simplify `_LogprobsComputeFn`: remove internal chunk loop, process one chunk only, return only `logprobs` (not `ids`)
- Move chunk loop to `extract_from_ids`: split hidden states into chunks, call `apply_model` per chunk, cat on specified device
- Add `device: torch.device | None = None` parameter to `extract_from_ids`

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: `_LogprobsComputeFn` is now single-chunk; chunking+cat happens in `extract_from_ids`

## Impact

- `inference/vllm_runner.py`: `_LogprobsComputeFn` (~30 lines → ~15), `extract_from_ids` (+chunk loop)
- `mil/training.py`: collate_fn passes `cat_device=train_device`
- `mil/eval.py`: auto-updated via shared `make_collate_fn`
