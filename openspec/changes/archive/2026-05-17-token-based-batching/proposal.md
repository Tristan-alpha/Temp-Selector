## Why

MIL training batches are currently sized by bag count (`batch_size=128`). But per-bag token count varies widely (72–8192 tokens), so a fixed-size batch can hold anywhere from 9K to 1M total tokens. This makes SGLang prefill KV cache usage unpredictable — a batch of long responses can overflow the `max_total_num_tokens=128K` limit and OOM. Changing `batch_size` to cap total tokens per batch eliminates the variance.

## What Changes

- Replace config key `mil.training.batch_size` with `mil.training.max_tokens_per_batch` (default ~100K, leaving ~28K headroom for p95 fluctuation)
- `collate_fn` accumulates bags until the next bag would exceed the token limit, then yields the batch
- DataLoader wrapper (or custom sampler) feeds bags to collate_fn in shard-and-flush style
- `num_workers` stays 0; batching logic lives in the collate path
- Per-batch log shows token count alongside bag count

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: collate_fn SHALL accumulate rows until total token count reaches `max_tokens_per_batch`, then flush and start a new batch

## Impact

- `mil/training.py` — collate_fn batching logic, config key rename, logging
- `mil/eval.py` — same collate_fn, config key rename
- `configs/base.yaml` — `batch_size` → `max_tokens_per_batch`
