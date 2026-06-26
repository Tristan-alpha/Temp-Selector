## Context

MIL training (`mil/training.py`) and evaluation (`mil/eval.py`) both pre-compute segment features via vLLM `extract_from_ids` before the training/eval loop. This is the dominant time cost. The extracted features (`List[Dict[str, Any]]` where each dict has `instances`, `label`, `temp_idx`) are deterministic given fixed dataset and extraction parameters.

## Goals / Non-Goals

**Goals:**
- Skip vLLM extraction on cache hit — instant reload from disk
- Cache key covers all parameters that affect segment feature content
- Works for both MIL training and MIL evaluation

**Non-Goals:**
- Cache invalidation (manual deletion when parameters change; filename encodes all parameters so different params → different file)
- PPO feature caching (features change per iteration as policy evolves)
- Cache compression (`.pt` uses pickle, adequate for 12 GB)

## Decisions

### 1. Cache file naming: dash-separated key components

```
datasets/cache/{split}-{segment_mode}-{pooling_mode}-{feature_mode}-{instance_dim}-{segment_size}.pt
```

Example:
```
datasets/cache/train-fixed_window-concat-topk_logprobs-64-64.pt
datasets/cache/val-fixed_window-concat-topk_logprobs-64-64.pt
```

**Why dash over underscore:** `segment_mode` values (`fixed_window`) and `feature_mode` values (`topk_logprobs`) use underscores internally. Dashes as field separators make the boundary between fields unambiguous.

### 2. Format: `torch.save(list_of_dicts, path)`

Each dict in the list has keys `instances` (tensor), `label` (float), `temp_idx` (int). `torch.save` handles nested structures directly — no serialization layer needed.

**Alternatives considered:**
- `safetensors` — only handles dict-of-tensors, not lists of mixed-type dicts.
- JSONL — tensors would need base64 encoding, wasteful for ~12 GB.

### 3. Cache check location: inline in `train()` and `evaluate_mil()`

A helper function `_load_or_build_segment_cache(dataset_rows, runner, collate_fn, split, config, ...)` that:
1. Computes cache path from config keys
2. If path exists → `torch.load(path)`
3. Else → run vLLM extraction → `torch.save(result, path)` → return result

### 4. No config key needed

Cache path is fully derived from existing config values (`data.segment_mode`, `data.segment_pooling`, `inference.feature_mode`, `data.instance_dim`, `data.segment_size`). Adding a config key for cache path adds maintenance burden without benefit — the derivation is deterministic and unambiguous.

### 5. Shared cache directory: `datasets/cache/`

Created automatically on first use. Not cleaned up automatically (manual `rm datasets/cache/*.pt` when needed).

## Risks / Trade-offs

- **[Risk]** Stale cache after dataset change → Mitigation: filename includes all extraction params, but does NOT include a dataset content hash. If the dataset JSONL changes but filename stays the same, cache is stale. → Low risk in practice (datasets are versioned by filename and rebuild is explicit).
- **[Risk]** 12 GB disk for concat pooling → trade-off between disk and GPU time. Manual cleanup when needed.
