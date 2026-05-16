## Context

BagDataset currently extracts logprob/hidden features during `__init__` by calling SGLangRunner across `hidden_batch_size` chunks. Each chunk's tensor results are `.tolist()`'d and patched into `token_features` dicts living in `all_rows`. By the end of `__init__`, all 9600+ rows have accumulated Python-list-based feature vectors — `hidden_batch_size` controls SGLang call frequency but not memory peak.

The user wants extraction to happen lazily in the DataLoader collate function, so memory peak equals exactly one training batch.

## Goals / Non-Goals

**Goals:**
- BagDataset stores only row metadata; no SGLang calls, no tensor materialization in `__init__`
- Per-batch SGLang extraction happens in `collate_fn` just before the training step
- Memory peak is bounded by training `batch_size`, not dataset size
- SGLang extraction uses per-sample temperatures (already supported by runner)

**Non-Goals:**
- Changing the SGLangRunner API (`extract_hidden`, `extract_logprobs` stay as-is)
- Changing PPO training/eval (already truly online)
- Changing feature vector dimension or pooling logic

## Decisions

### Decision 1: Collate function as the extraction point

Chose `collate_fn` over `__getitem__` because:
- SGLangRunner cannot be pickled → `num_workers` must be 0 → extraction MUST happen in the main process
- `collate_fn` runs in the main process and receives the full batch at once
- SGLang batch-prefill benefits from larger batches (the whole training batch goes as one prefill call)

**Alternative considered:** Keep extraction in `__getitem__` with num_workers=0. Rejected: would lose SGLang batching — one prefill call per row is 32-256x slower.

### Decision 2: BagDataset carries extractor reference for both train and val

BagDataset constructor stores `self.extractor` and `self.feature_mode` so collate_fn can access them. The same extractor (SGLangRunner) is shared across train and val loaders.

**Alternative considered:** Close over extractor in a lambda. Rejected: less transparent, harder to debug.

### Decision 3: `num_workers=0` for MIL DataLoader

Required because SGLangRunner's Engine holds GPU state and cannot be serialized. The MIL model is ~500K params with negligible per-step compute, so prefetching is unnecessary — the bottleneck is SGLang prefill, not training.

### Decision 4: Segments rebuilt in collate_fn, not loaded from JSONL

`segment_spans` exist in JSONL but are rebuilt in collate_fn from `token_texts` (consistent with current `__init__` behavior). Segment computation is O(n_tokens) and cheap compared to SGLang prefill.

## Risks / Trade-offs

**[Risk] SGLang Engine and MIL model GPU memory conflict**
→ Mitigation: MIL model ~500K params + activations ~150 MB; Engine uses `mem_fraction_static=0.80`. On a 40GB+ GPU the remaining 20% (8GB) is ample. If OOM occurs, reduce `batch_size` or `gpu_memory_utilization`.

**[Risk] Re-extraction per epoch wastes GPU time**
→ Mitigation: The user accepts this trade-off. Mean response length is only 502 tokens (median 365), so per-epoch extraction is ~9600 × 502 × 4096 × 50 epochs ≈ 240M tokens of SGLang prefill total. With larger batch_size (128-256), the overhead is acceptable.

**[Risk] DataLoader with num_workers=0 is synchronous**
→ Mitigation: SGLang prefill dominates training step time (>90%). Prefetching wouldn't help because the GPU is busy during both prefill and training.

## Open Questions

- Final MIL `batch_size` value: 128 or 256? (Will be set in config after user decision)
