## Context

Hidden state vectors (4096 floats per token) are the dominant data component in dataset JSONL files. Currently they are serialized inline via `json.dumps`. The root cause is `VLLMHiddenStateExtractor.extract()` calling `.tolist()` on the hidden state tensor, which converts native dtype (bf16 for Qwen3-8B) to Python float64 lists. These are then serialized as JSON decimal strings (~10 chars per float), wasting space and precision at every step: bf16 (2 bytes) → float64 (8 bytes) → JSON string (~10 bytes). For a 480-sample dataset, this produces 7.3 GB of JSONL, of which ~99% is hidden state vectors.

The conversion already uses `safetensors` in `inference/vllm_hidden_extractor.py` to read hidden states from vLLM's speculative decoding output. The natural extension is to keep hidden states in their native binary format throughout the pipeline.

## Goals / Non-Goals

**Goals:**
- Reduce dataset file size by ~80% for datasets with hidden states (bf16: 2 bytes/float vs ~10 bytes/float in JSON)
- Preserve exact native-dtype values (no `.tolist()` → float64 → JSON string loss; bf16 stays bf16)
- Zero changes to code that doesn't consume hidden states (`ppo/training.py`, `features/dataset_eval.py`, `mil/eval.py`)
- Backward compatibility: old JSONL files without hidden states load unchanged
- No new dependencies

**Non-Goals:**
- Replace JSONL entirely (strings, scalars, metadata stay as JSONL)
- Change the segmentation, pooling, or vectorization logic
- Compress non-hidden fields (token texts, logprobs, entropies)
- Change the config file format or path conventions

## Decisions

### Decision 1: Hybrid JSONL + safetensors sidecar (not msgpack or torch.save)

**Chosen**: Keep JSONL for metadata + one safetensors file per dataset for hidden states.

**Rationale**:
- safetensors is already in `requirements.txt` and proven in the codebase (`vllm_hidden_extractor.py` uses it)
- JSONL for non-hidden data means zero changes in files that don't read hidden states
- msgpack would require a new dependency and still store floats as float64 (8 bytes) unless explicitly downcast; all consumers would need to change
- `torch.save` uses pickle under the hood (security concern) and loads the entire file at once (not streamable)

### Decision 2: Sidecar path derived from JSONL path (not config-driven)

**Chosen**: `datasets/train.jsonl` → `datasets/train.jsonl.hidden.safetensors`.

**Rationale**: Config paths stay as `.jsonl`; sidecar is always derived via `str(Path(p)) + ".hidden.safetensors"`. This means `split_jsonl.py` and `subsample_jsonl.py` can derive the sidecar path from the JSONL path without extra config. No new config keys needed.

### Decision 3: Offset-based indexing (not row-keyed lookup)

**Chosen**: Each JSONL row has `_hidden_offset` (int) and `_hidden_count` (int) pointing into a concatenated `[total_tokens, hidden_dim]` tensor.

**Rationale**: A single flat tensor is the simplest safetensors layout. Offset/count indexing is trivial (just slice). Row-keyed lookup (dict of tensors) would require safetensors metadata to track keys, which is limited to `Dict[str, str]` and would be awkward for 10k+ rows.

### Decision 4: BagDataset patches hidden states into token feature dicts (not a separate tensor path)

**Chosen**: At load time, `BagDataset.__init__` reads the sidecar tensor, slices out each row's hidden states, and sets `token_features[j]["hidden"] = hs_chunk[j].tolist()`.

**Rationale**: The existing `token_to_vec()` in `vectorizer.py` already reads `token_feat.get("hidden")` from the dict. By patching the dict before vectorization, we reuse all existing code paths unchanged. The alternative (passing a separate hidden tensor through the entire pipeline) would touch `segment_pooling`, `token_to_vec`, `token_to_obs`, and all their callers.

### Decision 5: `_hidden_count` reflects actual hidden vectors, not token count

**Rationale**: The hidden extractor may return fewer hidden vectors than tokens (e.g., 255 vs 256). `_hidden_count` stores the actual number of hidden vectors. During patching, `BagDataset` patches `min(count, len(token_features))` tokens. Unpatched tokens keep `hidden=None`, which `vectorizer.py` handles with `or []` → zero-padding.

### Decision 6: Preserve native dtype (bf16) — do not force float32

**Chosen**: Store hidden states in the safetensors sidecar at whatever dtype vLLM produces (typically bf16 for Qwen3-8B). Do not convert to float32 or float64.

**Rationale**: vLLM's `extract_hidden_states` writes safetensors files in the model's native dtype (bf16). The current code calls `.tolist()` on these tensors, which converts bf16 → Python float64. For the binary sidecar, we skip `.tolist()` entirely and write the raw tensor slice directly to the sidecar safetensors file. This gives:
- bf16: 2 bytes/float → ~2 MB per row (256 tokens × 4096 dim)
- vs float32: 4 bytes/float → ~4 MB per row
- vs JSON: ~10 bytes/float → ~10 MB per row

The sidecar's dtype is self-describing (safetensors files encode dtype in their header), so `BagDataset` can read it back correctly without explicit dtype metadata in JSONL.

To make this work, `VLLMHiddenStateExtractor.extract()` needs to NOT call `.tolist()` — it should return raw torch tensors. Callers that need Python lists (e.g., `ppo/training.py` for token features) can call `.tolist()` themselves.

### Decision 7: mmap via `safe_open` + `get_slice` with `with` block (not `load_file`)

**Chosen**: Use Python's `with safetensors.safe_open(...) as f:` context manager to mmap the file, and `f.get_slice("hidden_states")` for lazy access. Put the entire per-row processing loop inside the `with` block.

**Rationale**:
- `safetensors.safe_open` mmaps the file — physical memory is allocated page-by-page on access, not upfront.
- `f.get_tensor(...)` would copy the entire tensor into memory (defeats the purpose).
- `f.get_slice(...)` returns a lazy view; `slice[offset:offset+count, :]` reads only the requested rows into a new tensor (copy on slice).
- Official safetensors docs confirm that `get_tensor()` result survives outside the `with` block (it's a copy). `get_slice()` slices also work outside but we keep the `with` block around for clarity and to guarantee the mmap is released after processing.

`BagDataset` usage pattern:
```python
hpath = dataset_path + ".hidden.safetensors"
if os.path.exists(hpath):
    with safetensors.safe_open(hpath, framework="pt") as f:
        hs = f.get_slice("hidden_states")       # mmap view, lazy
        for row in rows:
            chunk = hs[offset:offset+count, :]   # reads only those rows
            token_feats[j]["hidden"] = chunk[j].tolist()  # detach → Python floats
# with block ends → mmap released
```

**Alternatives considered**:
- `load_file` / `get_tensor` (full copy): simpler but won't scale to 10k+ samples (10+ GB tensors)
- Custom RAII wrapper class: unnecessary — Python's context manager is sufficient

### Decision 8: Merge build+split for hidden_state datasets (no intermediate "all" safetensors)

**Chosen**: When `feature_mode` is `"hidden_states"` or `"all"`, `scripts/build_dataset.py` performs group-aware splitting in-memory after hidden state extraction and writes train/val/test JSONL + safetensors sidecars directly. No giant intermediate "all" safetensors file is created.

**Rationale**: The current pipeline generates a monolithic `all_dataset.jsonl` (with embedded hidden vectors) then runs `split_jsonl.py` to produce train/val/test. With the sidecar design, the "all" safetensors would be the full dataset's hidden states concatenated — an unnecessary intermediate copy that wastes disk. By splitting in-memory after extraction but before writing, each split gets its own compact safetensors from the start.

**How it works**:
1. Generate responses + extract hidden states (same as now)
2. Build BagSample objects in-memory (same as now, but don't write yet)
3. Call `split_by_group(samples, ...)` from `utils/jsonl.py` to split in-memory
4. For each split: write JSONL + build+write safetensors sidecar

The "all" dataset path in config is now optional for hidden_state builds — output goes directly to train/val/test. For basic/topk_logits modes, the existing "all → split" flow is unchanged (no sidecar involved).

`split_by_group` uses `sample_prefix(sample_id)` by default, which groups all temperatures/votes of the same question together — same as the current pipeline.

## Risks / Trade-offs

- **Two-file consistency**: If the process crashes mid-write, JSONL and safetensors may disagree. Mitigation: write JSONL first, then safetensors. A crash during safetensors write means JSONL has `_hidden_offset` keys pointing to a missing/malformed sidecar — `read_hidden_tensor` catches this and returns None, causing BagDataset to fall back to zero hidden states.
- **Extra file to manage**: Each dataset has 2 files instead of 1. Mitigation: the sidecar is always alongside the JSONL with a predictable suffix. All I/O goes through `utils/dataset_io.py` helpers so scripts don't manage this directly.
- **mmap lifecycle**: `BagDataset` processing loop is inside `with safe_open(...) as f:`. The `with` block guarantees the mmap is released after processing. Chunk slices (`hs[offset:offset+count, :]`) create independent copies, so data survives outside the block.
- **Merged build+split changes pipeline**: `scripts/build_dataset.py` now does what used to be build + split in one step. The `scripts/run_pipeline.sh` needs updating. `scripts/split_jsonl.py` is kept for basic/topk_logits mode and for legacy datasets.
- **Memory during build**: Peak memory includes all in-flight BagSample objects + hidden tensors, now held until all 3 splits are written. Mitigation: bf16 at current scale (480 samples × 256 tokens × 4096 × 2 bytes ≈ 1 GB for hidden tensors), plus Python overhead. Acceptable for current scale; larger datasets may benefit from future streaming split (write each row to its split immediately instead of building in-memory lists).
