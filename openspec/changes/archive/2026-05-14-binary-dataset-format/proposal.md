## Why

Hidden state vectors (4096 floats per token) serialized as JSON text inflate dataset files to ~10 MB per sample — a 480-sample dataset reaches 7.3 GB. The root cause: `VLLMHiddenStateExtractor.extract()` calls `.tolist()` on bf16 tensors, converting them to Python float64 lists, which JSON serializes as verbose decimal strings. Storing in native bf16 via safetensors cuts size by ~80% (2 bytes vs ~10 bytes per float) and eliminates precision loss from all intermediate conversions.

## What Changes

- Add a `to_binary_dict()` method on `BagSample` that emits offset metadata instead of inline hidden vectors
- Create `utils/dataset_io.py` with shared helpers for reading/writing safetensors hidden-state sidecar files
- **BREAKING**: `VLLMHiddenStateExtractor.extract()` now returns `List[torch.Tensor]` (native dtype, bf16) instead of `List[List[List[float]]]`. Callers that need Python lists must call `.tolist()` themselves.
- **BREAKING**: `scripts/build_dataset.py` now writes hidden states to a companion `.hidden.safetensors` file instead of embedding them in JSONL rows
- `mil/training.py` BagDataset reads hidden states from the sidecar file at load time and patches token feature dicts
- `scripts/split_jsonl.py` and `scripts/subsample_jsonl.py` split/trim the sidecar alongside JSONL outputs
- Old JSONL files without hidden states continue to work (no sidecar → no patching)

## Capabilities

### New Capabilities

- `binary-hidden-storage`: Hidden state vectors are stored as float32 tensors in safetensors files alongside JSONL, replacing inline JSON serialization

### Modified Capabilities

<!-- No existing specs to modify -->

## Impact

- 6 files modified, 1 new file, 2 test files
- No new dependencies (safetensors already in requirements.txt)
- `ppo/training.py`, `features/dataset_eval.py`, `mil/eval.py` unchanged (they don't read hidden states from JSONL)
- Config paths unchanged (sidecar path derived automatically from JSONL path)
