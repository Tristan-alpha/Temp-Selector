## 1. Extractor: return native-dtype tensors

- [x] 1.1 Modify `VLLMHiddenStateExtractor.extract()` in `inference/vllm_hidden_extractor.py` to return `List[torch.Tensor]` (native dtype, bf16) instead of `List[List[List[float]]]`. Remove `.tolist()` call. Update callers in `ppo/training.py` to call `.tolist()` where Python lists are needed.

## 2. Schema & I/O layer

- [x] 2.1 Add `to_binary_dict(hidden_offset)` method to `BagSample` in `features/schema.py` — serializes everything except hidden vectors (sets to `None`), emits `_hidden_offset` and `_hidden_count` when hidden states exist
- [x] 2.2 Create `utils/dataset_io.py` with:
  - `hidden_path(dataset_path)` → derive sidecar path
  - `write_hidden_sidecar(dataset_path, tensors)` → build and write safetensors from list of torch/numpy tensors at native dtype
  - `read_hidden_offsets(row_dict)` → extract `(_hidden_offset, _hidden_count)`

## 3. Build pipeline (merged build+split for hidden_state mode)

- [x] 3.1 Modify `scripts/build_dataset.py`:
  - After hidden state extraction, collect hidden tensors per sample at native dtype
  - Use `split_by_group` from `utils/jsonl.py` to split samples in-memory
  - Write train/val/test JSONL + safetensors sidecars directly (no intermediate "all" safetensors)
  - Refactor duplicated vLLM/API backend writing logic into a shared helper
  - For basic/topk_logits modes: retain existing "all → split" flow (no sidecar)

## 4. Load pipeline

- [x] 4.1 Modify `BagDataset.__init__` in `mil/training.py`:
  - Check for sidecar via `os.path.exists(hidden_path(data_path))`
  - If exists: open `with safetensors.safe_open(...) as f:`, use `f.get_slice("hidden_states")` for lazy mmap access
  - Place the per-row processing loop inside the `with` block
  - For each row: `chunk = hs[offset:offset+count, :]` → `.tolist()` → patch `token_features[j]["hidden"]`

## 5. Split & subsample (for basic/topk_logits mode and legacy datasets)

- [x] 5.1 Modify `scripts/split_jsonl.py` to call `write_hidden_sidecar()` for each output split when the source has a sidecar
- [x] 5.2 Modify `scripts/subsample_jsonl.py` to call `write_hidden_sidecar()` for the output when the source has a sidecar

## 6. Tests

- [x] 6.1 Add `test_to_binary_dict_with_hidden` and `test_to_binary_dict_no_hidden` to `tests/test_schema.py`
- [x] 6.2 Create `tests/test_dataset_io.py` with round-trip tests: write hidden tensors at bf16/float32 → read via HiddenSidecar → verify identity. Test that mmap is released after close.

## 7. Verification

- [x] 7.1 Run `python -m pytest tests/test_schema.py tests/test_dataset_io.py -v` — all tests pass
- [x] 7.2 Run `python -m compileall -q features/schema.py utils/dataset_io.py inference/vllm_hidden_extractor.py scripts/build_dataset.py scripts/split_jsonl.py scripts/subsample_jsonl.py mil/training.py ppo/training.py`
- [x] 7.3 Update `CLAUDE.md` and `PIPELINE.md`: `utils/dataset_io.py` module, extractor return type change, merged build+split for hidden_state mode
