## 1. Create SGLangHiddenStateExtractor (thin wrapper)

- [x] 1.1 Create `inference/sglang_hidden_extractor.py` with `SGLangHiddenStateExtractor` class. Constructor receives `sglang.Engine`. `extract(prompts, responses) -> List[torch.Tensor]` forwards to `engine.generate(batch_prompts, batch_params, return_hidden_states=True)`, slices off prompt tokens, returns response tensors in native dtype. No internal batching.

## 2. Remove hidden sidecar emission from build stage

- [x] 2.1 `scripts/build_dataset.py`: for hidden_states/all mode, stop collecting `hidden_tensor` from payloads and skip `write_hidden_sidecar()`. Keep JSONL writing unchanged (token features without hidden).
- [x] 2.2 `scripts/split_jsonl.py` / `scripts/subsample_jsonl.py`: remove `split_hidden_sidecar()` calls.

## 3. Batch hidden extraction in BagDataset (caller-side batching)

- [x] 3.1 Modify `mil/training.py` `BagDataset.__init__` to accept optional `extractor` and `hidden_batch_size` parameters. When extractor provided and `feature_mode in {"hidden_states","all"}`: group rows into batches of `hidden_batch_size`, call `extractor.extract()` per batch, immediately `segment_pooling` → `self.rows`, free hidden states. Remove sidecar mmap path.
- [x] 3.2 `train_mil()`: create SGLang engine → wrap in `SGLangHiddenStateExtractor` → pass to BagDataset with `hidden_batch_size` from config → shutdown engine after datasets loaded.

## 4. MIL eval with batch extraction

- [x] 4.1 `mil/eval.py`: create SGLang engine → wrap in extractor → pass to `BagDataset` → evaluate → shutdown.

## 5. Cleanup

- [x] 5.1 Delete `utils/dataset_io.py`. Redirect all remaining `write_jsonl` / `load_jsonl` imports to `utils/jsonl.py` (functionally identical). No sidecar functions remain.
- [x] 5.2 `features/schema.py`: remove `to_binary_dict()`.
- [x] 5.3 Config: add `mil.training.hidden_batch_size` (default 256); remove `eagle_aux_hidden_state_layer_ids` from dataset.yaml.

## 6. Tests

- [x] 6.1 Add tests for `SGLangHiddenStateExtractor` with mocked engine.
- [x] 6.2 Remove sidecar-related tests from `test_dataset_io.py`.
- [ ] 6.3 `python -m pytest tests/ -v` — all pass.

## 7. Verification

- [ ] 7.1 Verify build_dataset produces JSONL only, no .hidden.safetensors
- [ ] 7.2 Verify MIL training with batch hidden extraction end-to-end
- [ ] 7.3 Update CLAUDE.md and PIPELINE.md
