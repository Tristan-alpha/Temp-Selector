## 1. Move segmentation from Stage 1 to BagDataset

- [x] 1.1 Remove `build_segments()` call from `features/build_dataset.py` (both vLLM and API paths); stop computing and writing `segment_spans` to BagSample JSONL. Keep `token_features[].text` and `response` — sufficient for downstream reconstruction.
- [x] 1.2 In `features/schema.py`, make `BagSample.segment_spans` optional (default to empty list) since it is no longer populated by Stage 1.
- [x] 1.3 Add `segment_mode` and `segment_size` params to `mil/training.py` `BagDataset.__init__`; at load time, extract token texts from `token_features[].text`, call `build_segments(texts, response, mode, size)` internally, then feed the computed spans to `segment_pooling()`.
- [x] 1.4 Update all `BagDataset` callers to pass `segment_mode`/`segment_size` from config.
- [x] 1.5 Update tests: adjust `BagDataset` constructor calls to include new params; update any test that checks `segment_spans` presence in JSONL output.

## 2. Rename feature_mode

- [x] 2.1 Rename `combined` / empty → `topk_logits` in all 9 config files; remove `logits_topk` alias if present
- [x] 2.2 Clean up `inference/vllm_runner.py` and `inference/api_runner.py`: `"combined"` → `"topk_logits"`, remove `"logits_topk"` alias if present

## 3. Unify dataset paths

- [x] 3.1 Update 6 ablation configs: all `all_dataset`/`train_dataset`/`val_dataset`/`test_dataset` paths → same as `base.yaml`
- [x] 3.2 Update `configs/ppo_control.yaml` with shared base paths
- [x] 3.3 Verify all 9 configs have identical dataset paths

## 4. New configs

- [x] 4.1 Create `configs/hidden_states.yaml`: `feature_mode: hidden_states`, `instance_dim: 4096`, shared dataset paths, dedicated ckpt paths
- [x] 4.2 Verify `configs/ppo_control.yaml` already exists (created earlier)

## 5. PPO training reads from train_dataset

- [x] 5.1 Add `load_train_prompts(dataset_path)` to `ppo/training.py` — extracts unique (question, answer) pairs from labeled train JSONL via `sample_prefix` dedup
- [x] 5.2 Replace `_load_prompts(data_path)` call in `train_ppo()` with `load_train_prompts(cfg["paths"]["train_dataset"])`
- [x] 5.3 Remove `--data` CLI arg from `ppo/training.py` main() and replace with `--train-data` (default to `paths.train_dataset`)

## 6. Verification

- [x] 6.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 6.2 Run `python -m compileall -q` on all modified files
- [x] 6.3 Verify all 9 configs parse
- [x] 6.4 Verify no remaining `raw_input` references outside `build_dataset.py` and config

## 7. Documentation

- [x] 7.1 Update README.md config table
- [x] 7.2 Update PIPELINE.md: config reference, segmentation section, PPO data flow
- [x] 7.3 Update CLAUDE.md: configs count (9), feature_mode values, raw_input scope
