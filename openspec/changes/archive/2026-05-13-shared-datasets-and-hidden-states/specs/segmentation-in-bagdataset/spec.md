## ADDED Requirements

### Requirement: BagDataset calls build_segments internally
`BagDataset.__init__` SHALL accept `segment_mode` and `segment_size` parameters and call `build_segments(token_texts, response, mode, size)` internally, rather than reading pre-computed `segment_spans` from JSONL.

#### Scenario: Segmentation at load time
- **WHEN** `BagDataset` loads a JSONL row with `token_features` and `response`
- **THEN** it extracts token texts and calls `build_segments(texts, response, mode, size)` to produce segment spans

#### Scenario: Config controls segmentation
- **WHEN** config has `data.segment_mode: fixed_window` and `data.segment_size: 32`
- **THEN** BagDataset uses fixed_window(32) segmentation regardless of Stage 1 settings

### Requirement: Stage 1 always uses step segmentation
`features/build_dataset.py` SHALL store `segment_spans` computed with `step` segmentation (matching the system prompt's `\n\n` format) into JSONL. This serves as the canonical segmentation for all downstream consumers. BagDataset may override this.

#### Scenario: Stage 1 writes step-based spans
- **WHEN** build_dataset runs
- **THEN** segment_spans are computed with `step` mode regardless of config `data.segment_mode`

### Requirement: All configs share Stage 1 output
All config files SHALL use identical `all_dataset`/`train_dataset`/`val_dataset`/`test_dataset` paths. No config requires its own Stage 1 re-run.

#### Scenario: arch_mlp_only uses base dataset
- **WHEN** loading `configs/arch_mlp_only.yaml`
- **THEN** `paths.train_dataset` is `datasets/train.jsonl` (same as base)
