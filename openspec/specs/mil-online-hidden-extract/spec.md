## ADDED Requirements

### Requirement: BagDataset stores metadata only, no extraction in init

BagDataset SHALL store only row metadata during `__init__` (no SGLang calls, no tensor materialization). Feature extraction is deferred to the collate function, which is invoked once per training batch by the DataLoader.

#### Scenario: BagDataset init does not perform extraction

- **WHEN** `BagDataset` loads a JSONL dataset with any `feature_mode` (including `"hidden_states"` or `"all"`)
- **THEN** no SGLang engine calls are made during `__init__`
- **AND** rows are stored as lightweight dicts containing prompt, response, label, temperature, token_features (basic), and metadata

#### Scenario: BagDataset getitem returns raw metadata

- **WHEN** `BagDataset.__getitem__` is called
- **THEN** it returns the row dict as-is, without building any tensors

### Requirement: engine is provided to collate_fn, not BagDataset

The SGLangRunner SHALL be passed to the collate function (via `functools.partial`), not to `BagDataset`. `BagDataset.__init__` SHALL accept only `data_path` and no extractor parameter.

#### Scenario: Engine passed to collate_fn

- **WHEN** `train_mil()` creates a SGLangRunner
- **THEN** it passes the runner to `make_collate_fn(extractor=runner, ...)` for both train and val DataLoaders
- **AND** BagDataset does not hold a reference to the runner

### Requirement: Build stage does not emit hidden states

When `feature_mode` is `"hidden_states"` or `"all"`, `scripts/build_dataset.py` SHALL NOT write `.hidden.safetensors` sidecar files and SHALL NOT include `_hidden_offset` / `_hidden_count` keys in JSONL rows. Token features in JSONL SHALL have `"hidden": null`.

#### Scenario: Build with hidden_states mode writes JSONL only

- **WHEN** `scripts/build_dataset.py --backend sglang` runs with `feature_mode="all"`
- **THEN** output is train/val/test JSONL files only, no safetensors sidecar

## REMOVED Requirements

### Requirement: Hidden states stored in safetensors sidecar

**Reason**: Hidden state extraction moved from build stage to MIL training stage

**Migration**: No action needed. Existing `.hidden.safetensors` files can be deleted. `BagDataset` constructs segment vectors from on-demand hidden extraction.
