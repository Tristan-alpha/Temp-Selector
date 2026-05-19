## ADDED Requirements

### Requirement: BagDataset stores metadata only, no extraction in init

BagDataset SHALL store only row metadata during `__init__` (no SGLang calls, no tensor materialization). Feature extraction is deferred to the collate function, which is invoked once per training batch by the DataLoader.

#### Scenario: BagDataset init does not perform extraction

- **WHEN** `BagDataset` loads a JSONL dataset with any `feature_mode` (including `"hidden_states"` or `"all"`)
- **THEN** no SGLang engine calls are made during `__init__`
- **AND** rows are stored as lightweight dicts containing prompt, response, individual_label, temperature, token_features (basic), and metadata

#### Scenario: BagDataset getitem returns raw metadata

- **WHEN** `BagDataset.__getitem__` is called
- **THEN** it returns the row dict as-is, without building any tensors

### Requirement: engine is provided to collate_fn, not BagDataset

The VLLMFeatureExporter SHALL be passed to `make_collate_fn` during pre-computation (via `functools.partial`). During training epochs, `make_cached_collate_fn` SHALL NOT receive an extractor. `BagDataset.__init__` SHALL accept only `data_path` and no extractor parameter.

#### Scenario: Engine passed to collate_fn during pre-computation

- **WHEN** `train_mil()` pre-computes segment features
- **THEN** it SHALL use `make_collate_fn(extractor=runner, ...)` for the pre-computation pass
- **AND** it SHALL switch to `make_cached_collate_fn(cache, ...)` for training epochs

### Requirement: Build stage does not emit hidden states

When `feature_mode` is `"hidden_states"` or `"all"`, `scripts/build_dataset.py` SHALL NOT write `.hidden.safetensors` sidecar files and SHALL NOT include `_hidden_offset` / `_hidden_count` keys in JSONL rows. Token features in JSONL SHALL have `"hidden": null`.

#### Scenario: Build with hidden_states mode writes JSONL only

- **WHEN** `scripts/build_dataset.py --backend sglang` runs with `feature_mode="all"`
- **THEN** output is train/val/test JSONL files only, no safetensors sidecar

### Requirement: MIL bag-label branches have explicit inline comments

MIL training (`mil/training.py`) and evaluation (`mil/eval.py`) SHALL annotate every `> 0.5` branch on bag labels with an inline comment that names the positive/negative bag semantics. The threshold check `> 0.5` SHALL be kept (not replaced with `==`) because labels are float tensors from collate_fn.

#### Scenario: Positive bag branch is explicitly commented

- **WHEN** branching on `y[i].item() > 0.5` for an error bag
- **THEN** the branch SHALL carry a comment such as `# label=1: positive bag (contains errors)`

#### Scenario: Negative bag branch is explicitly commented

- **WHEN** branching on `else` for a correct bag
- **THEN** the branch SHALL carry a comment such as `# label=0: negative bag (no errors)`

### Requirement: MIL collate_fn reads individual_label

`make_collate_fn` in `mil/utils.py` SHALL read the `individual_label` field from dataset rows to construct the `y` label tensor. The default value for missing keys SHALL remain `0.0`.

#### Scenario: Collate reads individual_label

- **WHEN** `make_collate_fn` processes a batch of rows
- **THEN** it SHALL extract `float(row["individual_label"])` for each row's label value

### Requirement: Feature extraction is pre-compute once, not per-batch

MIL training SHALL call `extract_from_ids` only during the pre-computation phase (once before epoch loop). Training epochs SHALL use the RAM-cached segment features. The pre-computation phase SHALL still use `make_collate_fn` with the extractor; training epochs SHALL use `make_cached_collate_fn`.

#### Scenario: No vLLM calls during training epochs

- **WHEN** MIL training enters the epoch loop
- **THEN** no call to `extract_from_ids` SHALL occur during collation
- **AND** all segment tensors SHALL come from the pre-computed cache

## REMOVED Requirements

### Requirement: Hidden states stored in safetensors sidecar

**Reason**: Hidden state extraction moved from build stage to MIL training stage

**Migration**: No action needed. Existing `.hidden.safetensors` files can be deleted. `BagDataset` constructs segment vectors from on-demand hidden extraction.
