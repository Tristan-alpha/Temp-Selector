## MODIFIED Requirements

### Requirement: BagDataset extracts hidden states on init via engine prefill

BagDataset SHALL store only row metadata during `__init__` (no SGLang calls, no tensor materialization). Feature extraction is deferred to the collate function, which is invoked once per training batch by the DataLoader.

#### Scenario: BagDataset init does not perform extraction

- **WHEN** `BagDataset` loads a JSONL dataset with any `feature_mode` (including `"hidden_states"` or `"all"`)
- **THEN** no SGLang engine calls are made during `__init__`
- **AND** rows are stored as lightweight dicts containing prompt, response, label, temperature, token_features (basic), and metadata

#### Scenario: BagDataset getitem returns raw metadata

- **WHEN** `BagDataset.__getitem__` is called
- **THEN** it returns the row dict as-is, without building any tensors

### Requirement: engine is provided to collate_fn, not BagDataset

The SGLangRunner SHALL be passed to the collate function (via `functools.partial`), not to `BagDataset`. `BagDataset.__init__` SHALL accept an optional `feature_mode` string but no extractor parameter.

#### Scenario: Engine passed to collate_fn

- **WHEN** `train_mil()` creates a SGLangRunner
- **THEN** it passes the runner to `partial(collate_fn, extractor=runner, ...)` for both train and val DataLoaders
- **AND** BagDataset does not hold a reference to the runner
