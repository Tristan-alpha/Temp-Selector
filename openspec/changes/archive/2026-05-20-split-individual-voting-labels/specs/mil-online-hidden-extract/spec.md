## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: BagDataset stores metadata only, no extraction in init

BagDataset SHALL store only row metadata during `__init__` (no SGLang calls, no tensor materialization). Feature extraction is deferred to the collate function, which is invoked once per training batch by the DataLoader.

#### Scenario: BagDataset init does not perform extraction

- **WHEN** `BagDataset` loads a JSONL dataset with any `feature_mode` (including `"hidden_states"` or `"all"`)
- **THEN** no SGLang engine calls are made during `__init__`
- **AND** rows are stored as lightweight dicts containing prompt, response, individual_label, temperature, token_features (basic), and metadata

#### Scenario: BagDataset getitem returns raw metadata

- **WHEN** `BagDataset.__getitem__` is called
- **THEN** it returns the row dict as-is, without building any tensors
