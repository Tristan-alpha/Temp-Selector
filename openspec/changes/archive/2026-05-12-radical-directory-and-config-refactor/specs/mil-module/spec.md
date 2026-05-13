## ADDED Requirements

### Requirement: MIL module is self-contained under mil/
The MIL model definition (MILModel and auxiliary temperature heads), training loop, evaluation logic, and evaluation metric functions SHALL reside under a single `mil/` top-level directory.

#### Scenario: All MIL symbols are importable from mil.model
- **WHEN** code executes `from mil.model import MILModel, InstanceEncoder, SinusoidalPositionalEncoding, AttentionAggregator, GlobalTempHead, DynamicTempHead, smoothness_loss`
- **THEN** all seven symbols are successfully imported

#### Scenario: MIL training is importable from mil.training
- **WHEN** code executes `from mil.training import BagDataset, RowTensor, collate_rows, train_mil`
- **THEN** all four symbols are successfully imported

#### Scenario: MIL eval and metrics are importable from mil.eval
- **WHEN** code executes `from mil.eval import evaluate_mil, compute_bag_metrics, compute_calibration, compute_multiclass_metrics, compute_auc, compute_attention_metrics`
- **THEN** all six symbols are successfully imported

#### Scenario: No separate temp_predictor file exists
- **WHEN** checking `mil/temp_predictor.py`
- **THEN** the file does not exist (content merged into `mil/model.py`)

#### Scenario: MIL training is runnable as a module
- **WHEN** user executes `python -m mil.training --config configs/base.yaml`
- **THEN** the MIL training loop starts without import errors

### Requirement: MIL module has no DDP dependency
`mil/training.py` SHALL NOT import or use `DistributedDataParallel`, `DistributedSampler`, or any `torch.distributed` primitives. Device selection SHALL be `torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")`.

#### Scenario: No DDP imports in mil/
- **WHEN** grep for `DistributedDataParallel\|DistributedSampler\|distributed\|setup_distributed\|barrier\|cleanup_distributed` in `mil/`
- **THEN** no matches are found

### Requirement: MIL module has no dependency on legacy paths
None of the files under `mil/` SHALL import from `models.*`, `training.*`, `rl.*`, or `eval.*` paths.

#### Scenario: All mil/ imports use current paths
- **WHEN** grep for `from models\.` or `from training\.` or `from rl\.` or `from eval\.` in `mil/`
- **THEN** no matches are found

### Requirement: MIL uses shared vectorizer utilities
`mil/training.py` SHALL import `token_to_vec` from `features.vectorizer` and SHALL NOT contain its own copy.

#### Scenario: No duplicate token_to_vec in mil/
- **WHEN** searching for `def token_to_vec` in `mil/`
- **THEN** no definition is found

### Requirement: MIL eval uses compute_attention_metrics
`mil/eval.py` SHALL call the `compute_attention_metrics` function for attention analysis instead of computing the same metrics inline.

#### Scenario: Attention metrics are computed via function call
- **WHEN** reading `mil/eval.py`
- **THEN** `compute_attention_metrics` is called (not duplicated inline)
