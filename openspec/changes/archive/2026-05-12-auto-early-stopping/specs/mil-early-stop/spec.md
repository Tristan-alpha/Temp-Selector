## ADDED Requirements

### Requirement: MIL training stops early on validation plateau
`mil/training.py` SHALL evaluate `bag_accuracy` on `paths.eval_dataset` after each epoch. If no improvement exceeds the previous best for `patience` consecutive epochs, training stops. The checkpoint from the epoch with the highest validation `bag_accuracy` SHALL be saved (not the final epoch).

#### Scenario: Early stop triggers
- **WHEN** `bag_accuracy` has not improved for `early_stop_patience` consecutive epochs
- **THEN** training terminates and the best-epoch checkpoint is retained on disk

#### Scenario: Continues while improving
- **WHEN** a new best `bag_accuracy` is reached
- **THEN** the patience counter resets to 0 and training continues

### Requirement: MIL config uses max_epochs and early_stop_patience
`mil.training` SHALL contain `max_epochs` (upper bound, default 50) and `early_stop_patience` (default 5). The old `epochs` key SHALL NOT exist.

#### Scenario: Config keys present
- **WHEN** loading `configs/base.yaml`
- **THEN** `mil.training.max_epochs` is 50 and `mil.training.early_stop_patience` is 5

#### Scenario: Old key absent
- **WHEN** loading `configs/base.yaml`
- **THEN** `mil.training.get("epochs")` returns None

### Requirement: Validation uses the same BagDataset as training
The validation loader SHALL use `BagDataset` with the same `temp_bins`, `instance_dim`, `pooling_mode`, and `segment_size` as training, reading from `paths.eval_dataset`.

#### Scenario: Validation data loaded correctly
- **WHEN** the training loop starts
- **THEN** an eval DataLoader is created from `paths.eval_dataset`
