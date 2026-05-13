## ADDED Requirements

### Requirement: MIL early stop uses inst_logit_separation
`_validate()` in `mil/training.py` SHALL compute `inst_logit_separation` = mean(inst_logit on error bags) − mean(inst_logit on correct bags). Training stops when separation stops improving for `early_stop_patience` epochs. The best checkpoint is the one with the highest separation.

#### Scenario: Positive separation is good
- **WHEN** error bag segments have higher inst_logit than correct bag segments
- **THEN** separation > 0 and the model is considered to be improving

#### Scenario: Separation stops improving
- **WHEN** no new maximum separation is reached for `early_stop_patience` epochs
- **THEN** training terminates

### Requirement: _validate collects per-bag inst_logit means
Instead of per-instance pooling across all bags, `_validate()` SHALL compute bag-level mean inst_logit first, then average across error bags and correct bags separately.

#### Scenario: Per-bag averaging
- **WHEN** computing separation on a validation batch
- **THEN** each bag contributes equally regardless of segment count
