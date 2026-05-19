## ADDED Requirements

### Requirement: make_collate_fn reads individual_label for bag labels

`make_collate_fn` SHALL construct the `label` tensor in its return dict from the `individual_label` field of each dataset row. The tensor key `"label"` in the return dict is unchanged (internal name), but the source field in the row dict SHALL be `individual_label`.

#### Scenario: Collate function label tensor construction

- **WHEN** `make_collate_fn` builds a batch dict
- **THEN** `batch["label"]` SHALL be a float tensor constructed from `row["individual_label"]` values
- **AND** missing `individual_label` SHALL default to `0.0`
