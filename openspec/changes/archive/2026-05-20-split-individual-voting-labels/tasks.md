## 1. Data generation: build_dataset.py

- [x] 1.1 Write `individual_label` (int, 0/1) computed from individual response correctness; promote existing `metadata.individual_correct` logic to this top-level field
- [x] 1.2 Write `voting_label` (int, 0/1) computed from majority-vote result (what `label` currently holds)
- [x] 1.3 Remove the old `label` field from sample dicts
- [x] 1.4 Update `n_positive`/`n_individual_correct` counters and log messages to reflect the new fields

## 2. MIL data utilities: mil/utils.py

- [x] 2.1 Change `make_collate_fn` to read `row.get("individual_label", 0)` instead of `row.get("label", 0)` at both call sites

## 3. MIL training: mil/training.py

- [x] 3.1 Change pos_weight computation to read `individual_label` (line 126)
- [x] 3.2 Add inline comments at `y[i].item() > 0.5` branches: `# label=1: positive bag (contains errors)` and `else: # label=0: negative bag (no errors)` (lines 241, 291)
- [x] 3.3 Add inline comments at `y_v[i].item() > 0.5` branch in validation separation logic (line 191)

## 4. MIL evaluation: mil/eval.py

- [x] 4.1 Add inline comments at `y[i].item() > 0.5` branches in per-segment analysis (lines 277-280)
- [x] 4.2 Add inline comment at `error_mask` construction (lines 345-348)
- [x] 4.3 Update bag label references in `compute_bag_metrics` if they reference the old field name

## 5. Dataset evaluation: features/dataset_eval.py

- [x] 5.1 Update `evaluate_dataset`: read `voting_label` for majority-vote accuracy stats, read `individual_label` for individual accuracy stats; report both
- [x] 5.2 Update `load_temperature_labels`: read `voting_label` instead of `label`; keep the `1 - label` flip for PPO consumers

## 6. Documentation cleanup

- [x] 6.1 Remove "flipped label" pitfall from CLAUDE.md (line 60); add entry about `individual_label` vs `voting_label` distinction
- [x] 6.2 Update mil/DESIGN.md: change label semantics table to reference `individual_label`; remove "flipped" framing
- [x] 6.3 Update ppo/DESIGN.md: update `ep_correct vs MIL label` table to reference `voting_label` for PPO and `individual_label` for MIL
- [x] 6.4 Update PIPELINE.md: update data flow diagram and label documentation
- [x] 6.5 Update README.md: update label convention documentation

## 7. Verification

- [x] 7.1 Run `python -m pytest tests/ -v` — all tests must pass; update test fixtures that reference `label` field to use `individual_label` or `voting_label` as appropriate
- [x] 7.2 Run `python -m compileall -q scripts/build_dataset.py mil/utils.py mil/training.py mil/eval.py features/dataset_eval.py` to catch syntax errors
- [x] 7.3 Verify docs are consistent: no remaining references to the old single `label` field or "flipped convention" in CLAUDE.md, PIPELINE.md, README.md, mil/DESIGN.md, ppo/DESIGN.md
