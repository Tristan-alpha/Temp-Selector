## Context

`features/dataset_eval.py` has `evaluate_dataset(data_path)` which analyzes a single JSONL file and returns stats. `main()` currently reads `test_dataset` from config (or `--data` override) and calls it once. The user wants to see all three splits at once.

## Goals / Non-Goals

**Goals:**
- Default behavior: analyze train, val, test in order
- Output: one JSON per split (`eval_stats_{split}.json`)
- One output JSON per split

**Non-Goals:**
- Changing `evaluate_dataset()` function signature or logic
- Adding cross-split comparison metrics

## Decisions

### Decision 1: Loop in main(), don't touch evaluate_dataset()

**Chosen**: `main()` loops over `[("train", paths["train_dataset"]), ("val", paths["val_dataset"]), ("test", paths["test_dataset"])]`. The old `--data` flag is removed — all three splits are always evaluated.

**Alternatives**:
- *Modify evaluate_dataset to accept multiple paths* → Rejected. Unnecessary coupling.

### Decision 2: One output file per split

**Chosen**: `datasets/eval_stats_{split}.json` for each split.

**Alternatives**:
- *Single combined JSON* → Rejected. Harder to diff between splits.

## Risks

None. Pure main() refactor, `evaluate_dataset()` is unchanged.
