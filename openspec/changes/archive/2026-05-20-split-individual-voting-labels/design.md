## Context

The current JSONL dataset has a single `label` field (0=correct, 1=error) that represents **majority voting** correctness. This creates two problems:

1. **Wrong supervision for MIL**: MIL is an instance-level model — it predicts which segments contain errors. Training it on majority-vote labels means individually-wrong responses in majority-correct questions get labeled as "no error," teaching the model to ignore real error patterns.
2. **Pervasive confusion**: Every file that touches `label` has comments warning about the "flipped" convention vs standard MIL. CLI.md, PIPELINE.md, mil/DESIGN.md, and ppo/DESIGN.md all document this gotcha.

The fix has two parts: split the single ambiguous field into two semantically-named fields, and add inline comments on every `> 0.5` branch to make the positive/negative bag semantics explicit.

## Goals / Non-Goals

**Goals:**
- MIL trains on `individual_label` (per-response correctness) — the right signal for error detection
- PPO temperature bias uses `voting_label` (majority-vote result) — the right signal for policy initialization
- Code is self-documenting: inline comments on every bag-label branch make the positive/negative semantics obvious, eliminating "flipped" warnings
- Existing pipeline stages that don't read `label` (PPO training/eval) are unaffected

**Non-Goals:**
- Changing the 0/1 encoding convention (0=correct, 1=error stays)
- Modifying PPO reward computation (PPO already computes correctness at runtime)
- Migrating existing JSONL files (just regenerate with updated build_dataset.py)

## Decisions

### Decision 1: Two separate fields, not one renamed field

**Chosen**: `individual_label` and `voting_label` as top-level JSONL keys.

**Alternatives considered**:
- *Rename `label` → `voting_label`, keep `individual_correct` in metadata* → Rejected. Burying the MIL-critical field in metadata makes it easy to overlook. Top-level fields signal importance.
- *One `label` for MIL + `voting_label` for PPO* → Rejected. "label" is too generic. The split forces every consumer to choose the right field.

### Decision 2: Keep 0=correct, 1=error encoding

**Chosen**: Both fields use the same encoding: 0=correct, 1=error.

**Alternatives considered**:
- *Flip to 1=correct to match `ep_correct`* → Rejected. Would require flipping every MIL `> 0.5` check and risk subtle bugs. Inline comments on each branch make the convention explicit at the code level without touching the encoding.
- *Use bool (True/False)* → Rejected. JSON doesn't have native bool type in all parsers, and `float` conversion in collate_fn works naturally with 0/1.

### Decision 3: Inline comments instead of a named constant

**Chosen**: Keep `> 0.5` for float robustness, add inline comments at every branch site that explicitly name the positive/negative bag semantics (e.g., `# label=1: positive bag (contains errors)`). No module-level constant.

**Alternatives considered**:
- *Named constant `POSITIVE_BAG_VALUE = 1` with `==` comparison* → Rejected. `==` on a float is fragile; collate_fn produces float tensors from `row.get("individual_label", 0)`. A stray `1.0` vs `1` mismatch would silently break.
- *Config key `mil.positive_bag_value`* → Rejected. Not a tunable hyperparameter.

### Decision 4: No backward compatibility shim

**Chosen**: Old JSONL files with `label` field are not supported. Run `scripts/build_dataset.py` to regenerate.

**Alternatives considered**:
- *Auto-detect old format and map `label` → `voting_label` (with warning)* → Rejected. The semantic difference (majority vs individual) means old data silently mis-trains MIL. Better to fail loudly with a missing key error.

### Decision 5: load_temperature_labels reads voting_label, still flips

**Chosen**: `load_temperature_labels` reads `voting_label` instead of `label`, and still returns `1 - label` (1=correct) for PPO consumers.

**Alternatives considered**:
- *Change PPO consumers to accept 0=correct* → Rejected. `ep_correct` uses 1=correct, and all PPO accuracy computations use `ep_correct`. Keeping the flip in `load_temperature_labels` isolates the encoding difference to one function.

## Risks / Trade-offs

- **[Risk] MIL eval metrics change meaning**: `compute_bag_metrics` currently measures "can MIL predict majority-vote errors?" With `individual_label`, it becomes "can MIL predict individual response errors?" — a harder but more meaningful task. Reported accuracy may drop.
  → **Mitigation**: Document the metric change in commit and release notes. The new metric is more honest.

- **[Risk] Existing checkpoints rely on old label semantics**: MIL models trained with old `label` field learned to predict majority-vote errors. They're still usable but less precise.
  → **Mitigation**: Checkpoints are cheap to regenerate (~5 min on single GPU). No compatibility shim needed.

- **[Trade-off] Two fields instead of one**: Slightly larger JSONL (one extra int per row).
  → Negligible. Typical dataset is ~50K rows, adding ~50KB total.

## Migration Plan

1. Update `scripts/build_dataset.py` to write both new fields
2. Update all consumers (MIL, PPO, eval) to read the correct field
3. Update docs to remove "flipped" warnings
4. Regenerate datasets with updated build script
5. No rollback needed for code changes (old JSONL files simply fail with KeyError on missing fields)
