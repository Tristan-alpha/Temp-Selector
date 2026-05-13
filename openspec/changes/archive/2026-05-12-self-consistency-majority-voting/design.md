## Context

The current majority voting implementation counts per-vote correctness checks (`verify_answer` against gold) and uses a threshold. True self-consistency extracts the answer from each response, finds the mode, and compares the mode to gold. This affects label generation (Stage 1), terminal reward (Stage 3), and online evaluation accuracy (Stage 4). The `individual_correct` statistic is intentionally left unchanged as a per-vote reference metric.

## Goals / Non-Goals

**Goals:**
- Label, terminal reward, and online evaluation accuracy all use true self-consistency
- Per-vote `individual_correct` statistic preserved as auxiliary reference

**Non-Goals:**
- Changing `dataset_eval.py` statistics
- Changing the `verify_answer` function itself
- Retraining existing models (behavior change is in label generation and evaluation)
- Handling ties differently than `Counter.most_common(1)`

## Decisions

### D1: Extraction via `math_verify.parse()`

The existing `math_verify` library already handles extraction. `parse(text, [LatexExtractionConfig(), ExprExtractionConfig()])` extracts LaTeX and plain expressions from free text. We reuse this rather than adding a new extraction dependency.

### D2: Mode from `collections.Counter`

```python
from collections import Counter
mode_answer = Counter(extracted_answers).most_common(1)[0][0]
```

`most_common(1)` returns `[(answer, count)]`. In case of a tie, it returns the first-encountered answer (deterministic given insertion order). This is standard self-consistency behavior — ties are broken arbitrarily.

### D3: Unparseable responses

If `extract_answer()` returns an empty string for some responses, those empty strings participate in counting. If the mode is an empty string (all responses unparseable), `verify_answer_by_value("", gold)` will return False, and the bag is labeled as error. This is a reasonable default — an unparseable response cannot match any correct answer.

### D4: Keep `individual_correct` as-is

`metadata["individual_correct"]` stays as `verify_answer(response, gold)` — a per-vote binary correctness check. This metric answers "what fraction of individual completions were correct at all?" which is useful for temperature calibration analysis independent of self-consistency.

### D5: Where to put the helper

A new function `self_consistency_correct(responses, gold)` encapsulates the "extract → mode → compare" logic in one callable, reducing code duplication across the 3 call sites.

```python
def self_consistency_correct(responses: List[str], gold: str) -> bool:
    """Return True if the modal extracted answer matches gold."""
    extracted = [extract_answer(r) for r in responses]
    mode_answer = Counter(extracted).most_common(1)[0][0]
    return verify_answer_by_value(mode_answer, gold)
```

## Risks / Trade-offs

- **[Label shift]** Changing label semantics means re-running Stage 1 produces different labels → existing MIL checkpoints are trained on the old label scheme → **Mitigation**: document this clearly; checkpoint filenames should indicate which label scheme was used if both coexist
- **[Tie breaking]** `Counter.most_common(1)` breaks ties arbitrarily (first encountered wins) → **Mitigation**: ties are rare in practice (8 votes with continuous answers); tie behavior is consistent with literature

## Migration Plan

1. Add `extract_answer()`, `verify_answer_by_value()`, `self_consistency_correct()` to `utils/answer_verifier.py`
2. Replace label computation in `features/build_dataset.py` (both vLLM and API paths)
3. Replace terminal reward in `ppo/training.py`
4. Replace accuracy in `ppo/eval.py`
5. Run tests, verify no regressions
6. Update PIPELINE.md, features/DESIGN.md, ppo/DESIGN.md
