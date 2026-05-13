## 1. Add extraction utilities

- [x] 1.1 Add `extract_answer(text)` to `utils/answer_verifier.py` using `math_verify.parse()` with `LatexExtractionConfig` and `ExprExtractionConfig`
- [x] 1.2 Add `verify_answer_by_value(prediction, gold)` to `utils/answer_verifier.py` — thin wrapper around `parse` + `verify`
- [x] 1.3 Add `self_consistency_correct(responses, gold)` to `utils/answer_verifier.py` — extract all answers, find mode via `Counter.most_common(1)`, compare mode to gold

## 2. Replace label computation (Stage 1)

- [x] 2.1 In `features/build_dataset.py` vLLM path (line ~142-148): replace per-vote `verify_answer` counting with `self_consistency_correct()`. Keep `metadata["individual_correct"]` unchanged (still per-vote `verify_answer`)
- [x] 2.2 In `features/build_dataset.py` API path (line ~210-216): same change as 2.1

## 3. Replace PPO terminal reward (Stage 3)

- [x] 3.1 In `ppo/training.py` EOS terminal check (line ~287-294): replace per-vote counting with `self_consistency_correct()`
- [x] 3.2 In `ppo/training.py` fallback terminal check (line ~303-310): same change as 3.1

## 4. Replace online evaluation accuracy (Stage 4)

- [x] 4.1 In `ppo/eval.py` `_evaluate_strategy_batch` (line ~221-228): replace per-vote counting with `self_consistency_correct()`. Keep `result.errors` counter (it counts verify_answer exceptions — unchanged)

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all 80 tests pass
- [x] 5.2 Verify `self_consistency_correct(["The answer is 3", "Final: 3", "x=3.", "I think 5"], "3")` returns True (mode="3" matches gold)

## 6. Documentation

- [x] 6.1 Update PIPELINE.md majority voting section: describe self-consistency extraction + mode comparison
- [x] 6.2 Update features/DESIGN.md majority voting section: replace "count of correct votes" with self-consistency description
- [x] 6.3 Update ppo/DESIGN.md terminal reward section: reflect self-consistency logic
