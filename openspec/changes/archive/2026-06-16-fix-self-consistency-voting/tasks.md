## 1. Fix _normalize_parsed

- [x] 1.1 In `utils/answer_verifier.py`, modify `_normalize_parsed` to use `str(parsed[0])` instead of `str(parsed[-1])`, with `nsimplify` on `Float` types

## 2. Tests

- [x] 2.1 Add test for fraction simplification: `\boxed{2/4}` and `\boxed{1/2}` bucket together
- [x] 2.2 Add test for decimal normalization: `\boxed{0.50}` and `\boxed{1/2}` bucket together
- [x] 2.3 Add test for mixed equivalent answers in `self_consistency_correct`

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — 145 passed
- [x] 3.2 Run `python -m compileall -q utils/answer_verifier.py` on modified file
- [x] 3.3 Docs: no updates needed — this is a bug fix that aligns Counter voting with sympy's built-in canonicalization
