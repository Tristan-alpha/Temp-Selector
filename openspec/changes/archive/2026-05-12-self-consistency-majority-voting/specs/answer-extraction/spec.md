## ADDED Requirements

### Requirement: extract_answer extracts math expressions from free text
`extract_answer(text)` in `utils/answer_verifier.py` SHALL return a normalized math expression string by calling `math_verify.parse()` with `LatexExtractionConfig` and `ExprExtractionConfig`.

#### Scenario: Valid math expression
- **WHEN** `extract_answer("Therefore the answer is \\boxed{x+2}")` is called
- **THEN** a non-empty string representing the parsed expression is returned

#### Scenario: Empty or unparseable text
- **WHEN** `extract_answer("")` or `extract_answer("gibberish")` is called
- **THEN** the function returns an empty string or a string representation of None

### Requirement: verify_answer_by_value checks extracted answers
`verify_answer_by_value(prediction, gold)` in `utils/answer_verifier.py` SHALL parse both strings with `math_verify.parse()` and return the boolean result of `math_verify.verify(gold_parsed, pred_parsed)`.

#### Scenario: Matching expressions
- **WHEN** `verify_answer_by_value("x+2", "2+x")` is called
- **THEN** returns True (commutative equivalence)

#### Scenario: Non-matching expressions
- **WHEN** `verify_answer_by_value("x+3", "x+2")` is called
- **THEN** returns False

### Requirement: Existing verify_answer continues to work
The existing `verify_answer(prediction, gold)` function SHALL remain unchanged and continue to be used for per-vote `individual_correct` computation.

#### Scenario: Unused in label computation
- **WHEN** reading the label computation in build_dataset.py
- **THEN** `verify_answer` is NOT used for label determination (replaced by `extract_answer` + `verify_answer_by_value` for the mode)
