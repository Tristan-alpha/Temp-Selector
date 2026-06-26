## ADDED Requirements

### Requirement: _normalize_parsed uses sympy object for canonicalization

`_normalize_parsed` SHALL return the `str()` representation of the sympy object (`parsed[0]`) rather than the LaTeX string (`parsed[-1]`). When the sympy object is a `Float`, it SHALL first be normalized via `sympy.nsimplify(expr, [sympy.Rational])` to convert finite decimals to rational numbers.

#### Scenario: Fraction simplification

- **WHEN** `parse` returns `[Half, r"\frac{2}{4}"]`
- **THEN** `_normalize_parsed` SHALL return `"1/2"` (from `str(Half)`)
- **AND NOT** `"\frac{2}{4}"` (from the LaTeX string)

#### Scenario: Decimal normalization

- **WHEN** `parse` returns `[Float('0.5'), "0.50"]`
- **THEN** `_normalize_parsed` SHALL return `"1/2"` (after `nsimplify`)
- **AND NOT** `"0.500000000000000"` or `"0.50"`

#### Scenario: Equivalent answers bucket together

- **WHEN** `self_consistency_correct` receives responses with `\boxed{1/2}`, `\boxed{2/4}`, `\boxed{0.5}`, and `\boxed{\frac{1}{2}}`
- **THEN** all four answers SHALL be counted in the same Counter bucket
- **AND** the mode SHALL have 4 votes
