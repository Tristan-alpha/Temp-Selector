"""Tests for answer extraction and self-consistency.  CPU-only."""

from utils.answer_verifier import (
    _extract_brace_content,
    _extract_last_dollar,
    _extract_last_number,
    extract_answer,
    extract_final_answer,
    verify_answer,
    verify_answer_by_value,
    self_consistency_correct,
)


# ═══════════════════════  unit: _extract_brace_content  ═══════════════════════

def test_brace_simple():
    assert _extract_brace_content(r"\boxed{42}", r"\boxed") == ["42"]


def test_brace_multiple():
    assert _extract_brace_content(r"\boxed{7} and \boxed{42}", r"\boxed") == ["7", "42"]


def test_brace_nested_frac():
    """Nested braces: \\boxed{\\frac{1}{2}} should capture \\frac{1}{2}."""
    result = _extract_brace_content(r"\boxed{\frac{1}{2}}", r"\boxed")
    assert result == [r"\frac{1}{2}"]


def test_brace_nested_multiple():
    """Multiple boxed with nested braces."""
    result = _extract_brace_content(
        r"\boxed{x+2} then \boxed{\frac{1}{2}}", r"\boxed"
    )
    assert result == ["x+2", r"\frac{1}{2}"]


def test_brace_deeply_nested():
    """Deeply nested: \\boxed{a + \\frac{b + \\frac{c}{d}}{e}}."""
    result = _extract_brace_content(
        r"\boxed{a + \frac{b + \frac{c}{d}}{e}}", r"\boxed"
    )
    assert len(result) == 1
    assert r"\frac{b + \frac{c}{d}}{e}" in result[0]


def test_brace_no_match():
    assert _extract_brace_content("no boxed here", r"\boxed") == []


def test_brace_unclosed():
    """Unclosed brace should not match."""
    assert _extract_brace_content(r"\boxed{unclosed", r"\boxed") == []


# ═══════════════════════  unit: _extract_last_dollar  ═══════════════════════

def test_dollar_single():
    assert _extract_last_dollar("solve $x+2$ please") == "x+2"


def test_dollar_multiple():
    """Last $...$ should win."""
    assert _extract_last_dollar("first $7$ then $42$") == "42"


def test_dollar_display_math():
    """$$...$$ display math."""
    result = _extract_last_dollar(r"the answer is $$\frac{1}{2}$$ done")
    assert r"\frac{1}{2}" in result


def test_dollar_no_match():
    assert _extract_last_dollar("no math here") is None


# ═══════════════════════  unit: _extract_last_number  ═══════════════════════

def test_number_integer():
    assert _extract_last_number("the answer is 42") == "42"


def test_number_negative():
    assert _extract_last_number("result is -3.5 today") == "-3.5"


def test_number_decimal():
    assert _extract_last_number("total: .5 and 3.14") == "3.14"


def test_number_multiple():
    """Last number wins."""
    assert _extract_last_number("x=7, y=42") == "42"


def test_number_no_match():
    assert _extract_last_number("no numbers here") is None


# ═══════════════════════  unit: extract_answer tiers  ═══════════════════════

def test_extract_tier1_boxed_nested_frac():
    """Tier 1: last \\boxed{} with nested braces."""
    ans = extract_answer(r"\boxed{7} then \boxed{\frac{1}{2}}")
    assert ans == r"\frac{1}{2}"


def test_extract_tier1_boxed_simple():
    ans = extract_answer(r"the answer is \boxed{x+2}")
    assert ans == "x+2"


def test_extract_tier2_dollar():
    """Tier 2: fallback to $...$ when no \\boxed{}."""
    ans = extract_answer("solve $x+2$ please")
    assert ans == "x+2"


def test_extract_tier2_dollar_display():
    """Tier 2: $$...$$ display math."""
    ans = extract_answer(r"therefore $$\frac{1}{2}$$ is the answer")
    assert ans == r"\frac{1}{2}"


def test_extract_tier3_number():
    """Tier 3: fallback to last number when no LaTeX at all."""
    ans = extract_answer("the answer is 42")
    assert ans == "42"


def test_extract_boxed_over_dollar():
    """\\boxed{} has priority over $...$."""
    ans = extract_answer(r"intermediate $7$ and \boxed{42}")
    assert ans == "42"


def test_extract_dollar_over_number():
    """$...$ has priority over bare number."""
    ans = extract_answer("the number 7 but $x+2$ gives answer")
    assert ans == "x+2"


def test_extract_empty_text():
    assert extract_answer("") == ""


def test_extract_unparseable():
    assert extract_answer("gibberish with no structure") == ""


# ═══════════════════════  unit: verify_answer_by_value  ═══════════════════════

def test_vav_match_simple():
    assert verify_answer_by_value("3", "3") is True


def test_vav_mismatch():
    assert verify_answer_by_value("5", "3") is False


def test_vav_frac_dfrac_equivalent():
    """\\frac{1}{2} and \\dfrac{1}{2} should be equivalent."""
    assert verify_answer_by_value(r"\frac{1}{2}", r"\dfrac{1}{2}") is True


def test_vav_frac_slash_equivalent():
    """\\frac{1}{2} and 1/2 should be equivalent."""
    assert verify_answer_by_value(r"\frac{1}{2}", r"1/2") is True


def test_vav_algebra_commutative():
    """x+2 and 2+x should be equivalent."""
    assert verify_answer_by_value("x+2", "2+x") is True


# ═══════════════════════  integration: self_consistency_correct  ══════════════

def test_sc_mode_matches_gold_boxed():
    """Mode \\boxed{3} matches gold 3."""
    assert self_consistency_correct(
        [r"\boxed{3}", r"\boxed{3}", r"\boxed{5}", r"\boxed{3}"], "3"
    ) is True


def test_sc_mode_differs_from_gold():
    """Mode \\boxed{5} does not match gold 3."""
    assert self_consistency_correct(
        [r"\boxed{5}", r"\boxed{5}", r"\boxed{3}", r"\boxed{5}"], "3"
    ) is False


def test_sc_plain_text_numbers():
    """Tier 3 extraction: plain numbers."""
    assert self_consistency_correct(
        ["The answer is 7", "I got 7", "7", "Maybe 5"], "7"
    ) is True


def test_sc_mixed_boxed_and_plain():
    """Some responses have \\boxed{}, some don't — extraction is per-response."""
    assert self_consistency_correct(
        [r"\boxed{7}", r"\boxed{7}", "answer is 7", "I think 5"], "7"
    ) is True


def test_sc_or_trick():
    """Responses with one value being "none of the above" (common edge case)."""
    assert self_consistency_correct(
        [r"\boxed{42}", r"\boxed{42}", r"\boxed{0}", r"\boxed{42}"], "42"
    ) is True


def test_sc_all_empty_extraction():
    """All responses unparseable — mode is empty string, won't match gold."""
    assert self_consistency_correct(
        ["gibberish", "nonsense", "blah"], "3"
    ) is False


# ═══════════════════════  unit: verify_answer  ═══════════════════════

def test_verify_answer_last_boxed():
    """Only the last \\boxed{...} is used; no fallback."""
    assert verify_answer(
        r"First \boxed{wrong} then \boxed{42}", "42") is True


def test_verify_answer_left_right_equivalent():
    """\\boxed{(3, \\frac{\\pi}{2})} matches \\left( 3, \\frac{\\pi}{2} \\right)."""
    gold = r"\left( 3, \frac{\pi}{2} \right)"
    pred = r"The answer is \boxed{(3, \frac{\pi}{2})}."
    assert verify_answer(pred, gold) is True


def test_verify_answer_no_boxed():
    """No \\boxed{} at all → False."""
    assert verify_answer("The answer is 42.", "42") is False


def test_verify_answer_frac_equivalent():
    """\\frac{1}{2} vs 1/2."""
    assert verify_answer(r"\boxed{\frac{1}{2}}", "1/2") is True


# ═══════════════════════  unit: extract_final_answer  ═══════════════════════

def test_extract_final_answer_simple():
    assert extract_final_answer(r"\boxed{42}") == "42"


def test_extract_final_answer_last_wins():
    assert extract_final_answer(r"\boxed{7} and \boxed{42}") == "42"


def test_extract_final_answer_no_boxed():
    assert extract_final_answer("no answer here") is None
