from __future__ import annotations

import re
import signal
from collections import Counter
from contextlib import contextmanager
from typing import Any, List, Optional

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


_TIMEOUT_SEC = 30


@contextmanager
def _time_limit(seconds: int):
    """Cross-platform timeout via SIGALRM (Unix only)."""
    if not hasattr(signal, "SIGALRM"):
        yield  # no timeout on Windows
        return
    previous = signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError))
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _parse(text: str, *, strip: bool = False) -> Any:
    return parse(text.strip() if strip else text,
                 extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])


def verify_answer(prediction: str, gold: str) -> bool:
    """Check if a full prediction text is mathematically equivalent to gold.

    Used for per-vote ``individual_correct`` statistics.
    """
    try:
        with _time_limit(_TIMEOUT_SEC):
            gold_parsed = _parse(gold, strip=True)
            pred_parsed = _parse(prediction)
            return bool(verify(gold_parsed, pred_parsed))
    except (TimeoutError, Exception):
        return False


# ═══════════════════════  answer extraction  ═══════════════════════

def _extract_brace_content(text: str, marker: str) -> List[str]:
    """Extract content inside all ``marker{...}`` groups with proper brace matching.

    Handles nested braces (e.g. ``\\boxed{\\frac{1}{2}}``).  Returns a list
    of captured contents, one per matching ``marker{...}`` occurrence.
    """
    results: List[str] = []
    i = 0
    while True:
        idx = text.find(marker, i)
        if idx == -1:
            break
        open_pos = idx + len(marker)
        if open_pos >= len(text) or text[open_pos] != "{":
            i = open_pos
            continue
        depth = 0
        start = open_pos + 1
        for j in range(open_pos, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[start:j])
                    i = j + 1
                    break
        else:
            # Unclosed brace — skip this match
            i = open_pos + 1
    return results


def _extract_last_dollar(text: str) -> Optional[str]:
    """Extract content inside the last ``$...$`` pair.

    Handles ``$$`` display math by skipping over consecutive ``$`` chars.
    Returns None if no dollar-delimited expression is found.
    """
    # Find the last $ that starts a non-empty expression
    i = len(text) - 1
    while i >= 0:
        if text[i] == "$":
            # Walk back to find the start of this $ group
            end = i
            while i >= 0 and text[i] == "$":
                i -= 1
            # Now text[i+1:end+1] is a run of $ chars
            # Search backwards for a matching run
            dollars = text[i + 1 : end + 1]
            prev = text.rfind(dollars, 0, i + 1)
            if prev != -1:
                content = text[prev + len(dollars) : i + 1]
                if content.strip():
                    return content.strip()
            continue
        i -= 1
    return None


def _extract_last_number(text: str) -> Optional[str]:
    """Extract the last numeric value from free text.

    Matches integers, decimals, and negative numbers.
    Returns the matched string or None.
    """
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        return matches[-1]
    return None


def _normalize_parsed(parsed: Any) -> str:
    """Convert a math_verify parse result into a canonical string."""
    if parsed is None or parsed == []:
        return ""
    if isinstance(parsed, list) and len(parsed) > 0:
        return str(parsed[-1])
    return str(parsed)


def extract_answer(text: str) -> str:
    """Extract a normalized math expression from LLM-generated text.

    Three-tier fallback, returning the first non-empty result:
    1. Last ``\\boxed{...}`` — the final boxed answer (handles nested braces).
    2. Last ``$...$`` or ``$$...$$`` — the final LaTeX math expression.
    3. Last number (integer or decimal) — fallback for plain-text answers.

    Every extracted answer is normalized through ``math_verify.parse()``.
    Returns an empty string if all extraction fails.
    """
    try:
        # Tier 1: last \boxed{...} with proper brace matching
        boxed = _extract_brace_content(text, r"\boxed")
        if boxed:
            parsed = _parse("$" + boxed[-1] + "$", strip=True)
            result = _normalize_parsed(parsed)
            if result:
                return result

        # Tier 2: last $...$ or $$...$$
        dollar = _extract_last_dollar(text)
        if dollar is not None:
            parsed = _parse("$" + dollar + "$", strip=True)
            result = _normalize_parsed(parsed)
            if result:
                return result

        # Tier 3: last number
        number = _extract_last_number(text)
        if number is not None:
            parsed = _parse(number, strip=True)
            result = _normalize_parsed(parsed)
            if result:
                return result

        return ""
    except Exception:
        return ""


# ═══════════════════════  verification  ═══════════════════════

def verify_answer_by_value(prediction: str, gold: str) -> bool:
    """Check if a normalized answer string is mathematically equivalent to gold.

    Both arguments are assumed to be answer expressions (not full response text).
    Automatically wraps in ``$...$`` for LaTeX parsing.
    Uses ``math_verify.parse()`` + ``math_verify.verify()``.
    """
    try:
        with _time_limit(_TIMEOUT_SEC):
            gold_parsed = _parse("$" + gold.strip() + "$", strip=True)
            pred_parsed = _parse("$" + prediction.strip() + "$", strip=True)
            return bool(verify(gold_parsed, pred_parsed))
    except (TimeoutError, Exception):
        return False


def self_consistency_correct(responses: List[str], gold: str) -> bool:
    """Self-consistency majority voting.

    Extracts the answer from each response, finds the modal (plurality)
    answer, and compares to gold using ``math_verify.verify()``.
    Returns True if the most common extracted answer matches gold.
    """
    extracted = [extract_answer(r) for r in responses]
    mode_answer = Counter(extracted).most_common(1)[0][0]
    return verify_answer_by_value(mode_answer, gold)
