from __future__ import annotations


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0
