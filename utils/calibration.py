"""Calibration and confidence utilities shared by evaluation scripts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, List, Sequence

import numpy as np


def _to_numpy_1d(values: Any, name: str) -> np.ndarray:
    """Convert list, numpy, or torch-like values to a 1-D float array."""
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _paired_arrays(confidences: Any, correctness: Any) -> tuple[np.ndarray, np.ndarray]:
    conf = _to_numpy_1d(confidences, "confidences")
    corr = _to_numpy_1d(correctness, "correctness")
    if conf.shape != corr.shape:
        raise ValueError("confidences and correctness must have the same shape")
    return np.clip(conf, 0.0, 1.0), np.clip(corr, 0.0, 1.0)


def brier_score(confidences: Any, correctness: Any) -> float:
    """Mean squared error between confidence and binary/soft correctness."""
    conf, corr = _paired_arrays(confidences, correctness)
    if conf.size == 0:
        return 0.0
    return float(np.mean((conf - corr) ** 2))


def binary_nll(confidences: Any, correctness: Any, eps: float = 1e-6) -> float:
    """Mean binary negative log likelihood for confidence/correctness pairs."""
    conf, corr = _paired_arrays(confidences, correctness)
    if conf.size == 0:
        return 0.0
    p = np.clip(conf, eps, 1.0 - eps)
    return float(-np.mean(corr * np.log(p) + (1.0 - corr) * np.log1p(-p)))


def reliability_bins(confidences: Any, correctness: Any,
                     n_bins: int = 10) -> List[dict[str, float | int | None]]:
    """Return confidence/accuracy/gap statistics for fixed-width bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    conf, corr = _paired_arrays(confidences, correctness)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[dict[str, float | int | None]] = []
    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == 0:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf > left) & (conf <= right)
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin_left": left,
                "bin_right": right,
                "count": 0,
                "mean_confidence": None,
                "accuracy": None,
                "gap": None,
            })
            continue
        mean_conf = float(conf[mask].mean())
        accuracy = float(corr[mask].mean())
        rows.append({
            "bin_left": left,
            "bin_right": right,
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "gap": abs(mean_conf - accuracy),
        })
    return rows


def expected_calibration_error(confidences: Any, correctness: Any,
                               n_bins: int = 10) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, _corr = _paired_arrays(confidences, correctness)
    if conf.size == 0:
        return 0.0
    total = float(conf.size)
    ece = 0.0
    for row in reliability_bins(confidences, correctness, n_bins=n_bins):
        if row["count"]:
            ece += (int(row["count"]) / total) * float(row["gap"])
    return float(ece)


def answer_entropy(answers: Sequence[Any] | Iterable[Any]) -> float:
    """Shannon entropy of an answer distribution, in nats."""
    values = list(answers)
    if not values:
        return 0.0
    counts = Counter(str(value) for value in values)
    total = float(len(values))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)


def selective_risk_curve(confidences: Any, correctness: Any) -> List[dict[str, float]]:
    """Risk/coverage curve after sorting examples by descending confidence."""
    conf, corr = _paired_arrays(confidences, correctness)
    if conf.size == 0:
        return []
    order = np.argsort(-conf, kind="mergesort")
    sorted_conf = conf[order]
    sorted_corr = corr[order]
    cumulative_correct = np.cumsum(sorted_corr)
    rows: List[dict[str, float]] = []
    n = float(conf.size)
    for idx in range(conf.size):
        covered = idx + 1
        accuracy = float(cumulative_correct[idx] / covered)
        rows.append({
            "coverage": float(covered / n),
            "risk": float(1.0 - accuracy),
            "accuracy": accuracy,
            "threshold": float(sorted_conf[idx]),
            "count": float(covered),
        })
    return rows
