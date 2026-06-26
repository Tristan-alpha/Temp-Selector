"""CPU tests for shared calibration utilities."""

from __future__ import annotations

import numpy as np
import torch

from utils.calibration import (
    answer_entropy,
    binary_nll,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    selective_risk_curve,
)


def test_perfect_calibration_zero_error():
    confidences = [0.0, 1.0, 0.0, 1.0]
    correctness = [0.0, 1.0, 0.0, 1.0]
    assert brier_score(confidences, correctness) == 0.0
    assert expected_calibration_error(confidences, correctness, n_bins=2) == 0.0
    assert binary_nll(confidences, correctness) < 1e-5


def test_overconfident_predictions_have_large_error():
    confidences = torch.tensor([0.99, 0.99, 0.99, 0.99])
    correctness = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert brier_score(confidences, correctness) > 0.45
    assert expected_calibration_error(confidences, correctness, n_bins=10) > 0.45


def test_binary_nll_clamps_extreme_probabilities():
    assert np.isfinite(binary_nll([0.0, 1.0], [1.0, 0.0]))
    assert binary_nll([0.0, 1.0], [1.0, 0.0]) > 10.0


def test_reliability_bins_include_empty_bins():
    rows = reliability_bins([0.05, 0.95], [0, 1], n_bins=4)
    assert len(rows) == 4
    assert rows[0]["count"] == 1
    assert rows[1]["count"] == 0
    assert rows[1]["mean_confidence"] is None
    assert rows[1]["accuracy"] is None
    assert rows[3]["count"] == 1


def test_numpy_and_torch_inputs_match_list_inputs():
    list_value = expected_calibration_error([0.25, 0.75], [0, 1], n_bins=2)
    numpy_value = expected_calibration_error(np.array([0.25, 0.75]), np.array([0, 1]), n_bins=2)
    torch_value = expected_calibration_error(torch.tensor([0.25, 0.75]), torch.tensor([0, 1]), n_bins=2)
    assert list_value == numpy_value == torch_value


def test_answer_entropy_counts_distribution():
    assert answer_entropy([]) == 0.0
    assert answer_entropy(["a", "a", "a"]) == 0.0
    assert abs(answer_entropy(["a", "b"]) - np.log(2.0)) < 1e-12


def test_selective_risk_curve_sorts_by_confidence():
    curve = selective_risk_curve([0.2, 0.9, 0.5], [0, 1, 0])
    assert [row["threshold"] for row in curve] == [0.9, 0.5, 0.2]
    assert curve[0]["coverage"] == 1 / 3
    assert curve[0]["risk"] == 0.0
    assert curve[-1]["coverage"] == 1.0
    assert curve[-1]["accuracy"] == 1 / 3
