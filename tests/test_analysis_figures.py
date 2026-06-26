"""Tests for paper analysis figure helpers."""

from __future__ import annotations

import math

import pytest
import torch

from scripts.plot_paper_analysis_figures import (
    assign_tertiles,
    group_temperature_votes,
    masked_segment_entropy,
    majority_summary_for_votes,
    vote_margin_from_answers,
)


def _row(sample_id: str, temp: float, vote: int, answer: str,
         correct: bool = True, voting_label: int = 0) -> dict:
    return {
        "sample_id": f"{sample_id}_t{temp}_v{vote}",
        "temperature": temp,
        "individual_label": 0 if correct else 1,
        "voting_label": voting_label,
        "token_ids": list(range(vote + 1)),
        "metadata": {
            "vote_id": vote,
            "extracted_answer": answer,
            "gold_answer": "42",
            "individual_correct": correct,
        },
    }


def test_group_temperature_votes_uses_sample_prefix_for_ids_with_t_marker():
    rows = [
        _row("math/test_case", 0.7, 0, "a"),
        _row("math/test_case", 0.7, 1, "a"),
        _row("math/test_case", 1.0, 0, "b"),
        _row("math/other_t_in_id", 0.7, 0, "c"),
    ]
    grouped = group_temperature_votes(rows)
    assert set(grouped) == {
        ("math/test_case", 0.7),
        ("math/test_case", 1.0),
        ("math/other_t_in_id", 0.7),
    }
    assert len(grouped[("math/test_case", 0.7)]) == 2


def test_vote_margin_counts_second_answer_or_zero():
    assert vote_margin_from_answers(["a", "a", "a", "b"]) == 0.5
    assert vote_margin_from_answers(["a", "a", "a", "a"]) == 1.0
    assert vote_margin_from_answers([]) == 0.0


def test_majority_summary_uses_dataset_majority_label_and_vote_margin():
    rows = [
        _row("q", 0.8, 2, "wrong", correct=False, voting_label=1),
        _row("q", 0.8, 0, "right", correct=True, voting_label=1),
        _row("q", 0.8, 1, "wrong", correct=False, voting_label=1),
    ]
    summary = majority_summary_for_votes(rows)
    assert summary["majority_correct"] == 0
    assert summary["majority_count"] == 2
    assert math.isclose(summary["sc_confidence"], 2 / 3)
    assert math.isclose(summary["vote_margin"], 1 / 3)
    assert summary["individual_correct"] == [1, 0, 0]
    assert summary["token_counts"] == [1, 2, 3]


def test_assign_tertiles_makes_equal_count_buckets_and_stable_cuts():
    labels, cuts = assign_tertiles([0.9, 0.1, 0.5, 0.8, 0.2, 0.6])
    assert labels == ["high", "low", "mid", "high", "low", "mid"]
    assert cuts == (0.2, 0.6)
    assert {label: labels.count(label) for label in set(labels)} == {
        "low": 2,
        "mid": 2,
        "high": 2,
    }


def test_masked_segment_entropy_uses_feature_index_one_and_mask():
    # Two segments, two token slots per segment, four token features per slot.
    features = torch.zeros(2, 8)
    token_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.uint8)
    reshaped = features.reshape(2, 2, 4)
    reshaped[0, 0, 1] = 0.2
    reshaped[0, 1, 1] = 0.6
    reshaped[1, 0, 1] = 1.0
    reshaped[1, 1, 1] = 9.0
    entry = {"features": features, "token_mask": token_mask}
    values = masked_segment_entropy(entry, segment_size=2, token_dim=4)
    assert values.tolist() == pytest.approx([0.4, 1.0])
