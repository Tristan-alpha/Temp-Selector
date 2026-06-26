"""Tests for per-temperature continuation label summaries."""

from __future__ import annotations

from scripts.build_prefix_continuations import continuation_temperature_summary


def test_temperature_summary_aggregates_correct_and_total():
    summary = continuation_temperature_summary([
        {"temperature": 0.1, "correct": True},
        {"temperature": 0.1, "correct": False},
        {"temperature": 0.3, "correct": True},
        {"temperature": 0.3, "correct": True},
    ])
    assert summary["per_temperature_stats"]["0.1"] == {
        "n_correct": 1,
        "n_total": 2,
        "success_rate": 0.5,
    }
    assert summary["per_temperature_stats"]["0.3"] == {
        "n_correct": 2,
        "n_total": 2,
        "success_rate": 1.0,
    }


def test_temperature_summary_selects_oracle_temperature():
    summary = continuation_temperature_summary([
        {"temperature": 0.1, "correct": False},
        {"temperature": 0.1, "correct": True},
        {"temperature": 0.3, "correct": True},
        {"temperature": 0.3, "correct": True},
        {"temperature": 0.5, "correct": False},
        {"temperature": 0.5, "correct": False},
    ])
    assert summary["oracle_temperature"] == 0.3
    assert summary["oracle_success_rate"] == 1.0


def test_temperature_summary_tie_breaks_to_lower_temperature():
    summary = continuation_temperature_summary([
        {"temperature": 0.1, "correct": True},
        {"temperature": 0.1, "correct": False},
        {"temperature": 0.3, "correct": True},
        {"temperature": 0.3, "correct": False},
    ])
    assert summary["oracle_temperature"] == 0.1
    assert summary["oracle_success_rate"] == 0.5


def test_temperature_summary_preserves_global_counts():
    continuations = [
        {"temperature": 0.1, "correct": True},
        {"temperature": 0.1, "correct": False},
        {"temperature": 0.3, "correct": True},
    ]
    summary = continuation_temperature_summary(continuations)
    n_correct = sum(int(item["correct"]) for item in continuations)
    n_total = len(continuations)
    stats_correct = sum(
        item["n_correct"] for item in summary["per_temperature_stats"].values()
    )
    stats_total = sum(
        item["n_total"] for item in summary["per_temperature_stats"].values()
    )
    assert stats_correct == n_correct
    assert stats_total == n_total


def test_temperature_summary_empty_continuations():
    summary = continuation_temperature_summary([])
    assert summary["per_temperature_stats"] == {}
    assert summary["oracle_temperature"] is None
    assert summary["oracle_success_rate"] is None
    assert summary["mean_success_rate"] == 0.0
    assert summary["temperature_success_variance"] == 0.0
