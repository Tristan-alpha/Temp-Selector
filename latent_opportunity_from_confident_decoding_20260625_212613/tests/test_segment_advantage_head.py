from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.segment_advantage import (  # noqa: E402
    build_advantage_examples,
    divergence_bin,
    first_divergence_position,
    generate_pairwise_indices,
    group_kfold_indices,
    roc_auc,
    selection_gain_rows,
    spearmanr,
)


def records() -> list[dict]:
    return [
        {
            "problem_id": "problem_a",
            "prefix_id": "a::p0",
            "source_sample_id": "a0",
            "pvm_group": "low",
            "prefix_pvm_score": 0.2,
            "greedy": {
                "reward": 0.0,
                "segment_token_ids": [1, 2, 3, 4],
            },
            "samples": [
                {
                    "candidate_id": "a-good",
                    "temperature": 0.7,
                    "seed_index": 0,
                    "reward": 1.0,
                    "child_pvm": 0.1,
                    "segment_token_ids": [1, 9, 3, 4],
                },
                {
                    "candidate_id": "a-bad",
                    "temperature": 1.1,
                    "seed_index": 1,
                    "reward": 0.0,
                    "child_pvm": 0.9,
                    "segment_token_ids": [1, 2, 3, 4],
                },
            ],
        },
        {
            "problem_id": "problem_b",
            "prefix_id": "b::p0",
            "source_sample_id": "b0",
            "pvm_group": "high",
            "prefix_pvm_score": 0.8,
            "greedy": {
                "reward": 1.0,
                "segment_token_ids": [5, 6, 7, 8],
            },
            "samples": [
                {
                    "candidate_id": "b-tie",
                    "temperature": 0.3,
                    "seed_index": 0,
                    "reward": 1.0,
                    "child_pvm": 0.6,
                    "segment_token_ids": [5, 6, 7, 8],
                },
                {
                    "candidate_id": "b-worse",
                    "temperature": 0.9,
                    "seed_index": 1,
                    "reward": 0.0,
                    "child_pvm": 0.7,
                    "segment_token_ids": [5, 6, 0, 8],
                },
            ],
        },
    ]


def test_first_divergence_position_and_bins() -> None:
    assert first_divergence_position([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence_position([9, 2, 3], [1, 2, 3]) == 0
    assert first_divergence_position([1, 2, 9], [1, 2, 3]) == 2
    assert first_divergence_position([1, 2, 3, 4], [1, 2, 3]) == 3
    assert divergence_bin(None) == "no_divergence"
    assert divergence_bin(4) == "0-4"
    assert divergence_bin(15) == "5-15"
    assert divergence_bin(31) == "16-31"
    assert divergence_bin(63) == "32-63"


def test_advantage_examples_labels_and_pairwise_pairs() -> None:
    examples = build_advantage_examples(records(), delta=0.05)
    by_id = {row["candidate_id"]: row for row in examples}

    assert by_id["a-good"]["delta_reward"] == pytest.approx(1.0)
    assert by_id["a-good"]["better_than_greedy_delta"] == pytest.approx(1.0)
    assert by_id["a-bad"]["delta_reward"] == pytest.approx(0.0)
    assert by_id["a-bad"]["better_than_greedy_delta"] == pytest.approx(0.0)
    assert by_id["a-good"]["first_divergence_position"] == 1
    assert by_id["a-bad"]["divergence_bin"] == "no_divergence"

    pairs = generate_pairwise_indices(examples, range(len(examples)))
    pair_ids = {
        (examples[good]["candidate_id"], examples[bad]["candidate_id"])
        for good, bad, _ in pairs
    }
    assert ("a-good", "a-bad") in pair_ids
    assert ("b-tie", "b-worse") in pair_ids
    assert all(good != bad for good, bad, _ in pairs)


def test_problem_group_kfold_has_no_leakage() -> None:
    examples = build_advantage_examples(records(), delta=0.05)
    splits = group_kfold_indices(examples, group_key="problem_id", folds=2, seed=123)
    assert len(splits) == 2
    for train_idx, test_idx in splits:
        train_groups = {examples[idx]["problem_id"] for idx in train_idx}
        test_groups = {examples[idx]["problem_id"] for idx in test_idx}
        assert train_groups.isdisjoint(test_groups)
        assert test_idx


def test_rank_metrics_and_selector_gain() -> None:
    examples = build_advantage_examples(records(), delta=0.05)
    scores = {
        idx: (1.0 if example["candidate_id"] == "a-good" else 0.0)
        for idx, example in enumerate(examples)
    }
    rows = selection_gain_rows(
        examples,
        scores=scores,
        indices=range(len(examples)),
        delta=0.05,
        scopes=("all", "low_pvm", "high_pvm"),
    )
    by_scope = {row["scope"]: row for row in rows}

    assert by_scope["all"]["Acc_advantage_best"] == pytest.approx(1.0)
    assert by_scope["low_pvm"]["opportunity_capture_rate"] == pytest.approx(1.0)
    assert by_scope["high_pvm"]["Acc_advantage_best"] == pytest.approx(1.0)
    assert spearmanr([0.1, 0.2, 0.3], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert roc_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx(1.0)
