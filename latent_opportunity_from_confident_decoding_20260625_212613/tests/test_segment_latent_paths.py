from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.segment_records import (  # noqa: E402
    opportunity_by_pvm_group,
    proposal_yield_by_temperature,
    selection_gain,
    standardize_segment_records,
)


def raw_rows() -> list[dict]:
    return [
        {
            "problem_id": "p0",
            "prefix_id": "p0::prefix",
            "source_sample_id": "p0_t0.1_v0",
            "prefix_text": "prefix low",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.2,
            "candidate_role": "greedy",
            "temperature": 0.0,
            "seed_index": 0,
            "generation_seed": 1,
            "segment_text": "bad greedy",
            "segment_token_ids": [1, 2],
            "child_pvm_score": 0.1,
            "correct": False,
        },
        {
            "problem_id": "p0",
            "prefix_id": "p0::prefix",
            "source_sample_id": "p0_t0.1_v0",
            "prefix_text": "prefix low",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.2,
            "candidate_role": "sample",
            "temperature": 0.3,
            "seed_index": 0,
            "generation_seed": 2,
            "segment_text": "recovering segment",
            "segment_token_ids": [9, 9],
            "child_pvm_score": 0.9,
            "correct": True,
        },
        {
            "problem_id": "p0",
            "prefix_id": "p0::prefix",
            "source_sample_id": "p0_t0.1_v0",
            "prefix_text": "prefix low",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.2,
            "candidate_role": "sample",
            "temperature": 0.3,
            "seed_index": 1,
            "generation_seed": 3,
            "segment_text": "recovering segment duplicate",
            "segment_token_ids": [9, 9],
            "child_pvm_score": 0.88,
            "correct": False,
        },
        {
            "problem_id": "p0",
            "prefix_id": "p0::prefix",
            "source_sample_id": "p0_t0.1_v0",
            "prefix_text": "prefix low",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.2,
            "candidate_role": "sample",
            "temperature": 0.7,
            "seed_index": 0,
            "generation_seed": 4,
            "segment_text": "noisy segment",
            "segment_token_ids": [7, 7],
            "child_pvm_score": 0.05,
            "correct": False,
        },
        {
            "problem_id": "p1",
            "prefix_id": "p1::prefix",
            "source_sample_id": "p1_t0.1_v0",
            "prefix_text": "prefix mid",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.6,
            "candidate_role": "greedy",
            "temperature": 0.0,
            "seed_index": 0,
            "generation_seed": 5,
            "segment_text": "strong greedy",
            "segment_token_ids": [3, 3],
            "child_pvm_score": 0.8,
            "correct": True,
        },
        {
            "problem_id": "p1",
            "prefix_id": "p1::prefix",
            "source_sample_id": "p1_t0.1_v0",
            "prefix_text": "prefix mid",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.6,
            "candidate_role": "sample",
            "temperature": 0.3,
            "seed_index": 0,
            "generation_seed": 6,
            "segment_text": "weaker sample",
            "segment_token_ids": [4, 4],
            "child_pvm_score": 0.5,
            "correct": False,
        },
        {
            "problem_id": "p2",
            "prefix_id": "p2::prefix",
            "source_sample_id": "p2_t0.1_v0",
            "prefix_text": "prefix high",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.9,
            "candidate_role": "greedy",
            "temperature": 0.0,
            "seed_index": 0,
            "generation_seed": 7,
            "segment_text": "high greedy",
            "segment_token_ids": [5, 5],
            "child_pvm_score": 0.8,
            "correct": True,
        },
        {
            "problem_id": "p2",
            "prefix_id": "p2::prefix",
            "source_sample_id": "p2_t0.1_v0",
            "prefix_text": "prefix high",
            "prefix_token_end": 64,
            "prefix_segments": 1,
            "prefix_pvm_score": 0.9,
            "candidate_role": "sample",
            "temperature": 0.7,
            "seed_index": 0,
            "generation_seed": 8,
            "segment_text": "high noisy",
            "segment_token_ids": [6, 6],
            "child_pvm_score": 0.4,
            "correct": False,
        },
    ]


def test_standardize_aggregates_duplicate_child_segment_rewards() -> None:
    records = standardize_segment_records(raw_rows())
    by_id = {row["prefix_id"]: row for row in records}

    assert [row["pvm_group"] for row in records] == ["low", "mid", "high"]
    p0_t03 = [
        sample for sample in by_id["p0::prefix"]["samples"]
        if sample["temperature"] == 0.3
    ]
    assert len(p0_t03) == 2
    assert {sample["reward"] for sample in p0_t03} == {0.5}
    assert {sample["n_total"] for sample in p0_t03} == {2}


def test_segment_opportunity_and_temperature_yield() -> None:
    records = standardize_segment_records(raw_rows())
    opportunity = opportunity_by_pvm_group(records, deltas=[0.0, 0.05, 0.10])
    by_group = {row["pvm_group"]: row for row in opportunity}

    assert by_group["low"]["opportunity_rate_delta_0.05"] == pytest.approx(1.0)
    assert by_group["mid"]["opportunity_rate_delta_0.05"] == pytest.approx(0.0)
    assert by_group["high"]["opportunity_rate_delta_0.05"] == pytest.approx(0.0)

    temp_rows = proposal_yield_by_temperature(records, delta=0.05)
    by_temp = {row["temperature"]: row for row in temp_rows}
    assert by_temp[0.3]["n_samples"] == 3
    assert by_temp[0.3]["yield"] == pytest.approx(2 / 3)
    assert by_temp[0.3]["best_of_n_yield"] == pytest.approx(0.5)


def test_pvm_selection_gain_prefers_high_child_pvm_sample() -> None:
    records = standardize_segment_records(raw_rows())
    rows = selection_gain(records)
    by_scope = {row["scope"]: row for row in rows}

    assert by_scope["all"]["n_prefixes"] == 3
    assert by_scope["all"]["Acc_PVM_best"] > by_scope["all"]["Acc_random_sampled"]
    assert by_scope["low_pvm"]["Acc_PVM_best"] == pytest.approx(0.5)
    assert by_scope["low_pvm"]["Acc_greedy"] == pytest.approx(0.0)

