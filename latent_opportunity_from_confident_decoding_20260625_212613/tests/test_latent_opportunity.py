from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "latent_opportunity" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_opportunity.candidate_builder import build_records_from_trace_targets
from latent_opportunity.opportunity_metrics import (
    build_prefix_summaries,
    negative_control_table,
    per_prefix_temperature_rows,
    temperature_aggregate_table,
)
from latent_opportunity.pvm_scoring import assign_pvm_groups


def synthetic_rows():
    traces = [
        {
            "trace_id": "p0::tok4",
            "problem_id": "p0",
            "source_sample_id": "p0_t1.0_v0",
            "prompt": "prompt 0",
            "prefix_text": "prefix 0",
            "prefix_token_ids": [1, 2, 3, 4],
            "token_index": 4,
            "relative_position": 0.25,
            "pvm_score_prefix": 0.2,
            "layer_ids": [2, 3],
            "topk_token_ids_by_layer": [[10, 40, 20], [10, 20, 30]],
            "topk_probs_by_layer": [[0.5, 0.3, 0.2], [0.7, 0.2, 0.1]],
            "hidden_state_path": "/tmp/h0.pt",
        },
        {
            "trace_id": "p1::tok5",
            "problem_id": "p1",
            "source_sample_id": "p1_t1.0_v0",
            "prompt": "prompt 1",
            "prefix_text": "prefix 1",
            "prefix_token_ids": [5, 6],
            "token_index": 5,
            "relative_position": 0.75,
            "pvm_score_prefix": 0.8,
            "layer_ids": [2, 3],
            "topk_token_ids_by_layer": [[11, 41, 21], [11, 21, 31]],
            "topk_probs_by_layer": [[0.45, 0.35, 0.2], [0.8, 0.15, 0.05]],
            "hidden_state_path": "/tmp/h1.pt",
        },
    ]
    targets = [
        {
            "trace_id": "p0::tok4",
            "layer_id": 2,
            "topk_token_ids": [10, 40, 20],
            "topk_probs": [0.5, 0.3, 0.2],
            "child_pvm_scores": [0.4, 0.9, 0.8],
        },
        {
            "trace_id": "p0::tok4",
            "layer_id": 3,
            "topk_token_ids": [10, 20, 30],
            "topk_probs": [0.7, 0.2, 0.1],
            "child_pvm_scores": [0.4, 0.8, 0.3],
        },
        {
            "trace_id": "p1::tok5",
            "layer_id": 2,
            "topk_token_ids": [11, 41, 21],
            "topk_probs": [0.45, 0.35, 0.2],
            "child_pvm_scores": [0.6, 0.75, 0.61],
        },
        {
            "trace_id": "p1::tok5",
            "layer_id": 3,
            "topk_token_ids": [11, 21, 31],
            "topk_probs": [0.8, 0.15, 0.05],
            "child_pvm_scores": [0.6, 0.61, 0.58],
        },
    ]
    return traces, targets


def test_candidate_builder_identifies_greedy_and_deduplicates_sources() -> None:
    traces, targets = synthetic_rows()
    prefixes, candidates, sources = build_records_from_trace_targets(traces, targets)
    prefixes = assign_pvm_groups(prefixes)

    assert len(prefixes) == 2
    assert {row["pvm_group"] for row in prefixes} == {"low", "high"}
    p0_greedy = [row for row in candidates if row["prefix_id"] == "p0::tok4" and row["is_final_greedy"]]
    assert len(p0_greedy) == 1
    assert p0_greedy[0]["candidate_token_id"] == 10

    token20 = [
        row for row in candidates
        if row["prefix_id"] == "p0::tok4" and row["candidate_token_id"] == 20
    ][0]
    assert token20["appears_final_topk_alt"] is True
    assert token20["appears_near_final_topk_alt"] is True
    assert token20["is_duplicate_candidate"] is True
    assert token20["delta_vs_final_greedy"] == pytest.approx(0.4)
    assert len([row for row in sources if row["prefix_id"] == "p0::tok4" and row["candidate_token_id"] == 20]) == 2


def test_candidate_builder_rejects_inconsistent_child_scores() -> None:
    traces, targets = synthetic_rows()
    targets[1] = dict(targets[1])
    targets[1]["child_pvm_scores"] = [0.4, 0.81, 0.3]
    with pytest.raises(ValueError, match="inconsistent PVM scores"):
        build_records_from_trace_targets(traces, targets)


def test_opportunity_and_temperature_metrics() -> None:
    traces, targets = synthetic_rows()
    prefixes, candidates, sources = build_records_from_trace_targets(traces, targets)
    prefixes = assign_pvm_groups(prefixes)
    summaries = build_prefix_summaries(
        prefixes,
        candidates,
        deltas=[0.01, 0.03],
        default_delta=0.03,
    )
    by_id = {row["prefix_id"]: row for row in summaries}
    assert by_id["p0::tok4"]["overall_opportunity_0.03"] is True
    assert by_id["p0::tok4"]["final_opportunity_0.03"] is True
    assert by_id["p1::tok5"]["final_opportunity_0.03"] is False
    assert by_id["p1::tok5"]["near_final_only_opportunity_0.03"] is True

    temp_rows = per_prefix_temperature_rows(
        prefixes,
        sources,
        temperatures=[0.0, 1.0],
        default_delta=0.03,
    )
    p0_final_t0 = [
        row for row in temp_rows
        if row["prefix_id"] == "p0::tok4"
        and row["source_family"] == "final_layer"
        and row["temperature"] == 0.0
    ][0]
    p0_final_t1 = [
        row for row in temp_rows
        if row["prefix_id"] == "p0::tok4"
        and row["source_family"] == "final_layer"
        and row["temperature"] == 1.0
    ][0]
    assert p0_final_t0["P_T_A_plus"] == 0.0
    assert p0_final_t1["P_T_A_plus"] > 0.0

    temp_table = temperature_aggregate_table(temp_rows)
    assert {row["source_family"] for row in temp_table} == {"final_layer", "near_final_oracle_layer"}

    controls = negative_control_table(
        summaries,
        sources,
        temp_rows,
        default_delta=0.03,
        n_bootstrap=20,
        seed=0,
    )
    assert {row["comparison"] for row in controls} == {
        "best_alt_vs_random_same_rank",
        "fixed_high_temperature_final_all",
        "fixed_high_temperature_final_opportunity_prefixes",
    }
