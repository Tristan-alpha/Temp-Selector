"""Synthetic-data tests for calibration evaluation scripts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import yaml

from scripts.eval_fixed_temperature_online import summarize_fixed_rollout
from scripts.eval_q_selector import load_eval_prompts
from scripts.eval_efficiency_tradeoff import evaluate_efficiency_tradeoff_from_rows
from scripts.eval_self_consistency_calibration import (
    evaluate_self_consistency_calibration,
    metrics_from_online_predictions,
)
from scripts.summarize_eval200_online import dataset_audit
from scripts.eval_temperature_sweep_calibration import temperature_sweep_from_rows


def _row(problem: str, temp: float, vote: int, answer: str, gold: str = "1") -> dict:
    correct = answer == gold
    return {
        "sample_id": f"{problem}_t{temp}_v{vote}",
        "prompt": f"{problem}?",
        "response": f"solution \\boxed{{{answer}}}" if answer != "NOANSWER" else "solution only",
        "individual_label": 0 if correct else 1,
        "voting_label": 0,
        "temperature": temp,
        "token_ids": list(range(vote + 1)),
        "tokens": ["x"] * (vote + 1),
        "metadata": {
            "gold_answer": gold,
            "vote_id": vote,
            "num_votes": 4,
        },
    }


def _rows() -> list[dict]:
    return [
        _row("q1", 0.1, 0, "1"),
        _row("q1", 0.1, 1, "1"),
        _row("q1", 0.1, 2, "2"),
        _row("q1", 0.1, 3, "NOANSWER"),
        _row("q2", 0.1, 0, "2"),
        _row("q2", 0.1, 1, "2"),
        _row("q2", 0.1, 2, "1"),
        _row("q2", 0.1, 3, "NOANSWER"),
    ]


def test_temperature_sweep_from_synthetic_rows():
    result = temperature_sweep_from_rows(_rows(), temperatures=[0.1], n_bins=2)
    row = result["temperatures"][0]
    assert row["temperature"] == 0.1
    assert row["n_groups"] == 2
    assert row["num_votes"] == 4
    assert row["majority_vote_accuracy"] == 0.5
    assert 0.0 <= row["self_consistency_ece"] <= 1.0


def test_efficiency_tradeoff_from_synthetic_rows():
    result = evaluate_efficiency_tradeoff_from_rows(
        _rows(), temperatures=[0.1], thresholds=[0.75], min_votes=2, max_votes=4,
    )
    assert len(result["fixed_votes"]) == 1
    assert len(result["confidence_early_stop"]) == 1
    early = result["confidence_early_stop"][0]
    assert early["average_votes_used"] <= 4
    assert "token_reduction_vs_fixed_8_votes" in early


def test_self_consistency_calibration_reads_dataset_and_online_json(tmp_path):
    data_path = tmp_path / "test.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"paths": {"test_dataset": str(data_path)}}),
        encoding="utf-8",
    )
    online_path = tmp_path / "online.json"
    online_path.write_text(json.dumps({
        "seed": 42,
        "num_votes": 4,
        "predictions": [
            {
                "problem_id": "q1",
                "majority_correct": 1,
                "individual_correct": [1, 1, 0, 0],
                "sc_confidence": 0.5,
                "answer_entropy": 1.0,
                "token_counts": [1, 2, 3, 4],
            }
        ],
    }), encoding="utf-8")
    result = evaluate_self_consistency_calibration(
        str(cfg_path),
        split="test",
        online_results=[str(online_path)],
        fixed_temperatures=[0.1],
    )
    assert "fixed_temperature_0.1" in result["strategies"]
    assert "prefix_value_selector" in result["strategies"]
    assert result["per_seed"][0]["seed"] == 42


def test_metrics_from_online_predictions_falls_back_to_individual_correct():
    metrics = metrics_from_online_predictions([
        {"majority_correct": 1, "individual_correct": [1, 1, 0, 0], "token_counts": [1, 1, 1, 1]},
        {"majority_correct": 0, "individual_correct": [0, 0, 1, 0], "token_counts": [1, 1, 1, 1]},
    ])
    assert metrics["n_predictions"] == 2
    assert 0.0 <= metrics["ece"] <= 1.0


def test_load_eval_prompts_uses_input_override(tmp_path):
    default_path = tmp_path / "default.jsonl"
    override_path = tmp_path / "override.jsonl"
    default_path.write_text(json.dumps(_row("default", 0.1, 0, "1")) + "\n", encoding="utf-8")
    override_path.write_text(json.dumps(_row("override", 0.1, 0, "1")) + "\n", encoding="utf-8")
    prompts, data_path = load_eval_prompts(
        {"paths": {"test_dataset": str(default_path)}},
        input_path=str(override_path),
    )
    assert data_path == str(override_path)
    assert [item["problem_id"] for item in prompts] == ["override"]


def test_fixed_temperature_summarizer_calibration_metrics():
    rollout = SimpleNamespace(
        majority_correct=[1, 0],
        individual_correct=[[1, 1, 0, 0], [0, 0, 1, 0]],
        extracted_answers=[["1", "1", "2", "<NO_ANSWER>"], ["2", "2", "1", "<NO_ANSWER>"]],
        majority_answers=["1", "2"],
        majority_counts=[2, 2],
        sc_confidences=[0.5, 0.5],
        answer_entropies=[1.0, 1.0],
        temperatures=[[1.0, 1.0], [1.0]],
        segment_counts=[[1, 1, 1, 1], [1, 1, 1, 1]],
        token_counts=[[1, 2, 3, 4], [2, 2, 2, 2]],
    )
    metrics = summarize_fixed_rollout(
        "fixed_temperature_1.0",
        42,
        1.0,
        rollout,
        [{"problem_id": "q1"}, {"problem_id": "q2"}],
        4,
        1.5,
    )
    assert metrics["majority_accuracy"] == 0.5
    assert metrics["pass_at_1_accuracy"] == 0.5
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["selected_temperature_distribution"] == {"1.0": 3}


def test_dataset_audit_counts_prompts_and_train_overlap(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(json.dumps(_row("train_seen", 0.1, 0, "1")) + "\n", encoding="utf-8")
    rows = [
        _row("eval_a", 0.1, 0, "1"),
        _row("eval_a", 0.1, 1, "1"),
        _row("train_seen", 0.1, 0, "1"),
    ]
    eval_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    audit = dataset_audit(str(eval_path), str(train_path))
    assert audit["n_rows"] == 3
    assert audit["n_prompts"] == 2
    assert audit["train_overlap_count"] == 1
