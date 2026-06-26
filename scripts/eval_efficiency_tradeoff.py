#!/usr/bin/env python3
"""Evaluate confidence early-stop tradeoffs from fixed-temperature votes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_temperature_sweep_calibration import (
    group_dataset_votes,
    majority_vote_summary,
    metrics_from_vote_summaries,
    vote_index,
)
from utils.jsonl import load_jsonl


def _average(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _cost_metrics(summaries: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "average_votes_used": _average([float(item["num_votes"]) for item in summaries]),
        "average_generated_tokens": _average([float(item["total_tokens"]) for item in summaries]),
        "total_generated_tokens": float(sum(int(item["total_tokens"]) for item in summaries)),
    }


def _majority_confidence_so_far(rows: Sequence[Dict[str, Any]]) -> float:
    summary = majority_vote_summary(rows)
    return float(summary["sc_confidence"])


def early_stop_rows(rows: Sequence[Dict[str, Any]],
                    threshold: float,
                    min_votes: int,
                    max_votes: int) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=vote_index)[:max_votes]
    selected: List[Dict[str, Any]] = []
    for row in ordered:
        selected.append(row)
        if len(selected) < min_votes:
            continue
        if _majority_confidence_so_far(selected) >= float(threshold):
            break
    return selected


def evaluate_efficiency_tradeoff_from_rows(rows: Sequence[Dict[str, Any]],
                                           temperatures: Sequence[float] | None = None,
                                           thresholds: Sequence[float] = (0.75, 0.875, 1.0),
                                           min_votes: int = 4,
                                           max_votes: int = 8,
                                           n_bins: int = 10) -> Dict[str, Any]:
    grouped = group_dataset_votes(rows)
    available_temps = sorted({temp for _, temp in grouped})
    selected_temps = [float(temp) for temp in temperatures] if temperatures is not None else available_temps
    fixed_votes: List[Dict[str, Any]] = []
    early_stop: List[Dict[str, Any]] = []

    baseline_by_temp: Dict[float, Dict[str, Any]] = {}
    for temp in selected_temps:
        buckets = [
            bucket
            for (_pid, group_temp), bucket in sorted(grouped.items())
            if group_temp == float(temp)
        ]
        baseline_summaries = [
            majority_vote_summary(sorted(bucket, key=vote_index)[:max_votes])
            for bucket in buckets
        ]
        baseline_metrics = metrics_from_vote_summaries(baseline_summaries, n_bins=n_bins)
        baseline_metrics.pop("reliability_bins", None)
        baseline = {
            "strategy": "fixed_votes",
            "temperature": float(temp),
            "max_votes": int(max_votes),
            **baseline_metrics,
            **_cost_metrics(baseline_summaries),
        }
        fixed_votes.append(baseline)
        baseline_by_temp[float(temp)] = baseline

        for threshold in thresholds:
            stopped_summaries = [
                majority_vote_summary(early_stop_rows(
                    bucket, threshold=float(threshold),
                    min_votes=min_votes, max_votes=max_votes,
                ))
                for bucket in buckets
            ]
            stopped_metrics = metrics_from_vote_summaries(stopped_summaries, n_bins=n_bins)
            stopped_metrics.pop("reliability_bins", None)
            costs = _cost_metrics(stopped_summaries)
            token_reduction = 1.0 - (
                costs["average_generated_tokens"] /
                max(1.0, float(baseline["average_generated_tokens"]))
            )
            vote_reduction = 1.0 - (
                costs["average_votes_used"] /
                max(1.0, float(baseline["average_votes_used"]))
            )
            early_stop.append({
                "strategy": "confidence_early_stop",
                "temperature": float(temp),
                "threshold": float(threshold),
                "min_votes": int(min_votes),
                "max_votes": int(max_votes),
                **stopped_metrics,
                **costs,
                "token_reduction_vs_fixed_8_votes": token_reduction,
                "vote_reduction_vs_fixed_8_votes": vote_reduction,
                "accuracy_drop_vs_fixed_8_votes": (
                    float(baseline["majority_vote_accuracy"]) -
                    float(stopped_metrics["majority_vote_accuracy"])
                ),
                "ece_change_vs_fixed_8_votes": (
                    float(stopped_metrics["self_consistency_ece"]) -
                    float(baseline["self_consistency_ece"])
                ),
                "brier_change_vs_fixed_8_votes": (
                    float(stopped_metrics["self_consistency_brier"]) -
                    float(baseline["self_consistency_brier"])
                ),
                "nll_change_vs_fixed_8_votes": (
                    float(stopped_metrics["self_consistency_nll"]) -
                    float(baseline["self_consistency_nll"])
                ),
            })

    return {
        "fixed_votes": fixed_votes,
        "confidence_early_stop": early_stop,
    }


def evaluate_efficiency_tradeoff(config_path: str,
                                 split: str = "test",
                                 input_path: str | None = None,
                                 temperatures: Sequence[float] | None = None,
                                 thresholds: Sequence[float] = (0.75, 0.875, 1.0),
                                 min_votes: int = 4,
                                 max_votes: int = 8,
                                 n_bins: int = 10) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_path = input_path or cfg["paths"][f"{split}_dataset"]
    if temperatures is None:
        temperatures = cfg.get("inference", {}).get("temperature_grid")
    rows = load_jsonl(data_path)
    result = evaluate_efficiency_tradeoff_from_rows(
        rows,
        temperatures=temperatures,
        thresholds=thresholds,
        min_votes=min_votes,
        max_votes=max_votes,
        n_bins=n_bins,
    )
    result.update({
        "config": config_path,
        "split": split,
        "input_path": data_path,
        "thresholds": [float(value) for value in thresholds],
        "min_votes": int(min_votes),
        "max_votes": int(max_votes),
        "n_bins": int(n_bins),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="results/efficiency_tradeoff.json")
    parser.add_argument("--temperatures", nargs="*", type=float, default=None)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.75, 0.875, 1.0])
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument("--max-votes", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    result = evaluate_efficiency_tradeoff(
        args.config,
        split=args.split,
        input_path=args.input,
        temperatures=args.temperatures,
        thresholds=args.thresholds,
        min_votes=args.min_votes,
        max_votes=args.max_votes,
        n_bins=args.n_bins,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "n_fixed_rows": len(result["fixed_votes"]),
        "n_early_stop_rows": len(result["confidence_early_stop"]),
    }, indent=2))


if __name__ == "__main__":
    main()
