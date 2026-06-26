#!/usr/bin/env python3
"""Summarize self-consistency calibration for fixed and online strategies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_temperature_sweep_calibration import (
    temperature_sweep_from_rows,
)
from utils.calibration import (
    binary_nll,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    selective_risk_curve,
)
from utils.jsonl import load_jsonl


DEFAULT_FIXED_TEMPERATURES = [0.1, 0.3, 0.7, 1.0]


def _prediction_confidence(prediction: Dict[str, Any], num_votes: int | None = None) -> float:
    if "sc_confidence" in prediction:
        return float(prediction["sc_confidence"])
    if "majority_count" in prediction:
        votes = int(prediction.get("num_votes", num_votes or 0))
        return float(prediction["majority_count"]) / max(1, votes)
    individual = prediction.get("individual_correct", [])
    if individual:
        correct = sum(int(value) for value in individual)
        return max(correct, len(individual) - correct) / len(individual)
    return 0.0


def _prediction_entropy(prediction: Dict[str, Any]) -> float:
    if "answer_entropy" in prediction:
        return float(prediction["answer_entropy"])
    return 0.0


def metrics_from_online_predictions(predictions: Sequence[Dict[str, Any]],
                                    n_bins: int = 10,
                                    num_votes: int | None = None) -> Dict[str, Any]:
    confidences = [_prediction_confidence(pred, num_votes=num_votes) for pred in predictions]
    correctness = [int(pred.get("majority_correct", 0)) for pred in predictions]
    token_counts = [
        int(token)
        for pred in predictions
        for token in pred.get("token_counts", [])
    ]
    individual_correct = [
        int(value)
        for pred in predictions
        for value in pred.get("individual_correct", [])
    ]
    if not predictions:
        return {
            "majority_accuracy": 0.0,
            "individual_accuracy": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "nll": 0.0,
            "mean_confidence": 0.0,
            "mean_answer_entropy": 0.0,
            "mean_tokens": 0.0,
            "n_predictions": 0,
            "confidence_accuracy_bins": reliability_bins([], [], n_bins=n_bins),
            "selective_risk_curve": [],
        }
    return {
        "majority_accuracy": sum(correctness) / len(correctness),
        "individual_accuracy": sum(individual_correct) / max(1, len(individual_correct)),
        "ece": expected_calibration_error(confidences, correctness, n_bins=n_bins),
        "brier": brier_score(confidences, correctness),
        "nll": binary_nll(confidences, correctness),
        "mean_confidence": sum(confidences) / len(confidences),
        "mean_answer_entropy": (
            sum(_prediction_entropy(pred) for pred in predictions) / len(predictions)
        ),
        "mean_tokens": sum(token_counts) / max(1, len(token_counts)),
        "n_predictions": len(predictions),
        "confidence_accuracy_bins": reliability_bins(confidences, correctness, n_bins=n_bins),
        "selective_risk_curve": selective_risk_curve(confidences, correctness),
    }


def _append_strategy(strategies: Dict[str, List[Dict[str, Any]]],
                     strategy_name: str,
                     predictions: Sequence[Dict[str, Any]]) -> None:
    if predictions:
        strategies.setdefault(strategy_name, []).extend(dict(pred) for pred in predictions)


def collect_online_predictions(paths: Sequence[str]) -> tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    strategies: Dict[str, List[Dict[str, Any]]] = {}
    per_seed: List[Dict[str, Any]] = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(data.get("seed", -1))
        top_predictions = data.get("predictions", [])
        _append_strategy(strategies, "prefix_value_selector", top_predictions)
        if top_predictions:
            per_seed.append({
                "path": path,
                "seed": seed,
                "strategy": "prefix_value_selector",
                "predictions": top_predictions,
                "num_votes": int(data.get("num_votes", 0)),
            })

        best_fixed = data.get("best_fixed", {})
        best_fixed_predictions = best_fixed.get("predictions", [])
        if best_fixed_predictions:
            name = f"validation_best_fixed_online_t{float(data.get('best_fixed_temperature', 0.0)):.1f}"
            _append_strategy(strategies, name, best_fixed_predictions)
            per_seed.append({
                "path": path,
                "seed": seed,
                "strategy": name,
                "predictions": best_fixed_predictions,
                "num_votes": int(best_fixed.get("num_votes", data.get("num_votes", 0))),
            })

        random_result = data.get("random_temperature_per_segment") or data.get("random")
        if isinstance(random_result, dict):
            random_predictions = random_result.get("predictions", [])
            if random_predictions:
                _append_strategy(strategies, "random_temperature_per_segment", random_predictions)
                per_seed.append({
                    "path": path,
                    "seed": seed,
                    "strategy": "random_temperature_per_segment",
                    "predictions": random_predictions,
                    "num_votes": int(random_result.get("num_votes", data.get("num_votes", 0))),
                })
    return strategies, per_seed


def evaluate_self_consistency_calibration(config_path: str,
                                          split: str = "test",
                                          input_path: str | None = None,
                                          online_results: Sequence[str] = (),
                                          fixed_temperatures: Sequence[float] = DEFAULT_FIXED_TEMPERATURES,
                                          n_bins: int = 10) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_path = input_path or cfg["paths"][f"{split}_dataset"]
    rows = load_jsonl(data_path)

    fixed = temperature_sweep_from_rows(
        rows, temperatures=fixed_temperatures, n_bins=n_bins,
    )
    strategy_metrics: Dict[str, Any] = {}
    for row in fixed["temperatures"]:
        strategy_metrics[f"fixed_temperature_{row['temperature']:.1f}"] = {
            "source": "dataset_jsonl",
            "temperature": row["temperature"],
            "majority_accuracy": row["majority_vote_accuracy"],
            "individual_accuracy": row["pass_at_1_accuracy"],
            "ece": row["self_consistency_ece"],
            "brier": row["self_consistency_brier"],
            "nll": row["self_consistency_nll"],
            "mean_confidence": row["self_consistency_confidence"],
            "mean_answer_entropy": row["answer_entropy"],
            "mean_tokens": row["mean_tokens"],
            "n_predictions": row["n_groups"],
            "confidence_accuracy_bins": fixed["reliability_bins"][str(float(row["temperature"]))],
            "selective_risk_curve": row["selective_risk_curve"],
        }

    online_by_strategy, per_seed_predictions = collect_online_predictions(online_results)
    for strategy, predictions in sorted(online_by_strategy.items()):
        strategy_metrics[strategy] = {
            "source": "online_rollout_json",
            **metrics_from_online_predictions(predictions, n_bins=n_bins),
        }

    per_seed_metrics = []
    for item in per_seed_predictions:
        per_seed_metrics.append({
            "path": item["path"],
            "seed": item["seed"],
            "strategy": item["strategy"],
            **metrics_from_online_predictions(
                item["predictions"], n_bins=n_bins,
                num_votes=item.get("num_votes"),
            ),
        })

    return {
        "config": config_path,
        "split": split,
        "input_path": data_path,
        "online_results": list(online_results),
        "fixed_temperatures": [float(value) for value in fixed_temperatures],
        "n_bins": n_bins,
        "strategies": strategy_metrics,
        "per_seed": per_seed_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--online-results", nargs="*", default=[])
    parser.add_argument("--fixed-temperatures", nargs="*", type=float, default=DEFAULT_FIXED_TEMPERATURES)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output", default="results/self_consistency_calibration.json")
    args = parser.parse_args()

    result = evaluate_self_consistency_calibration(
        args.config,
        split=args.split,
        input_path=args.input,
        online_results=args.online_results,
        fixed_temperatures=args.fixed_temperatures,
        n_bins=args.n_bins,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "n_strategies": len(result["strategies"]),
        "n_per_seed": len(result["per_seed"]),
    }, indent=2))


if __name__ == "__main__":
    main()
