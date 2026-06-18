#!/usr/bin/env python3
"""Compare legacy and full online evaluations with paired cluster bootstrap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load(path: str, method: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if method == "legacy":
        ppo = data["ppo"]
        return {
            "accuracy": float(ppo["accuracy"]),
            "individual_accuracy": float(ppo.get("individual_accuracy", 0.0)),
            "total_tokens": int(ppo.get("total_tokens", 0)),
            "predictions": ppo.get("predictions", []),
        }
    return {
        "accuracy": float(data["majority_accuracy"]),
        "individual_accuracy": float(data.get("individual_accuracy", 0.0)),
        "total_tokens": int(data.get("total_tokens", 0)),
        "predictions": data.get("predictions", []),
    }


def paired_bootstrap(legacy_runs: List[Dict[str, Any]], full_runs: List[Dict[str, Any]],
                     iterations: int, seed: int) -> Tuple[float, float, float]:
    if len(legacy_runs) != len(full_runs):
        raise ValueError("legacy and full must contain the same number of seeds")
    per_seed = []
    common_ids = None
    for legacy, full in zip(legacy_runs, full_runs):
        legacy_map = {p["problem_id"]: float(p["majority_correct"]) for p in legacy["predictions"]}
        full_map = {p["problem_id"]: float(p["majority_correct"]) for p in full["predictions"]}
        ids = set(legacy_map) & set(full_map)
        common_ids = ids if common_ids is None else common_ids & ids
        per_seed.append((legacy_map, full_map))
    if not common_ids:
        raise ValueError("paired predictions with matching problem_id are required")
    ids = sorted(common_ids)
    differences = np.array([
        np.mean([full_map[pid] - legacy_map[pid] for legacy_map, full_map in per_seed])
        for pid in ids
    ], dtype=np.float64)
    rng = random.Random(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        draw = [differences[rng.randrange(len(differences))] for _ in ids]
        samples[i] = float(np.mean(draw))
    return float(differences.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", nargs="+", default=None)
    parser.add_argument("--historical-legacy-accuracy", type=float, default=None)
    parser.add_argument("--full", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--value-metrics", default=None)
    parser.add_argument("--output", default="results/comparison.json")
    args = parser.parse_args()
    full = [_load(path, "full") for path in args.full]
    full_accuracy = float(np.mean([run["accuracy"] for run in full]))
    full_tokens = float(np.mean([run["total_tokens"] for run in full]))
    if args.legacy:
        legacy = [_load(path, "legacy") for path in args.legacy]
        mean_diff, ci_low, ci_high = paired_bootstrap(
            legacy, full, args.iterations, args.seed,
        )
        legacy_accuracy = float(np.mean([run["accuracy"] for run in legacy]))
        legacy_individual = float(np.mean([run["individual_accuracy"] for run in legacy]))
        legacy_tokens = float(np.mean([run["total_tokens"] for run in legacy]))
        token_difference = (full_tokens - legacy_tokens) / max(1.0, legacy_tokens)
        ci = [ci_low, ci_high]
        ci_excludes_zero = ci_low > 0.0
        token_within_budget = abs(token_difference) <= 0.05
        comparison_type = "paired_recomputed_legacy"
    else:
        if args.historical_legacy_accuracy is None:
            parser.error("provide --legacy runs or --historical-legacy-accuracy")
        legacy_accuracy = float(args.historical_legacy_accuracy)
        mean_diff = full_accuracy - legacy_accuracy
        legacy_individual = None
        legacy_tokens = None
        token_difference = None
        ci = None
        ci_excludes_zero = None
        token_within_budget = None
        comparison_type = "historical_aggregate_reference"
    result = {
        "comparison_type": comparison_type,
        "legacy_accuracy": legacy_accuracy,
        "full_accuracy": full_accuracy,
        "accuracy_difference": mean_diff,
        "bootstrap_95_ci": ci,
        "legacy_individual_accuracy": legacy_individual,
        "full_individual_accuracy": float(np.mean([run["individual_accuracy"] for run in full])),
        "legacy_mean_total_tokens": legacy_tokens,
        "full_mean_total_tokens": full_tokens,
        "relative_token_difference": token_difference,
        "meets_accuracy_gain": mean_diff >= 0.02,
        "ci_excludes_zero": ci_excludes_zero,
        "token_budget_within_5_percent": token_within_budget,
    }
    result["effective"] = (
        bool(result["meets_accuracy_gain"] and result["ci_excludes_zero"] and
             result["token_budget_within_5_percent"])
        if args.legacy else None
    )
    if not args.legacy:
        result["statistical_note"] = (
            "Historical baseline contains no per-problem predictions, so paired bootstrap "
            "and matched token-budget comparisons are unavailable."
        )
    if args.value_metrics:
        result["prefix_value_metrics"] = json.loads(
            Path(args.value_metrics).read_text(encoding="utf-8")
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(
        "| Method | Majority accuracy | Individual accuracy | Mean total tokens |\n"
        "|---|---:|---:|---:|\n"
        f"| Legacy | {legacy_accuracy:.4f} | "
        f"{legacy_individual if legacy_individual is not None else 'N/A'} | "
        f"{legacy_tokens if legacy_tokens is not None else 'N/A'} |\n"
        f"| Full | {full_accuracy:.4f} | {result['full_individual_accuracy']:.4f} | {full_tokens:.0f} |\n\n"
        + (
            f"Paired difference: {mean_diff:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}].\n"
            if ci is not None else
            f"Difference from historical aggregate: {mean_diff:+.4f}. Paired CI unavailable.\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"output={output} table={markdown}")


if __name__ == "__main__":
    main()
