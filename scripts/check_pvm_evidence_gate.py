#!/usr/bin/env python3
"""Check static Prefix Value Model evidence before launching PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def evaluate_gate(metrics: Dict[str, Any]) -> Dict[str, Any]:
    n_total_distribution = metrics.get("n_total_distribution", {})
    n_total_ok = bool(n_total_distribution) and set(n_total_distribution) == {"32"}
    brier_gain = float(metrics.get("constant_brier", 1.0)) - float(metrics.get("brier", 1.0))
    nll_gain = (
        float(metrics.get("constant_binomial_nll", float("inf"))) -
        float(metrics.get("binomial_nll", float("inf")))
    )
    quartiles = metrics.get("phi_quartiles", {})
    quartile_delta = float(quartiles.get("observed_rate_delta", 0.0))
    checks = {
        "n_total_is_32": n_total_ok,
        "beats_constant_brier_or_nll": brier_gain > 0.0 or nll_gain > 0.0,
        "pair_accuracy_gt_0_58": float(metrics.get("pair_accuracy", 0.0)) > 0.58,
        "spearman_gt_0_10": float(metrics.get("spearman", 0.0)) > 0.10,
        "top_phi_beats_bottom_phi": quartile_delta > 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "brier_gain_vs_constant": brier_gain,
        "binomial_nll_gain_vs_constant": nll_gain,
        "quartile_observed_rate_delta": quartile_delta,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(
        cfg["paths"]["prefix_value_ckpt"], map_location="cpu", weights_only=False,
    )
    result = evaluate_gate(checkpoint.get("validation_metrics", {}))
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
