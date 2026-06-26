#!/usr/bin/env python3
"""Evaluate Prefix Value Model calibration against continuation success rates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mil.prefix_data import IndexDataset, continuation_collate
from mil.prefix_value import PrefixValueModel, calibrated_probability
from utils.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
)
from utils.jsonl import load_jsonl


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _binomial_nll_from_probability(probabilities: torch.Tensor,
                                   n_correct: torch.Tensor,
                                   n_total: torch.Tensor) -> torch.Tensor:
    p = probabilities.to(torch.float64).clamp(1e-6, 1.0 - 1e-6)
    correct = n_correct.to(torch.float64)
    total = n_total.to(torch.float64).clamp_min(1.0)
    return -(correct * torch.log(p) + (total - correct) * torch.log1p(-p)) / total


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().to(torch.float64)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    sorted_values = values[order]
    start = 0
    n = int(values.numel())
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(values: torch.Tensor, targets: torch.Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    rank_values = _average_ranks(values)
    rank_targets = _average_ranks(targets)
    vx = rank_values - rank_values.mean()
    vy = rank_targets - rank_targets.mean()
    denom = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy))
    if float(denom.item()) <= 0.0:
        return 0.0
    return float(torch.sum(vx * vy).item() / denom.item())


def _record_stage(record: Dict[str, Any]) -> str:
    stage = record.get("prefix_stage")
    if stage:
        return str(stage)
    n_segments = int(record.get("n_segments", 0))
    prefix_segments = int(record.get("prefix_segments", 0))
    if n_segments <= 0 or prefix_segments <= 0:
        return "unknown"
    q = prefix_segments / n_segments
    if q < 0.30:
        return "early"
    if q < 0.65:
        return "middle"
    return "late"


def _mean_tensor(values: torch.Tensor) -> float:
    return float(values.mean().item()) if values.numel() else 0.0


def _stage_metrics(records: Sequence[Dict[str, Any]],
                   probabilities: torch.Tensor,
                   observed_rate: torch.Tensor,
                   posterior_targets: torch.Tensor,
                   per_record_nll: torch.Tensor,
                   n_bins: int) -> Dict[str, Any]:
    by_stage: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_stage[_record_stage(record)].append(idx)
    result: Dict[str, Any] = {}
    for stage, indices in sorted(by_stage.items()):
        idx = torch.tensor(indices, dtype=torch.long)
        stage_prob = probabilities[idx]
        stage_obs = observed_rate[idx]
        stage_post = posterior_targets[idx]
        result[stage] = {
            "n_prefixes": len(indices),
            "mean_phi": _mean_tensor(stage_prob),
            "observed_success_rate": _mean_tensor(stage_obs),
            "posterior_mean": _mean_tensor(stage_post),
            "brier": brier_score(stage_prob, stage_post),
            "ece": expected_calibration_error(stage_prob, stage_obs, n_bins=n_bins),
            "binomial_nll": _mean_tensor(per_record_nll[idx]),
        }
    return result


def _quartile_metrics(probabilities: torch.Tensor,
                      observed_rate: torch.Tensor) -> Dict[str, Any]:
    n = int(probabilities.numel())
    if n == 0:
        return {
            "bottom_n": 0,
            "top_n": 0,
            "bottom_mean_phi": 0.0,
            "top_mean_phi": 0.0,
            "bottom_observed_rate": 0.0,
            "top_observed_rate": 0.0,
            "observed_rate_delta": 0.0,
        }
    k = max(1, n // 4)
    order = torch.argsort(probabilities)
    bottom = order[:k]
    top = order[-k:]
    return {
        "bottom_n": int(bottom.numel()),
        "top_n": int(top.numel()),
        "bottom_mean_phi": _mean_tensor(probabilities[bottom]),
        "top_mean_phi": _mean_tensor(probabilities[top]),
        "bottom_observed_rate": _mean_tensor(observed_rate[bottom]),
        "top_observed_rate": _mean_tensor(observed_rate[top]),
        "observed_rate_delta": _mean_tensor(observed_rate[top]) - _mean_tensor(observed_rate[bottom]),
    }


def evaluate_prefix_capability_calibration(config_path: str,
                                           split: str = "val",
                                           checkpoint_path: str | None = None,
                                           continuations_path: str | None = None,
                                           feature_cache_path: str | None = None,
                                           batch_size: int | None = None,
                                           n_bins: int = 10,
                                           device_name: str | None = None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = checkpoint_path or cfg["paths"]["prefix_value_ckpt"]
    continuations_path = continuations_path or cfg["paths"][f"{split}_continuations"]
    feature_cache_path = feature_cache_path or cfg["paths"][f"{split}_feature_cache"]
    batch_size = batch_size or int(cfg["prefix_value"]["training"].get("batch_size", 32))

    records = load_jsonl(continuations_path)
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    cache_by_id = {str(entry["sample_id"]): entry for entry in cache}
    missing = sorted({
        str(record["source_sample_id"])
        for record in records
        if str(record["source_sample_id"]) not in cache_by_id
    })
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"feature cache is missing {len(missing)} source_sample_id values: {preview}")

    device = torch.device(device_name) if device_name else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = cfg["prefix_value"]["model"]
    model = PrefixValueModel(
        token_dim=int(cfg["data"]["instance_dim"]),
        segment_size=int(cfg["data"]["segment_size"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        max_segments=int(model_cfg.get("max_segments", 8192)),
        n_temps=int(model_cfg.get("n_temps", 0)),
        prompt_dim=int(model_cfg.get("prompt_dim", 0) or 0),
        prompt_integration=str(model_cfg.get("prompt_integration", "none")),
    ).to(device)
    model.load_state_dict(checkpoint["prefix_value"])
    model.eval()
    calibration_temperature = float(checkpoint.get("calibration_temperature", 1.0))

    loader = DataLoader(
        IndexDataset(len(records)), batch_size=batch_size, shuffle=False,
        num_workers=0,
        collate_fn=partial(continuation_collate, cache_by_id, records),
    )
    logits_all: List[torch.Tensor] = []
    correct_all: List[torch.Tensor] = []
    total_all: List[torch.Tensor] = []
    target_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            output = model(
                batch["features"],
                batch["token_mask"],
                batch["segment_mask"],
                prompt_hidden=batch.get("prompt_hidden"),
            )
            logits_all.append(output["terminal_logits"].detach().cpu())
            correct_all.append(batch["n_correct"].detach().cpu())
            total_all.append(batch["n_total"].detach().cpu())
            target_all.append(batch["target"].detach().cpu())

    if not logits_all:
        bins = reliability_bins([], [], n_bins=n_bins)
        return {
            "config": config_path,
            "split": split,
            "checkpoint": checkpoint_path,
            "continuations": continuations_path,
            "feature_cache": feature_cache_path,
            "n_prefixes": 0,
            "calibration_temperature": calibration_temperature,
            "brier": 0.0,
            "ece": 0.0,
            "binomial_nll": 0.0,
        }, bins

    logits = torch.cat(logits_all)
    n_correct = torch.cat(correct_all)
    n_total = torch.cat(total_all).clamp_min(1.0)
    posterior_targets = torch.cat(target_all)
    probabilities = calibrated_probability(logits, calibration_temperature).cpu()
    observed_rate = n_correct / n_total
    per_record_nll = _binomial_nll_from_probability(probabilities, n_correct, n_total)
    bins = reliability_bins(probabilities, observed_rate, n_bins=n_bins)
    quartiles = _quartile_metrics(probabilities, observed_rate)

    result = {
        "config": config_path,
        "split": split,
        "checkpoint": checkpoint_path,
        "continuations": continuations_path,
        "feature_cache": feature_cache_path,
        "n_prefixes": int(probabilities.numel()),
        "calibration_temperature": calibration_temperature,
        "brier": brier_score(probabilities, posterior_targets),
        "brier_observed": brier_score(probabilities, observed_rate),
        "brier_posterior": brier_score(probabilities, posterior_targets),
        "ece": expected_calibration_error(probabilities, observed_rate, n_bins=n_bins),
        "ece_posterior": expected_calibration_error(probabilities, posterior_targets, n_bins=n_bins),
        "binomial_nll": _mean_tensor(per_record_nll),
        "spearman": _spearman(probabilities, observed_rate),
        "mean_phi": _mean_tensor(probabilities),
        "observed_success_rate": _mean_tensor(observed_rate),
        "posterior_mean": _mean_tensor(posterior_targets),
        "top_bottom_quartile_observed_success_delta": quartiles["observed_rate_delta"],
        "phi_quartiles": quartiles,
        "stage_metrics": _stage_metrics(
            records, probabilities, observed_rate, posterior_targets, per_record_nll, n_bins,
        ),
        "n_total_distribution": {
            str(int(value.item())): int((n_total == value).sum().item())
            for value in torch.unique(n_total)
        },
    }
    return result, bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--continuations", default=None)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="results/prefix_capability_calibration.json")
    parser.add_argument("--bins-output", default="results/prefix_capability_reliability_bins.json")
    args = parser.parse_args()

    result, bins = evaluate_prefix_capability_calibration(
        args.config,
        split=args.split,
        checkpoint_path=args.checkpoint,
        continuations_path=args.continuations,
        feature_cache_path=args.feature_cache,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
        device_name=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    bins_output = Path(args.bins_output)
    bins_output.parent.mkdir(parents=True, exist_ok=True)
    bins_output.write_text(json.dumps(bins, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "bins_output": str(bins_output),
        "n_prefixes": result["n_prefixes"],
        "brier": result["brier"],
        "ece": result["ece"],
        "binomial_nll": result["binomial_nll"],
    }, indent=2))


if __name__ == "__main__":
    main()
