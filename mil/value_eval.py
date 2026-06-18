"""Evaluate calibrated Prefix Value Model on terminal test outcomes."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from inference.vllm_runner import VLLMFeatureExporter
from mil.eval import compute_auc
from mil.prefix_data import IndexDataset, precompute_feature_cache, terminal_collate
from mil.prefix_value import PrefixValueModel, calibrated_probability
from utils.jsonl import load_jsonl


def evaluate(config_path: str, parallel_size: int | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{max(0, n_gpu - 1)}") if n_gpu else torch.device("cpu")
    checkpoint = torch.load(cfg["paths"]["prefix_value_ckpt"], map_location=device, weights_only=False)
    model = PrefixValueModel(
        token_dim=int(cfg["data"]["instance_dim"]),
        segment_size=int(cfg["data"]["segment_size"]),
        hidden_dim=int(cfg["prefix_value"]["model"]["hidden_dim"]),
        max_segments=int(cfg["prefix_value"]["model"].get("max_segments", 128)),
    ).to(device)
    model.load_state_dict(checkpoint["prefix_value"])
    model.eval()
    temperature = float(checkpoint.get("calibration_temperature", 1.0))

    cache_path = Path(cfg["paths"]["test_feature_cache"])
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        rows = load_jsonl(cfg["paths"]["test_dataset"])
        extractor = VLLMFeatureExporter(
            model_name_or_path=cfg["inference"]["model_name_or_path"],
            max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
            parallel_size=parallel_size,
            gpu_memory_utilization=float(cfg["inference"].get("gpu_memory_utilization", 0.90)),
            reserve_training_gpu=True,
        )
        cache = precompute_feature_cache(
            rows, extractor,
            segment_size=int(cfg["data"]["segment_size"]),
            token_dim=int(cfg["data"]["instance_dim"]),
            top_k=int(cfg["inference"]["top_k_logprobs"]),
            max_tokens_per_batch=int(cfg["prefix_value"]["training"].get("max_tokens_per_batch", 131072)),
            device=device, description="Prefix test features",
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
    loader = DataLoader(
        IndexDataset(len(cache)), batch_size=int(cfg["prefix_value"]["training"]["batch_size"]),
        shuffle=False, num_workers=0, collate_fn=partial(terminal_collate, cache),
    )
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["features"].to(device), batch["token_mask"].to(device),
                batch["segment_mask"].to(device),
            )
            logits_all.append(output["terminal_logits"].cpu())
            targets_all.append(batch["target"])
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    probabilities = calibrated_probability(logits, temperature)
    predictions = (probabilities >= 0.5).float()
    ece = 0.0
    edges = torch.linspace(0.0, 1.0, 11)
    for idx in range(10):
        mask = (probabilities > edges[idx]) & (probabilities <= edges[idx + 1])
        if torch.any(mask):
            ece += float(mask.float().mean() * torch.abs(
                probabilities[mask].mean() - targets[mask].mean()
            ))
    return {
        "terminal_accuracy": float((predictions == targets).float().mean().item()),
        "terminal_auc": compute_auc(targets, logits),
        "terminal_brier": float(torch.mean((probabilities - targets) ** 2).item()),
        "terminal_ece": ece,
        "calibration_temperature": temperature,
        "validation_prefix_metrics": checkpoint.get("validation_metrics", {}),
        "n_test_trajectories": len(cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--output", default="results/prefix_value_metrics.json")
    args = parser.parse_args()
    metrics = evaluate(args.config, args.parallel_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
