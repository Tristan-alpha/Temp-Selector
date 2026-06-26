#!/usr/bin/env python3
"""Train layer-wise value lens probes from cached prefix-end hidden states."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mil.prefix_data import IndexDataset, build_ranking_pairs, continuation_collate
from mil.prefix_value import PrefixValueModel, calibrated_probability
from utils.jsonl import load_jsonl


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "hidden": "#5477C4",
    "hybrid": "#71B436",
    "logit": "#CC6F47",
    "pvm": "#464C55",
    "constant": "#A3A8B3",
    "low": "#CC6F47",
    "mid": "#B8A037",
    "high": "#5477C4",
    "correct": "#5477C4",
    "incorrect": "#CC6F47",
}


@dataclass(frozen=True)
class SplitData:
    name: str
    metadata: list[dict[str, Any]]
    hidden: torch.Tensor
    logit_features: torch.Tensor
    layer_ids: list[int]


class LinearValueProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class HybridValueProbe(nn.Module):
    def __init__(self, hidden_dim: int, extra_dim: int = 1):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.hidden_head = nn.Linear(hidden_dim, 1)
        self.extra_head = nn.Linear(extra_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        return (
            self.hidden_head(self.hidden_norm(hidden)).squeeze(-1)
            + self.extra_head(extra).squeeze(-1)
        )


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "axes.titlecolor": TOKENS["ink"],
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 9.5,
        "xtick.color": TOKENS["muted"],
        "ytick.color": TOKENS["muted"],
        "grid.color": TOKENS["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": TOKENS["surface"],
        "savefig.bbox": "tight",
    })


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_split_cache(cache_dir: Path, manifest: Mapping[str, Any], split: str) -> SplitData:
    metadata = read_jsonl(cache_dir / f"{split}_metadata.jsonl")
    chunks = [
        item for item in manifest.get("chunks", [])
        if str(item.get("split")) == split
    ]
    if not chunks:
        raise RuntimeError(f"cache manifest has no chunks for split={split}")
    chunks = sorted(chunks, key=lambda item: int(item["layer_ids"][0]))
    hidden_parts: list[torch.Tensor] = []
    logit_parts: list[torch.Tensor] = []
    layer_ids: list[int] = []
    for item in chunks:
        packed = torch.load(cache_dir / str(item["cache_file"]), map_location="cpu", weights_only=False)
        part_layers = [int(x) for x in packed["layer_ids"]]
        if layer_ids and part_layers[0] <= layer_ids[-1]:
            raise RuntimeError(f"non-increasing layer order in {item['cache_file']}")
        hidden = packed["prefix_hidden"]
        logits = packed["logit_features"]
        if hidden.shape[0] != len(metadata):
            raise RuntimeError(
                f"{item['cache_file']} has {hidden.shape[0]} rows but metadata has {len(metadata)}"
            )
        if logits.shape[:2] != hidden.shape[:2]:
            raise RuntimeError(f"logit feature shape mismatch in {item['cache_file']}")
        hidden_parts.append(hidden)
        logit_parts.append(logits)
        layer_ids.extend(part_layers)
    return SplitData(
        name=split,
        metadata=metadata,
        hidden=torch.cat(hidden_parts, dim=1),
        logit_features=torch.cat(logit_parts, dim=1),
        layer_ids=layer_ids,
    )


def split_problem_indices(
    metadata: Sequence[Mapping[str, Any]],
    dev_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    by_problem: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(metadata):
        by_problem[str(row["problem_id"])].append(idx)
    problems = sorted(by_problem)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(problems)
    n_dev = max(1, int(round(len(problems) * float(dev_fraction)))) if problems else 0
    dev_problems = set(problems[:n_dev])
    train_idx: list[int] = []
    dev_idx: list[int] = []
    for problem in sorted(by_problem):
        if problem in dev_problems:
            dev_idx.extend(by_problem[problem])
        else:
            train_idx.extend(by_problem[problem])
    if set(train_idx) & set(dev_idx):
        raise RuntimeError("train/dev index overlap detected")
    train_problems = {str(metadata[i]["problem_id"]) for i in train_idx}
    if train_problems & dev_problems:
        raise RuntimeError("train/dev problem_id overlap detected")
    return sorted(train_idx), sorted(dev_idx)


def _array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _int_array(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([int(row[key]) for row in rows], dtype=np.int64)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    n = int(values.size)
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size:
        return 0.0
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = float(np.sqrt(np.sum(vx * vx) * np.sum(vy * vy)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(vx * vy) / denom)


def expected_calibration_error(probabilities: Sequence[float], targets: Sequence[float],
                               n_bins: int = 10) -> float:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        if idx == 0:
            mask = (p >= edges[idx]) & (p <= edges[idx + 1])
        else:
            mask = (p > edges[idx]) & (p <= edges[idx + 1])
        if np.any(mask):
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return ece


def binomial_nll_from_probs(
    probabilities: Sequence[float],
    n_correct: Sequence[int],
    n_total: Sequence[int],
    eps: float = 1e-6,
) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), eps, 1.0 - eps)
    c = np.asarray(n_correct, dtype=np.float64).reshape(-1)
    t = np.maximum(np.asarray(n_total, dtype=np.float64).reshape(-1), 1.0)
    return -(c * np.log(p) + (t - c) * np.log1p(-p)) / t


def pair_accuracy(
    probabilities: Sequence[float],
    records: Sequence[Mapping[str, Any]],
    seed: int,
    max_pairs_per_problem: int = 64,
) -> tuple[float, int]:
    pairs = build_ranking_pairs(list(records), seed=int(seed), max_pairs_per_problem=max_pairs_per_problem)
    if not pairs:
        return 0.0, 0
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = _array(records, "target")
    correct = 0
    total = 0
    for a, b in pairs:
        if targets[a] == targets[b]:
            continue
        correct += int((probs[a] > probs[b]) == (targets[a] > targets[b]))
        total += 1
    return correct / max(1, total), total


def stage_metrics(
    probabilities: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    per_record_nll: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    by_stage: dict[str, list[int]] = defaultdict(list)
    targets = _array(records, "target")
    observed = _array(records, "observed_success_rate")
    for idx, row in enumerate(records):
        by_stage[str(row.get("prefix_stage", "unknown"))].append(idx)
    result: dict[str, dict[str, float | int]] = {}
    for stage, indices in sorted(by_stage.items()):
        idx = np.asarray(indices, dtype=np.int64)
        result[stage] = {
            "n_prefixes": int(idx.size),
            "brier": float(np.mean((probabilities[idx] - targets[idx]) ** 2)),
            "binomial_nll": float(np.mean(per_record_nll[idx])),
            "mean_phi": float(np.mean(probabilities[idx])),
            "observed_rate": float(np.mean(observed[idx])),
        }
    return result


def evaluate_probabilities(
    probabilities: Sequence[float],
    records: Sequence[Mapping[str, Any]],
    seed: int,
    label: str,
) -> dict[str, Any]:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-6, 1.0 - 1e-6)
    targets = _array(records, "target")
    observed = _array(records, "observed_success_rate")
    n_correct = _int_array(records, "n_correct")
    n_total = _int_array(records, "n_total")
    per_record_nll = binomial_nll_from_probs(probs, n_correct, n_total)
    p_acc, n_pairs = pair_accuracy(probs, records, seed=seed)
    constant_probability = float(np.clip(n_correct.sum() / max(1, n_total.sum()), 1e-6, 1.0 - 1e-6))
    result = {
        "label": label,
        "brier": float(np.mean((probs - targets) ** 2)),
        "ece": expected_calibration_error(probs, targets),
        "binomial_nll": float(np.mean(per_record_nll)),
        "spearman": spearman(probs, observed),
        "pair_accuracy": float(p_acc),
        "n_pairs": int(n_pairs),
        "n_prefixes": int(len(records)),
        "constant_mean_probability": constant_probability,
        "n_total_distribution": {
            str(int(value)): int(np.sum(n_total == value))
            for value in np.unique(n_total)
        },
        "stage_metrics": stage_metrics(probs, records, per_record_nll),
    }
    if len(probs) > 0:
        k = max(1, len(probs) // 4)
        order = np.argsort(probs, kind="mergesort")
        bottom = order[:k]
        top = order[-k:]
        result["phi_quartiles"] = {
            "bottom_n": int(bottom.size),
            "top_n": int(top.size),
            "bottom_mean_phi": float(np.mean(probs[bottom])),
            "top_mean_phi": float(np.mean(probs[top])),
            "bottom_observed_rate": float(np.mean(observed[bottom])),
            "top_observed_rate": float(np.mean(observed[top])),
            "observed_rate_delta": float(np.mean(observed[top]) - np.mean(observed[bottom])),
        }
    return result


def constant_metrics(records: Sequence[Mapping[str, Any]], seed: int) -> dict[str, Any]:
    n_correct = _int_array(records, "n_correct")
    n_total = _int_array(records, "n_total")
    p = float(np.clip(n_correct.sum() / max(1, n_total.sum()), 1e-6, 1.0 - 1e-6))
    return evaluate_probabilities(np.full(len(records), p), records, seed=seed, label="constant")


def logit_scalar(probabilities: Sequence[float]) -> torch.Tensor:
    p = torch.as_tensor(probabilities, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(p).view(-1, 1)


def binomial_nll_logits(logits: torch.Tensor, n_correct: torch.Tensor, n_total: torch.Tensor) -> torch.Tensor:
    total = n_total.to(logits.dtype).clamp_min(1.0)
    correct = n_correct.to(logits.dtype)
    loss = -(correct * torch.nn.functional.logsigmoid(logits) +
             (total - correct) * torch.nn.functional.logsigmoid(-logits)) / total
    return loss.mean()


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    features: torch.Tensor,
    extra: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, features.shape[0], max(1, int(batch_size))):
        end = min(start + int(batch_size), features.shape[0])
        x = features[start:end].float().to(device)
        if extra is None:
            logits = model(x)
        else:
            logits = model(x, extra[start:end].float().to(device))
        outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def fit_temperature_for_logits(
    logits: torch.Tensor,
    n_correct: torch.Tensor,
    n_total: torch.Tensor,
) -> float:
    if logits.numel() == 0:
        return 1.0
    log_temperature = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-4, 100.0)
        loss = binomial_nll_logits(logits.float() / temperature, n_correct.float(), n_total.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-4, 100.0).item())


def train_probe(
    *,
    family: str,
    layer_id: int,
    train_features: torch.Tensor,
    dev_features: torch.Tensor,
    test_features: torch.Tensor,
    train_extra: torch.Tensor | None,
    dev_extra: torch.Tensor | None,
    test_extra: torch.Tensor | None,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    if family == "hybrid":
        model: nn.Module = HybridValueProbe(int(train_features.shape[1]), int(train_extra.shape[1]))
    else:
        model = LinearValueProbe(int(train_features.shape[1]))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    n_correct_train = torch.as_tensor(_int_array(train_records, "n_correct"), dtype=torch.float32)
    n_total_train = torch.as_tensor(_int_array(train_records, "n_total"), dtype=torch.float32)
    n_correct_dev = torch.as_tensor(_int_array(dev_records, "n_correct"), dtype=torch.float32)
    n_total_dev = torch.as_tensor(_int_array(dev_records, "n_total"), dtype=torch.float32)
    family_offsets = {"hidden": 17, "hybrid": 31, "logit": 47}
    rng = np.random.default_rng(
        int(args.seed) + int(layer_id) * 997 + family_offsets.get(family, 0)
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_nll = float("inf")
    best_epoch = 0
    patience = 0
    indices = np.arange(len(train_records))
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        rng.shuffle(indices)
        for start in range(0, len(indices), int(args.batch_size)):
            batch_idx = indices[start:start + int(args.batch_size)]
            idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
            x = train_features[idx_t].float().to(device)
            c = n_correct_train[idx_t].to(device)
            t = n_total_train[idx_t].to(device)
            if family == "hybrid":
                logits = model(x, train_extra[idx_t].float().to(device))
            else:
                logits = model(x)
            loss = binomial_nll_logits(logits, c, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        dev_logits = predict_logits(model, dev_features, dev_extra, args.eval_batch_size, device)
        dev_nll = float(binomial_nll_logits(dev_logits, n_correct_dev, n_total_dev).item())
        if dev_nll < best_dev_nll - 1e-7:
            best_dev_nll = dev_nll
            best_epoch = epoch
            patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= int(args.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    dev_logits = predict_logits(model, dev_features, dev_extra, args.eval_batch_size, device)
    test_logits = predict_logits(model, test_features, test_extra, args.eval_batch_size, device)
    temperature = fit_temperature_for_logits(dev_logits, n_correct_dev, n_total_dev)
    dev_probs = torch.sigmoid(dev_logits / temperature).numpy()
    test_probs = torch.sigmoid(test_logits / temperature).numpy()
    dev_metrics = evaluate_probabilities(dev_probs, dev_records, seed=int(args.seed), label=f"{family}_dev")
    test_metrics = evaluate_probabilities(test_probs, test_records, seed=int(args.seed), label=f"{family}_test")
    row: dict[str, Any] = {
        "family": family,
        "layer_id": int(layer_id),
        "best_epoch": int(best_epoch),
        "best_dev_uncalibrated_binomial_nll": best_dev_nll,
        "calibration_temperature": temperature,
    }
    for prefix, metrics in (("dev", dev_metrics), ("test", test_metrics)):
        for key in ("brier", "binomial_nll", "spearman", "ece", "pair_accuracy", "n_pairs", "n_prefixes"):
            row[f"{prefix}_{key}"] = metrics[key]
    return row, test_probs


def assign_tertiles(values: Sequence[float]) -> list[str]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    cuts = np.quantile(arr, [1 / 3, 2 / 3])
    labels: list[str] = []
    for value in arr:
        if value <= cuts[0]:
            labels.append("low")
        elif value <= cuts[1]:
            labels.append("mid")
        else:
            labels.append("high")
    return labels


def _config_path_from_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> Path | None:
    value = manifest.get("config")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = cache_dir / path
    return path


def _feature_cache_path(cfg: Mapping[str, Any], split: str) -> Path:
    return Path(str(cfg["paths"][f"{split}_feature_cache"]))


def _continuation_path(cfg: Mapping[str, Any], split: str) -> Path:
    return Path(str(cfg["paths"][f"{split}_continuations"]))


@torch.no_grad()
def score_pvm_split(
    *,
    cfg: Mapping[str, Any],
    split: str,
    pvm_checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    records = load_jsonl(str(_continuation_path(cfg, split)))
    cache = torch.load(_feature_cache_path(cfg, split), map_location="cpu", weights_only=False)
    cache_by_id = {str(entry["sample_id"]): entry for entry in cache}
    missing = sorted({
        str(record["source_sample_id"])
        for record in records
        if str(record["source_sample_id"]) not in cache_by_id
    })
    if missing:
        raise RuntimeError(f"PVM feature cache for split={split} is missing {len(missing)} source ids")
    checkpoint = torch.load(pvm_checkpoint_path, map_location=device, weights_only=False)
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
        IndexDataset(len(records)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(continuation_collate, cache_by_id, records),
    )
    probs: list[float] = []
    for batch in loader:
        prompt_hidden = batch["prompt_hidden"].to(device) if "prompt_hidden" in batch else None
        logits = model(
            batch["features"].to(device),
            batch["token_mask"].to(device),
            batch["segment_mask"].to(device),
            prompt_hidden=prompt_hidden,
        )["terminal_logits"]
        phi = calibrated_probability(logits, calibration_temperature)
        probs.extend(float(item) for item in phi.detach().cpu().tolist())
    del cache, cache_by_id, model, checkpoint
    gc.collect()
    return np.asarray(probs, dtype=np.float64)


def load_or_score_pvm(
    *,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    train_data: SplitData,
    val_data: SplitData,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if args.skip_pvm_baseline:
        return None, None, {"skipped": True}
    config_path = Path(args.config) if args.config else _config_path_from_manifest(cache_dir, manifest)
    if config_path is None:
        raise RuntimeError("--config is required when cache manifest does not contain a config path")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    checkpoint_path = Path(args.pvm_checkpoint or cfg["paths"]["prefix_value_ckpt"])
    device = torch.device(args.pvm_device)
    train_scores = score_pvm_split(
        cfg=cfg,
        split="train",
        pvm_checkpoint_path=checkpoint_path,
        device=device,
        batch_size=int(args.pvm_batch_size),
    )
    val_scores = score_pvm_split(
        cfg=cfg,
        split="val",
        pvm_checkpoint_path=checkpoint_path,
        device=device,
        batch_size=int(args.pvm_batch_size),
    )
    if len(train_scores) != len(train_data.metadata) or len(val_scores) != len(val_data.metadata):
        raise RuntimeError("PVM score lengths do not match layer cache metadata")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    meta = {
        "skipped": False,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "calibration_temperature": float(checkpoint.get("calibration_temperature", 1.0)),
        "checkpoint_validation_metrics": checkpoint.get("validation_metrics", {}),
    }
    return train_scores, val_scores, meta


def baseline_alignment(
    computed: Mapping[str, Any],
    reference: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    tolerances = {
        "brier": float(args.baseline_brier_tolerance),
        "binomial_nll": float(args.baseline_nll_tolerance),
        "spearman": float(args.baseline_rank_tolerance),
        "pair_accuracy": float(args.baseline_rank_tolerance),
    }
    passed = True
    for key, tol in tolerances.items():
        if key not in reference:
            continue
        delta = abs(float(computed[key]) - float(reference[key]))
        ok = delta <= tol
        checks[key] = {
            "computed": float(computed[key]),
            "reference": float(reference[key]),
            "abs_delta": delta,
            "tolerance": tol,
            "passed": ok,
        }
        passed = passed and ok
    return {"passed": passed, "checks": checks}


def subset_rows(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> list[dict[str, Any]]:
    return [dict(rows[int(i)]) for i in indices]


def subset_tensor(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return tensor[torch.as_tensor(indices, dtype=torch.long)]


def plot_layer_metrics(path: Path, metric_rows: Sequence[Mapping[str, Any]], pvm_metrics: Mapping[str, Any] | None) -> None:
    if not metric_rows:
        return
    families = [family for family in ("hidden", "hybrid", "logit") if any(row["family"] == family for row in metric_rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for family in families:
        rows = sorted([row for row in metric_rows if row["family"] == family], key=lambda row: int(row["layer_id"]))
        x = [int(row["layer_id"]) for row in rows]
        axes[0].plot(x, [float(row["test_brier"]) for row in rows], marker="o", linewidth=1.5,
                     label=family, color=COLORS.get(family))
        axes[1].plot(x, [float(row["test_spearman"]) for row in rows], marker="o", linewidth=1.5,
                     label=family, color=COLORS.get(family))
    if pvm_metrics:
        axes[0].axhline(float(pvm_metrics["brier"]), color=COLORS["pvm"], linestyle="--", label="PVM")
        axes[1].axhline(float(pvm_metrics["spearman"]), color=COLORS["pvm"], linestyle="--", label="PVM")
    axes[0].set_title("Held-out Brier by layer")
    axes[1].set_title("Held-out Spearman by layer")
    axes[0].set_ylabel("Brier (lower is better)")
    axes[1].set_ylabel("Spearman (higher is better)")
    for ax in axes:
        ax.set_xlabel("Layer id")
        ax.grid(True, axis="y")
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_hybrid_delta(path: Path, metric_rows: Sequence[Mapping[str, Any]], pvm_metrics: Mapping[str, Any] | None) -> None:
    if not pvm_metrics:
        return
    rows = sorted([row for row in metric_rows if row["family"] == "hybrid"], key=lambda row: int(row["layer_id"]))
    if not rows:
        return
    x = [int(row["layer_id"]) for row in rows]
    brier_gain = [float(pvm_metrics["brier"]) - float(row["test_brier"]) for row in rows]
    spearman_gain = [float(row["test_spearman"]) - float(pvm_metrics["spearman"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharex=True)
    axes[0].axhline(0.0, color=COLORS["pvm"], linewidth=1)
    axes[0].plot(x, brier_gain, marker="o", color=COLORS["hybrid"])
    axes[0].set_title("Hybrid Brier gain vs PVM")
    axes[0].set_ylabel("PVM Brier - hybrid Brier")
    axes[1].axhline(0.0, color=COLORS["pvm"], linewidth=1)
    axes[1].plot(x, spearman_gain, marker="o", color=COLORS["hybrid"])
    axes[1].set_title("Hybrid Spearman gain vs PVM")
    axes[1].set_ylabel("Hybrid Spearman - PVM Spearman")
    for ax in axes:
        ax.set_xlabel("Layer id")
        ax.grid(True, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_best_vs_final(path: Path, metric_rows: Sequence[Mapping[str, Any]], pvm_metrics: Mapping[str, Any] | None,
                       final_layer_id: int) -> None:
    candidates: list[tuple[str, float, float]] = []
    if pvm_metrics:
        candidates.append(("PVM", float(pvm_metrics["brier"]), float(pvm_metrics["spearman"])))
    for family in ("hidden", "hybrid", "logit"):
        rows = [row for row in metric_rows if row["family"] == family]
        if not rows:
            continue
        best = min(rows, key=lambda row: float(row["test_brier"]))
        final = next((row for row in rows if int(row["layer_id"]) == int(final_layer_id)), None)
        candidates.append((f"best {family}", float(best["test_brier"]), float(best["test_spearman"])))
        if final is not None:
            candidates.append((f"final {family}", float(final["test_brier"]), float(final["test_spearman"])))
    if not candidates:
        return
    labels = [item[0] for item in candidates]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, [item[1] for item in candidates], color=COLORS["hidden"])
    axes[0].set_title("Brier comparison")
    axes[0].set_ylabel("Brier")
    axes[1].bar(labels, [item[2] for item in candidates], color=COLORS["hybrid"])
    axes[1].set_title("Spearman comparison")
    axes[1].set_ylabel("Spearman")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_value_trajectories(
    path: Path,
    layer_ids: Sequence[int],
    predictions_by_layer: np.ndarray | None,
    records: Sequence[Mapping[str, Any]],
    pvm_scores: np.ndarray | None,
) -> None:
    if predictions_by_layer is None or predictions_by_layer.size == 0:
        return
    groups: list[tuple[str, np.ndarray, str]] = []
    observed = _array(records, "observed_success_rate")
    groups.append(("observed >= 0.5", observed >= 0.5, COLORS["correct"]))
    groups.append(("observed < 0.5", observed < 0.5, COLORS["incorrect"]))
    if pvm_scores is not None:
        buckets = assign_tertiles(pvm_scores)
        for bucket in ("low", "mid", "high"):
            mask = np.asarray([item == bucket for item in buckets], dtype=bool)
            groups.append((f"PVM {bucket}", mask, COLORS[bucket]))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = [int(layer) for layer in layer_ids]
    for label, mask, color in groups:
        if not np.any(mask):
            continue
        ax.plot(x, predictions_by_layer[mask].mean(axis=0), marker="o", linewidth=1.6,
                label=f"{label} (n={int(mask.sum())})", color=color)
    ax.set_title("Hidden-probe value trajectories")
    ax.set_xlabel("Layer id")
    ax.set_ylabel("Mean predicted continuation value")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize_results(
    metric_rows: Sequence[Mapping[str, Any]],
    pvm_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_probe_rows": len(metric_rows),
        "pvm_baseline": dict(pvm_metrics) if pvm_metrics else None,
    }
    for family in ("hidden", "hybrid", "logit"):
        rows = [row for row in metric_rows if row["family"] == family]
        if not rows:
            continue
        best_brier = min(rows, key=lambda row: float(row["test_brier"]))
        best_spearman = max(rows, key=lambda row: float(row["test_spearman"]))
        item = {
            "best_brier_layer": int(best_brier["layer_id"]),
            "best_brier": float(best_brier["test_brier"]),
            "best_spearman_layer": int(best_spearman["layer_id"]),
            "best_spearman": float(best_spearman["test_spearman"]),
            "best_pair_accuracy": float(max(float(row["test_pair_accuracy"]) for row in rows)),
        }
        if pvm_metrics:
            item["best_brier_gain_vs_pvm"] = float(pvm_metrics["brier"]) - item["best_brier"]
            item["best_spearman_gain_vs_pvm"] = item["best_spearman"] - float(pvm_metrics["spearman"])
            item["promising_by_plan"] = (
                item["best_brier_gain_vs_pvm"] >= 0.01
                or item["best_spearman_gain_vs_pvm"] >= 0.05
            )
        summary[family] = item
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--pvm-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-pvm-baseline", action="store_true")
    parser.add_argument("--probe-families", default="hidden,hybrid,logit")
    parser.add_argument("--max-layers", type=int, default=0,
                        help="Use only the first N cached layers; intended for smoke tests.")
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--pvm-device", default="cpu")
    parser.add_argument("--pvm-batch-size", type=int, default=32)
    parser.add_argument("--baseline-brier-tolerance", type=float, default=0.02)
    parser.add_argument("--baseline-nll-tolerance", type=float, default=0.05)
    parser.add_argument("--baseline-rank-tolerance", type=float, default=0.05)
    return parser.parse_args()


def resolve_train_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--train-device=cuda requested but CUDA is unavailable")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.cache_dir / "manifest.json")
    train_data = load_split_cache(args.cache_dir, manifest, "train")
    val_data = load_split_cache(args.cache_dir, manifest, "val")
    if train_data.layer_ids != val_data.layer_ids:
        raise RuntimeError("train and val layer ids differ")
    if int(args.max_layers) > 0:
        keep = int(args.max_layers)
        train_data = SplitData(
            train_data.name, train_data.metadata,
            train_data.hidden[:, :keep], train_data.logit_features[:, :keep],
            train_data.layer_ids[:keep],
        )
        val_data = SplitData(
            val_data.name, val_data.metadata,
            val_data.hidden[:, :keep], val_data.logit_features[:, :keep],
            val_data.layer_ids[:keep],
        )

    train_problem_ids = {str(row["problem_id"]) for row in train_data.metadata}
    val_problem_ids = {str(row["problem_id"]) for row in val_data.metadata}
    problem_overlap = len(train_problem_ids & val_problem_ids)
    if problem_overlap:
        raise RuntimeError(f"train/val problem_id overlap detected: {problem_overlap}")
    subtrain_idx, dev_idx = split_problem_indices(train_data.metadata, args.dev_fraction, args.seed)
    subtrain_rows = subset_rows(train_data.metadata, subtrain_idx)
    dev_rows = subset_rows(train_data.metadata, dev_idx)
    test_rows = val_data.metadata

    pvm_train, pvm_val, pvm_meta = load_or_score_pvm(
        cache_dir=args.cache_dir,
        manifest=manifest,
        train_data=train_data,
        val_data=val_data,
        args=args,
    )
    pvm_metrics: dict[str, Any] | None = None
    pvm_alignment: dict[str, Any] | None = None
    if pvm_train is not None and pvm_val is not None:
        for row, score in zip(train_data.metadata, pvm_train):
            row["pvm_phi"] = float(score)
        for row, score in zip(val_data.metadata, pvm_val):
            row["pvm_phi"] = float(score)
        write_csv(args.output_dir / "pvm_prefix_scores_train.csv", [
            {**row, "pvm_phi": float(score)}
            for row, score in zip(train_data.metadata, pvm_train)
        ])
        write_csv(args.output_dir / "pvm_prefix_scores_val.csv", [
            {**row, "pvm_phi": float(score)}
            for row, score in zip(val_data.metadata, pvm_val)
        ])
        pvm_metrics = evaluate_probabilities(pvm_val, val_data.metadata, seed=args.seed, label="pvm")
        pvm_alignment = baseline_alignment(
            pvm_metrics,
            pvm_meta.get("checkpoint_validation_metrics", {}),
            args,
        )
        if pvm_alignment["checks"] and not pvm_alignment["passed"]:
            write_json(args.output_dir / "baseline_alignment_failure.json", {
                "pvm_metrics": pvm_metrics,
                "pvm_meta": pvm_meta,
                "alignment": pvm_alignment,
            })
            raise RuntimeError("computed PVM baseline did not match checkpoint validation metadata")
    else:
        for rows in (train_data.metadata, val_data.metadata):
            if rows and "pvm_phi" not in rows[0]:
                pvm_meta["hybrid_available"] = False

    constant = constant_metrics(val_data.metadata, seed=args.seed)
    device = resolve_train_device(args.train_device)
    families = [item.strip() for item in args.probe_families.split(",") if item.strip()]
    if pvm_train is None and "hybrid" in families and "pvm_phi" not in train_data.metadata[0]:
        families = [family for family in families if family != "hybrid"]

    train_hidden = subset_tensor(train_data.hidden, subtrain_idx)
    dev_hidden = subset_tensor(train_data.hidden, dev_idx)
    train_logit = subset_tensor(train_data.logit_features, subtrain_idx)
    dev_logit = subset_tensor(train_data.logit_features, dev_idx)
    test_hidden = val_data.hidden
    test_logit = val_data.logit_features

    pvm_train_extra = None
    pvm_dev_extra = None
    pvm_test_extra = None
    if "hybrid" in families:
        full_train_scores = np.asarray([float(row["pvm_phi"]) for row in train_data.metadata])
        pvm_train_extra = logit_scalar(full_train_scores[subtrain_idx])
        pvm_dev_extra = logit_scalar(full_train_scores[dev_idx])
        pvm_test_extra = logit_scalar(np.asarray([float(row["pvm_phi"]) for row in val_data.metadata]))

    metric_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    layer_ids = train_data.layer_ids
    for layer_pos, layer_id in enumerate(layer_ids):
        print(f"[value-lens] layer {layer_id} ({layer_pos + 1}/{len(layer_ids)})", flush=True)
        if "hidden" in families:
            row, probs = train_probe(
                family="hidden",
                layer_id=layer_id,
                train_features=train_hidden[:, layer_pos, :],
                dev_features=dev_hidden[:, layer_pos, :],
                test_features=test_hidden[:, layer_pos, :],
                train_extra=None,
                dev_extra=None,
                test_extra=None,
                train_records=subtrain_rows,
                dev_records=dev_rows,
                test_records=test_rows,
                args=args,
                device=device,
            )
            metric_rows.append(row)
            predictions["hidden"].append(probs)
        if "logit" in families:
            row, probs = train_probe(
                family="logit",
                layer_id=layer_id,
                train_features=train_logit[:, layer_pos, :],
                dev_features=dev_logit[:, layer_pos, :],
                test_features=test_logit[:, layer_pos, :],
                train_extra=None,
                dev_extra=None,
                test_extra=None,
                train_records=subtrain_rows,
                dev_records=dev_rows,
                test_records=test_rows,
                args=args,
                device=device,
            )
            metric_rows.append(row)
            predictions["logit"].append(probs)
        if "hybrid" in families:
            row, probs = train_probe(
                family="hybrid",
                layer_id=layer_id,
                train_features=train_hidden[:, layer_pos, :],
                dev_features=dev_hidden[:, layer_pos, :],
                test_features=test_hidden[:, layer_pos, :],
                train_extra=pvm_train_extra,
                dev_extra=pvm_dev_extra,
                test_extra=pvm_test_extra,
                train_records=subtrain_rows,
                dev_records=dev_rows,
                test_records=test_rows,
                args=args,
                device=device,
            )
            metric_rows.append(row)
            predictions["hybrid"].append(probs)

    hidden_predictions = None
    if predictions.get("hidden"):
        hidden_predictions = np.stack(predictions["hidden"], axis=1)
    write_csv(args.output_dir / "metrics_by_layer.csv", metric_rows)
    write_json(args.output_dir / "baseline_metrics.json", {
        "constant": constant,
        "pvm": pvm_metrics,
        "pvm_meta": pvm_meta,
        "pvm_alignment": pvm_alignment,
    })
    summary = summarize_results(metric_rows, pvm_metrics)
    summary.update({
        "constant_baseline": constant,
        "problem_split": {
            "train_problem_ids": len(train_problem_ids),
            "val_problem_ids": len(val_problem_ids),
            "train_val_problem_overlap": problem_overlap,
            "subtrain_prefixes": len(subtrain_rows),
            "dev_prefixes": len(dev_rows),
            "test_prefixes": len(test_rows),
        },
    })
    write_json(args.output_dir / "summary.json", summary)
    plot_layer_metrics(args.output_dir / "fig_layer_metrics.png", metric_rows, pvm_metrics)
    plot_hybrid_delta(args.output_dir / "fig_hybrid_delta_vs_pvm.png", metric_rows, pvm_metrics)
    plot_best_vs_final(args.output_dir / "fig_best_vs_final.png", metric_rows, pvm_metrics, final_layer_id=layer_ids[-1])
    plot_value_trajectories(
        args.output_dir / "fig_value_trajectories.png",
        layer_ids,
        hidden_predictions,
        val_data.metadata,
        pvm_val,
    )
    write_json(args.output_dir / "run_manifest.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/train_layer_value_lens.py",
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "runtime": {
            "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.time() - started,
        },
        "args": vars(args) | {
            "cache_dir": str(args.cache_dir),
            "output_dir": str(args.output_dir),
            "config": str(args.config) if args.config else None,
            "pvm_checkpoint": str(args.pvm_checkpoint) if args.pvm_checkpoint else None,
        },
        "outputs": {
            "metrics_by_layer": str(args.output_dir / "metrics_by_layer.csv"),
            "baseline_metrics": str(args.output_dir / "baseline_metrics.json"),
            "summary": str(args.output_dir / "summary.json"),
            "fig_layer_metrics": str(args.output_dir / "fig_layer_metrics.png"),
            "fig_hybrid_delta_vs_pvm": str(args.output_dir / "fig_hybrid_delta_vs_pvm.png"),
            "fig_best_vs_final": str(args.output_dir / "fig_best_vs_final.png"),
            "fig_value_trajectories": str(args.output_dir / "fig_value_trajectories.png"),
        },
    })
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "n_metric_rows": len(metric_rows),
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
