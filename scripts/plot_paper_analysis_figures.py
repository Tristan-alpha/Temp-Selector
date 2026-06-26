#!/usr/bin/env python3
"""Build paper analysis figures from existing tf-mil artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mil.prefix_data import IndexDataset, continuation_collate, record_per_temperature_stats
from mil.prefix_value import PrefixValueModel, calibrated_probability
from utils.calibration import (
    answer_entropy,
    binary_nll,
    brier_score,
    expected_calibration_error,
)
from utils.jsonl import load_jsonl, sample_prefix


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#5477C4",
    "blue_light": "#CEDFFE",
    "gold": "#B8A037",
    "gold_light": "#FFEA8F",
    "orange": "#CC6F47",
    "orange_light": "#FFBDA1",
    "olive": "#71B436",
    "olive_light": "#BEEB96",
    "pink": "#BD569B",
    "pink_light": "#F5BACC",
    "neutral": "#7A828F",
    "neutral_dark": "#464C55",
}

FONT_FAMILY = ["DejaVu Sans", "sans-serif"]
MONO_FONT_FAMILY = ["DejaVu Sans Mono", "monospace"]

BUCKET_ORDER = ["low", "mid", "high"]
BUCKET_LABELS = {
    "low": "Low PVM",
    "mid": "Mid PVM",
    "high": "High PVM",
}
BUCKET_COLORS = {
    "low": COLORS["orange"],
    "mid": COLORS["gold"],
    "high": COLORS["blue"],
}
SUCCESS_COLORS = {
    "correct": COLORS["blue"],
    "incorrect": COLORS["orange"],
}


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "axes.titlecolor": TOKENS["ink"],
        "font.family": FONT_FAMILY,
        "font.size": 9.5,
        "xtick.color": TOKENS["muted"],
        "ytick.color": TOKENS["muted"],
        "grid.color": TOKENS["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": TOKENS["surface"],
        "savefig.bbox": "tight",
    })


def add_figure_header(fig: plt.Figure, title: str, subtitle: str,
                      title_width: int = 92, subtitle_width: int = 122) -> None:
    title_wrapped = textwrap.fill(title.strip(), width=title_width, break_long_words=False)
    subtitle_wrapped = textwrap.fill(subtitle.strip(), width=subtitle_width, break_long_words=False)
    fig.text(0.06, 0.975, title_wrapped, ha="left", va="top",
             fontsize=16, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.06, 0.93, subtitle_wrapped, ha="left", va="top",
             fontsize=10, color=TOKENS["muted"])


def style_axis(ax: plt.Axes, *, percent: bool = False, y_limits: Tuple[float, float] | None = None) -> None:
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    if percent:
        ax.yaxis.set_major_formatter(lambda y, _: f"{100.0 * y:.0f}%")
    if y_limits is not None:
        ax.set_ylim(*y_limits)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> List[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220)
        paths.append(path)
    plt.close(fig)
    return paths


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def second_count(counts: Sequence[Tuple[Any, int]]) -> int:
    return int(counts[1][1]) if len(counts) > 1 else 0


def vote_margin_from_answers(answers: Sequence[Any]) -> float:
    if not answers:
        return 0.0
    counts = Counter(str(answer) for answer in answers).most_common()
    top = int(counts[0][1]) if counts else 0
    return float(top - second_count(counts)) / max(1, len(answers))


def row_answer(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping):
        answer = metadata.get("extracted_answer")
        if answer:
            return str(answer)
    return "<NO_ANSWER>"


def row_correct(row: Mapping[str, Any]) -> int:
    if "individual_label" in row:
        return 1 - int(row["individual_label"])
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping) and "individual_correct" in metadata:
        return int(bool(metadata["individual_correct"]))
    return 0


def row_vote_index(row: Mapping[str, Any]) -> int:
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping) and "vote_id" in metadata:
        return int(metadata["vote_id"])
    sample_id = str(row.get("sample_id", ""))
    if "_v" in sample_id:
        try:
            return int(sample_id.rsplit("_v", 1)[1])
        except ValueError:
            return 0
    return 0


def group_temperature_votes(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, float], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        pid = sample_prefix(str(row.get("sample_id", "")))
        grouped[(pid, float(row.get("temperature", 0.0)))].append(row)
    return dict(grouped)


def majority_summary_for_votes(votes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(votes, key=row_vote_index)
    answers = [row_answer(row) for row in ordered]
    counts = Counter(answers).most_common()
    top_answer, top_count = counts[0] if counts else ("<NO_ANSWER>", 0)
    individual_correct = [row_correct(row) for row in ordered]
    token_counts = [len(row.get("token_ids", [])) for row in ordered]
    gold = ordered[0].get("metadata", {}).get("gold_answer", "") if ordered else ""
    majority_correct = 0
    if ordered:
        # Dataset labels already encode majority correctness for each fixed-T group.
        majority_correct = 1 - int(ordered[0].get("voting_label", 1))
    return {
        "problem_id": sample_prefix(str(ordered[0].get("sample_id", ""))) if ordered else "",
        "temperature": float(ordered[0].get("temperature", 0.0)) if ordered else 0.0,
        "gold_answer": str(gold),
        "num_votes": len(ordered),
        "top_answer": str(top_answer),
        "majority_count": int(top_count),
        "majority_correct": int(majority_correct),
        "sc_confidence": float(top_count) / max(1, len(ordered)),
        "answer_entropy": answer_entropy(answers),
        "vote_margin": vote_margin_from_answers(answers),
        "individual_correct": individual_correct,
        "token_counts": token_counts,
        "total_tokens": int(sum(token_counts)),
    }


def temperature_landscape_table(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = group_temperature_votes(rows)
    summaries = [
        majority_summary_for_votes(votes)
        for _, votes in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    detail_rows = []
    for item in summaries:
        detail_rows.append({
            "problem_id": item["problem_id"],
            "temperature": item["temperature"],
            "majority_correct": item["majority_correct"],
            "sc_confidence": item["sc_confidence"],
            "answer_entropy": item["answer_entropy"],
            "vote_margin": item["vote_margin"],
            "num_votes": item["num_votes"],
            "total_tokens": item["total_tokens"],
        })

    by_temp: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        by_temp[float(item["temperature"])].append(item)

    rows_out: List[Dict[str, Any]] = []
    for temp in sorted(by_temp):
        items = by_temp[temp]
        confidences = [float(item["sc_confidence"]) for item in items]
        majority_correct = [int(item["majority_correct"]) for item in items]
        individual_correct = [
            int(value)
            for item in items
            for value in item["individual_correct"]
        ]
        token_counts = [
            int(value)
            for item in items
            for value in item["token_counts"]
        ]
        vote_counts = [int(item["num_votes"]) for item in items]
        rows_out.append({
            "temperature": temp,
            "n_prompts": len(items),
            "num_votes": int(vote_counts[0]) if len(set(vote_counts)) == 1 else "",
            "n_votes": len(individual_correct),
            "majority_accuracy": float(np.mean(majority_correct)) if majority_correct else 0.0,
            "pass_at_1_accuracy": float(np.mean(individual_correct)) if individual_correct else 0.0,
            "ece": expected_calibration_error(confidences, majority_correct),
            "brier": brier_score(confidences, majority_correct),
            "nll": binary_nll(confidences, majority_correct),
            "sc_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "answer_entropy": float(np.mean([float(item["answer_entropy"]) for item in items])),
            "vote_margin": float(np.mean([float(item["vote_margin"]) for item in items])),
            "mean_tokens_per_vote": float(np.mean(token_counts)) if token_counts else 0.0,
            "median_tokens_per_vote": float(statistics.median(token_counts)) if token_counts else 0.0,
        })
    return rows_out, detail_rows


def assign_tertiles(values: Sequence[float]) -> Tuple[List[str], Tuple[float, float]]:
    if not values:
        return [], (0.0, 0.0)
    order = np.argsort(np.asarray(values, dtype=float), kind="mergesort")
    labels = [""] * len(values)
    for rank, index in enumerate(order):
        bucket = BUCKET_ORDER[min(2, int(rank * 3 / len(values)))]
        labels[int(index)] = bucket
    sorted_values = np.asarray(values, dtype=float)[order]
    cut_low = float(sorted_values[max(0, len(values) // 3 - 1)])
    cut_high = float(sorted_values[max(0, (2 * len(values)) // 3 - 1)])
    return labels, (cut_low, cut_high)


def bootstrap_mean_ci(values: Sequence[float], rng: np.random.Generator,
                      n_bootstrap: int = 1000,
                      confidence: float = 0.95) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1 or n_bootstrap <= 0:
        return mean, mean, mean
    samples = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return mean, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def load_pvm_inputs(config_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    records = load_jsonl(str(cfg["paths"]["val_continuations"]))
    cache = torch.load(cfg["paths"]["val_feature_cache"], map_location="cpu", weights_only=False)
    cache_by_id = {str(entry["sample_id"]): entry for entry in cache}
    missing = sorted({
        str(record["source_sample_id"])
        for record in records
        if str(record["source_sample_id"]) not in cache_by_id
    })
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"feature cache missing {len(missing)} source ids: {preview}")
    checkpoint = torch.load(cfg["paths"]["prefix_value_ckpt"], map_location="cpu", weights_only=False)
    return cfg, records, cache_by_id, checkpoint


def score_pvm_prefixes(config_path: Path, device_name: str = "cpu",
                       batch_size: int | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg, records, cache_by_id, checkpoint = load_pvm_inputs(config_path)
    device = torch.device(device_name)
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
    effective_batch = int(batch_size or cfg["prefix_value"]["training"].get("batch_size", 32))
    loader = DataLoader(
        IndexDataset(len(records)),
        batch_size=effective_batch,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(continuation_collate, cache_by_id, records),
    )
    phi_values: List[float] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            token_mask = batch["token_mask"].to(device)
            segment_mask = batch["segment_mask"].to(device)
            prompt_hidden = (
                batch["prompt_hidden"].to(device)
                if "prompt_hidden" in batch else None
            )
            logits = model(
                features, token_mask, segment_mask, prompt_hidden=prompt_hidden,
            )["terminal_logits"]
            probs = calibrated_probability(logits, calibration_temperature)
            phi_values.extend(float(value) for value in probs.detach().cpu().tolist())

    labels, cuts = assign_tertiles(phi_values)
    scored: List[Dict[str, Any]] = []
    for idx, (record, phi, bucket) in enumerate(zip(records, phi_values, labels)):
        stats = record_per_temperature_stats(record)
        observed_success = float(record["n_correct"]) / max(1.0, float(record["n_total"]))
        scored.append({
            "record_index": idx,
            "problem_id": str(record.get("problem_id", "")),
            "source_sample_id": str(record["source_sample_id"]),
            "prefix_segments": int(record["prefix_segments"]),
            "prefix_stage": str(record.get("prefix_stage", "unknown")),
            "source_individual_label": int(record.get("source_individual_label", -1)),
            "pvm_phi": float(phi),
            "pvm_bucket": bucket,
            "observed_success_rate": observed_success,
            "n_correct": int(record["n_correct"]),
            "n_total": int(record["n_total"]),
            "per_temperature_stats": stats,
        })
    meta = {
        "config": str(config_path),
        "checkpoint": cfg["paths"]["prefix_value_ckpt"],
        "continuations": cfg["paths"]["val_continuations"],
        "feature_cache": cfg["paths"]["val_feature_cache"],
        "calibration_temperature": calibration_temperature,
        "tertile_cut_low": cuts[0],
        "tertile_cut_high": cuts[1],
        "n_prefixes": len(scored),
    }
    return scored, meta


def prefix_stratification_tables(scored: Sequence[Mapping[str, Any]],
                                 n_bootstrap: int,
                                 rng: np.random.Generator) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prefix_rows: List[Dict[str, Any]] = []
    for row in scored:
        base = {key: row[key] for key in [
            "record_index", "problem_id", "source_sample_id", "prefix_segments",
            "prefix_stage", "source_individual_label", "pvm_phi", "pvm_bucket",
            "observed_success_rate", "n_correct", "n_total",
        ]}
        prefix_rows.append(dict(base))

    curve_rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    totals: Dict[Tuple[str, float], List[int]] = defaultdict(list)
    corrects: Dict[Tuple[str, float], List[int]] = defaultdict(list)
    for row in scored:
        bucket = str(row["pvm_bucket"])
        for temp_key, item in row["per_temperature_stats"].items():
            key = (bucket, float(temp_key))
            grouped[key].append(float(item["success_rate"]))
            totals[key].append(int(item["n_total"]))
            corrects[key].append(int(item["n_correct"]))

    for bucket in BUCKET_ORDER:
        temps = sorted(temp for b, temp in grouped if b == bucket)
        for temp in temps:
            values = grouped[(bucket, temp)]
            mean, ci_low, ci_high = bootstrap_mean_ci(values, rng, n_bootstrap=n_bootstrap)
            curve_rows.append({
                "pvm_bucket": bucket,
                "bucket_label": BUCKET_LABELS[bucket],
                "temperature": temp,
                "n_prefixes": len(values),
                "n_continuations": int(sum(totals[(bucket, temp)])),
                "n_correct": int(sum(corrects[(bucket, temp)])),
                "success_rate_mean": mean,
                "success_rate_ci_low": ci_low,
                "success_rate_ci_high": ci_high,
            })
    return prefix_rows, curve_rows


def masked_segment_entropy(entry: Mapping[str, Any], segment_size: int = 64,
                           token_dim: int = 64) -> np.ndarray:
    features = entry["features"].float()
    token_mask = entry["token_mask"].float()
    if features.ndim != 2:
        raise ValueError("features must be [segments, segment_size * token_dim]")
    if features.shape[1] != segment_size * token_dim:
        raise ValueError(
            f"expected feature width {segment_size * token_dim}, got {features.shape[1]}"
        )
    if token_mask.shape != (features.shape[0], segment_size):
        raise ValueError("token_mask shape must match [segments, segment_size]")
    reshaped = features.reshape(features.shape[0], segment_size, token_dim)
    entropy = reshaped[:, :, 1]
    denom = token_mask.sum(dim=1).clamp_min(1.0)
    segment_entropy = (entropy * token_mask).sum(dim=1) / denom
    return segment_entropy.detach().cpu().numpy().astype(float)


def resample_curve(values: Sequence[float], points: int = 12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(points, dtype=float)
    if arr.size == 1:
        return np.full(points, float(arr[0]), dtype=float)
    source_x = np.linspace(0.0, 1.0, arr.size)
    target_x = np.linspace(0.0, 1.0, points)
    return np.interp(target_x, source_x, arr)


def entropy_slope(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    x = np.arange(arr.size, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def entropy_tables(scored: Sequence[Mapping[str, Any]],
                   cache_by_id: Mapping[str, Mapping[str, Any]],
                   segment_size: int,
                   token_dim: int,
                   n_bootstrap: int,
                   rng: np.random.Generator,
                   curve_points: int = 12) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    terminal_curves: Dict[str, List[np.ndarray]] = {"correct": [], "incorrect": []}
    prefix_rows: List[Dict[str, Any]] = []
    seen_source_ids = set()
    for row in scored:
        source_id = str(row["source_sample_id"])
        entry = cache_by_id[source_id]
        ent = masked_segment_entropy(entry, segment_size=segment_size, token_dim=token_dim)
        prefix_len = max(1, min(int(row["prefix_segments"]), int(ent.size)))
        prefix_ent = ent[:prefix_len]
        slope = entropy_slope(prefix_ent)
        prefix_rows.append({
            "record_index": int(row["record_index"]),
            "problem_id": row["problem_id"],
            "source_sample_id": source_id,
            "pvm_phi": float(row["pvm_phi"]),
            "pvm_bucket": row["pvm_bucket"],
            "observed_success_rate": float(row["observed_success_rate"]),
            "prefix_segments": int(row["prefix_segments"]),
            "prefix_entropy_mean": float(np.mean(prefix_ent)),
            "prefix_entropy_last": float(prefix_ent[-1]),
            "prefix_entropy_slope": float(slope),
        })
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        outcome = "correct" if float(entry["terminal_target"]) >= 0.5 else "incorrect"
        terminal_curves[outcome].append(resample_curve(ent, points=curve_points))

    curve_rows: List[Dict[str, Any]] = []
    x_grid = np.linspace(0.0, 1.0, curve_points)
    for outcome in ["correct", "incorrect"]:
        curves = terminal_curves[outcome]
        if not curves:
            continue
        arr = np.vstack(curves)
        for point_idx, progress in enumerate(x_grid):
            values = arr[:, point_idx]
            mean, ci_low, ci_high = bootstrap_mean_ci(values, rng, n_bootstrap=n_bootstrap)
            curve_rows.append({
                "outcome": outcome,
                "progress": float(progress),
                "point_index": point_idx,
                "n_trajectories": int(arr.shape[0]),
                "entropy_mean": mean,
                "entropy_ci_low": ci_low,
                "entropy_ci_high": ci_high,
            })

    slopes = [float(row["prefix_entropy_slope"]) for row in prefix_rows]
    slope_labels, slope_cuts = assign_tertiles(slopes)
    heat_groups: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row, slope_bucket in zip(prefix_rows, slope_labels):
        row["entropy_slope_bucket"] = slope_bucket
        heat_groups[(str(row["pvm_bucket"]), slope_bucket)].append(float(row["observed_success_rate"]))
    heatmap_rows: List[Dict[str, Any]] = []
    for pvm_bucket in BUCKET_ORDER:
        for slope_bucket in BUCKET_ORDER:
            values = heat_groups.get((pvm_bucket, slope_bucket), [])
            heatmap_rows.append({
                "pvm_bucket": pvm_bucket,
                "entropy_slope_bucket": slope_bucket,
                "n_prefixes": len(values),
                "mean_continuation_success": float(np.mean(values)) if values else math.nan,
            })

    phi = np.asarray([float(row["pvm_phi"]) for row in prefix_rows], dtype=float)
    observed = np.asarray([float(row["observed_success_rate"]) for row in prefix_rows], dtype=float)
    slope_arr = np.asarray([float(row["prefix_entropy_slope"]) for row in prefix_rows], dtype=float)
    metrics = {
        "pearson_slope_phi": float(scipy_stats.pearsonr(slope_arr, phi).statistic) if len(prefix_rows) > 1 else 0.0,
        "spearman_slope_phi": float(scipy_stats.spearmanr(slope_arr, phi).statistic) if len(prefix_rows) > 1 else 0.0,
        "pearson_slope_success": float(scipy_stats.pearsonr(slope_arr, observed).statistic) if len(prefix_rows) > 1 else 0.0,
        "spearman_slope_success": float(scipy_stats.spearmanr(slope_arr, observed).statistic) if len(prefix_rows) > 1 else 0.0,
        "slope_cut_low": float(slope_cuts[0]),
        "slope_cut_high": float(slope_cuts[1]),
    }
    return prefix_rows, curve_rows, heatmap_rows, metrics


def plot_temperature_landscape(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> List[Path]:
    x = np.asarray([float(row["temperature"]) for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.095, top=0.82, hspace=0.34, wspace=0.24)
    add_figure_header(
        fig,
        "Temperature changes the reasoning ensemble trade-off",
        "Fixed-temperature self-consistency on 200 problems, 15 temperatures, and 8 votes per problem. "
        "The same intervention changes accuracy, calibration, diversity, agreement, and token cost.",
    )
    panels = [
        ("majority_accuracy", "Majority accuracy", True, COLORS["blue"]),
        ("pass_at_1_accuracy", "Pass@1 / individual accuracy", True, COLORS["olive"]),
        ("ece", "Self-consistency ECE", False, COLORS["orange"]),
        ("answer_entropy", "Answer entropy", False, COLORS["pink"]),
        ("vote_margin", "Normalized vote margin", True, COLORS["gold"]),
        ("mean_tokens_per_vote", "Mean tokens per vote", False, COLORS["neutral_dark"]),
    ]
    for ax, (key, title, percent, color) in zip(axes.flat, panels):
        y = np.asarray([float(row[key]) for row in rows])
        ax.plot(x, y, marker="o", markersize=4.2, color=color, linewidth=1.8)
        ax.axvline(1.0, color=TOKENS["ink"], linestyle="--", linewidth=1.0, alpha=0.55)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=8)
        style_axis(ax, percent=percent)
        ax.set_xlabel("Temperature")
    axes[0, 0].annotate(
        "T=1.0",
        xy=(1.0, axes[0, 0].get_ylim()[1]),
        xytext=(4, -18),
        textcoords="offset points",
        color=TOKENS["muted"],
        fontsize=8.5,
    )
    return save_figure(fig, out_dir, "figure1_temperature_landscape")


def plot_prefix_stratification(curve_rows: Sequence[Mapping[str, Any]], out_dir: Path,
                               meta: Mapping[str, Any]) -> List[Path]:
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.76)
    add_figure_header(
        fig,
        "Temperature utility depends on prefix value",
        "Validation prefixes are scored offline by the existing PVM and split into equal-count tertiles. "
        "Curves show continuation success by temperature with bootstrap 95% intervals across prefixes.",
    )
    for bucket in BUCKET_ORDER:
        rows = sorted([row for row in curve_rows if row["pvm_bucket"] == bucket], key=lambda r: float(r["temperature"]))
        x = np.asarray([float(row["temperature"]) for row in rows])
        y = np.asarray([float(row["success_rate_mean"]) for row in rows])
        lo = np.asarray([float(row["success_rate_ci_low"]) for row in rows])
        hi = np.asarray([float(row["success_rate_ci_high"]) for row in rows])
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.5,
                label=BUCKET_LABELS[bucket], color=BUCKET_COLORS[bucket])
        ax.fill_between(x, lo, hi, color=BUCKET_COLORS[bucket], alpha=0.14, linewidth=0)
    style_axis(ax, percent=True, y_limits=(0.0, 1.0))
    ax.set_xlabel("Continuation temperature")
    ax.set_ylabel("Continuation success rate")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.03))
    ax.text(
        0.99, -0.18,
        f"phi tertile cuts: {float(meta['tertile_cut_low']):.3f}, {float(meta['tertile_cut_high']):.3f}; "
        f"n={int(meta['n_prefixes'])} prefixes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=TOKENS["muted"],
        fontsize=8.5,
    )
    return save_figure(fig, out_dir, "figure2_prefix_value_stratification")


def plot_entropy_dynamics(prefix_rows: Sequence[Mapping[str, Any]],
                          curve_rows: Sequence[Mapping[str, Any]],
                          heatmap_rows: Sequence[Mapping[str, Any]],
                          metrics: Mapping[str, float],
                          out_dir: Path) -> List[Path]:
    fig = plt.figure(figsize=(13.6, 7.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0], height_ratios=[1.0, 1.0])
    ax_curve = fig.add_subplot(gs[:, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.095, top=0.80, wspace=0.27, hspace=0.38)
    add_figure_header(
        fig,
        "Entropy dynamics provide an observable reasoning-state signal",
        "Segment entropy is recovered from cached PVM features. Curves compare correct and incorrect terminal trajectories; "
        "prefix-level slope is compared with PVM value and continuation success.",
    )

    for outcome in ["correct", "incorrect"]:
        rows = sorted([row for row in curve_rows if row["outcome"] == outcome], key=lambda r: float(r["progress"]))
        x = np.asarray([float(row["progress"]) for row in rows])
        y = np.asarray([float(row["entropy_mean"]) for row in rows])
        lo = np.asarray([float(row["entropy_ci_low"]) for row in rows])
        hi = np.asarray([float(row["entropy_ci_high"]) for row in rows])
        color = SUCCESS_COLORS[outcome]
        ax_curve.plot(x, y, marker="o", linewidth=2.0, markersize=4.3,
                      label=outcome.title(), color=color)
        ax_curve.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
    style_axis(ax_curve)
    ax_curve.set_title("A. Entropy curve by terminal outcome", loc="left", fontsize=10.5, fontweight="bold")
    ax_curve.set_xlabel("Normalized reasoning progress")
    ax_curve.set_ylabel("Mean segment entropy")
    ax_curve.legend(loc="upper right")

    slope = np.asarray([float(row["prefix_entropy_slope"]) for row in prefix_rows])
    phi = np.asarray([float(row["pvm_phi"]) for row in prefix_rows])
    success = np.asarray([float(row["observed_success_rate"]) for row in prefix_rows])
    colors = [BUCKET_COLORS[str(row["pvm_bucket"])] for row in prefix_rows]
    ax_scatter.scatter(slope, phi, c=colors, s=22, alpha=0.72,
                       edgecolors=TOKENS["panel"], linewidths=0.35)
    if len(slope) > 1:
        coeff = np.polyfit(slope, phi, 1)
        x_line = np.linspace(float(np.min(slope)), float(np.max(slope)), 100)
        ax_scatter.plot(x_line, coeff[0] * x_line + coeff[1],
                        color=TOKENS["ink"], linestyle="--", linewidth=1.0, alpha=0.6)
    style_axis(ax_scatter)
    ax_scatter.set_title("B. Prefix entropy slope vs PVM value", loc="left", fontsize=10.5, fontweight="bold")
    ax_scatter.set_xlabel("Prefix entropy slope")
    ax_scatter.set_ylabel("PVM phi")
    ax_scatter.text(
        0.98, 0.04,
        f"vs phi: rho={metrics['spearman_slope_phi']:.2f}, r={metrics['pearson_slope_phi']:.2f}\n"
        f"vs success: rho={metrics['spearman_slope_success']:.2f}, r={metrics['pearson_slope_success']:.2f}",
        transform=ax_scatter.transAxes,
        ha="right",
        va="bottom",
        color=TOKENS["muted"],
        fontsize=8.5,
    )

    matrix = np.full((3, 3), np.nan, dtype=float)
    counts = np.zeros((3, 3), dtype=int)
    idx = {bucket: i for i, bucket in enumerate(BUCKET_ORDER)}
    for row in heatmap_rows:
        i = idx[str(row["pvm_bucket"])]
        j = idx[str(row["entropy_slope_bucket"])]
        matrix[i, j] = float(row["mean_continuation_success"])
        counts[i, j] = int(row["n_prefixes"])
    masked = np.ma.masked_invalid(matrix)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "success_blue", ["#F4F5F7", "#CEDFFE", "#5477C4", "#2E4780"]
    )
    image = ax_heat.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax_heat.set_title("C. Continuation success by state buckets", loc="left", fontsize=10.5, fontweight="bold")
    ax_heat.set_xticks(range(3), ["Low slope", "Mid slope", "High slope"])
    ax_heat.set_yticks(range(3), [BUCKET_LABELS[b] for b in BUCKET_ORDER])
    ax_heat.set_xlabel("Entropy-slope tertile")
    ax_heat.set_ylabel("PVM tertile")
    for i in range(3):
        for j in range(3):
            if np.isfinite(matrix[i, j]):
                ax_heat.text(j, i, f"{matrix[i, j]:.2f}\n(n={counts[i, j]})",
                             ha="center", va="center", fontsize=8.2,
                             color=TOKENS["ink"] if matrix[i, j] < 0.72 else TOKENS["panel"])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Mean continuation success")
    return save_figure(fig, out_dir, "figure3_entropy_dynamics_vs_success")


def summary_text(out_dir: Path,
                 args: argparse.Namespace,
                 temp_rows: Sequence[Mapping[str, Any]],
                 pvm_meta: Mapping[str, Any],
                 entropy_metrics: Mapping[str, float],
                 outputs: Sequence[Path]) -> str:
    best_majority = max(temp_rows, key=lambda row: float(row["majority_accuracy"]))
    best_ece = min(temp_rows, key=lambda row: float(row["ece"]))
    lines = [
        "# Existing-Data Paper Analysis Figures",
        "",
        "This bundle was generated from existing artifacts only. It does not run vLLM, generate new continuations, train PPO, or train a value model.",
        "",
        "## Sources",
        "",
        f"- Temperature landscape input: `{args.temperature_input}`",
        f"- PVM config: `{args.pvm_config}`",
        f"- PVM checkpoint: `{pvm_meta['checkpoint']}`",
        f"- Prefix continuations: `{pvm_meta['continuations']}`",
        f"- Feature cache: `{pvm_meta['feature_cache']}`",
        "",
        "## Denominators",
        "",
        f"- Figure 1 majority accuracy: `{int(temp_rows[0]['n_prompts'])}` prompts per temperature.",
        f"- Figure 1 pass@1: `{int(temp_rows[0]['n_votes'])}` votes per temperature.",
        f"- Figure 2 PVM prefixes: `{int(pvm_meta['n_prefixes'])}` validation prefixes, split into equal-count tertiles.",
        "- Figure 3 terminal entropy curves use unique source trajectories in the feature cache that appear in the validation continuation records.",
        "",
        "## Figure Captions",
        "",
        "- Figure 1: Temperature changes a bundle of ensemble properties. In this run the best majority-accuracy temperature is "
        f"`{float(best_majority['temperature']):.1f}` while the lowest-ECE temperature is `{float(best_ece['temperature']):.1f}`, "
        "supporting the trade-off framing rather than a single accuracy-only view.",
        "- Figure 2: Prefix value stratification shows whether a temperature is useful depends on the prefix state. These validation-prefix curves are diagnostic evidence, not a new held-out benchmark.",
        "- Figure 3: Entropy dynamics are an observable reasoning-state signal. Correct and incorrect terminal trajectories have different entropy trajectories; prefix entropy slope is reported as a diagnostic and is weakly correlated with PVM value in this validation slice.",
        "",
        "## Entropy Correlations",
        "",
        f"- Prefix entropy slope vs PVM phi: Spearman `{entropy_metrics['spearman_slope_phi']:.4f}`, Pearson `{entropy_metrics['pearson_slope_phi']:.4f}`.",
        f"- Prefix entropy slope vs continuation success: Spearman `{entropy_metrics['spearman_slope_success']:.4f}`, Pearson `{entropy_metrics['pearson_slope_success']:.4f}`.",
        "",
        "## Outputs",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.relative_to(out_dir)}`")
    lines.append("")
    return "\n".join(lines)


def default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"analysis_figures_{stamp}"


def build_figures(args: argparse.Namespace) -> Dict[str, Any]:
    configure_matplotlib()
    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "tables"
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    temp_rows_raw = load_jsonl(str(args.temperature_input))
    temp_rows, temp_detail_rows = temperature_landscape_table(temp_rows_raw)
    write_csv(data_dir / "temperature_landscape.csv", temp_rows)
    write_csv(data_dir / "temperature_landscape_by_problem.csv", temp_detail_rows)

    scored, pvm_meta = score_pvm_prefixes(Path(args.pvm_config), device_name=args.device, batch_size=args.batch_size)
    prefix_rows, strat_rows = prefix_stratification_tables(scored, int(args.bootstrap), rng)
    write_csv(data_dir / "pvm_prefix_scores.csv", prefix_rows)
    write_csv(data_dir / "prefix_value_stratification.csv", strat_rows)

    cfg, _, cache_by_id, _ = load_pvm_inputs(Path(args.pvm_config))
    entropy_prefix_rows, entropy_curve_rows, entropy_heatmap_rows, entropy_metrics = entropy_tables(
        scored,
        cache_by_id,
        segment_size=int(cfg["data"]["segment_size"]),
        token_dim=int(cfg["data"]["instance_dim"]),
        n_bootstrap=int(args.bootstrap),
        rng=rng,
        curve_points=int(args.entropy_points),
    )
    write_csv(data_dir / "entropy_prefix_metrics.csv", entropy_prefix_rows)
    write_csv(data_dir / "entropy_terminal_curves.csv", entropy_curve_rows)
    write_csv(data_dir / "entropy_state_heatmap.csv", entropy_heatmap_rows)

    outputs: List[Path] = []
    outputs.extend(plot_temperature_landscape(temp_rows, out_dir))
    outputs.extend(plot_prefix_stratification(strat_rows, out_dir, pvm_meta))
    outputs.extend(plot_entropy_dynamics(entropy_prefix_rows, entropy_curve_rows, entropy_heatmap_rows, entropy_metrics, out_dir))
    for csv_path in sorted(data_dir.glob("*.csv")):
        outputs.append(csv_path)

    summary = summary_text(out_dir, args, temp_rows, pvm_meta, entropy_metrics, outputs)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    outputs.append(summary_path)
    manifest = {
        "out_dir": str(out_dir),
        "bootstrap": int(args.bootstrap),
        "device": args.device,
        "outputs": [str(path) for path in outputs],
        "sources": {
            "temperature_input": str(args.temperature_input),
            "pvm_config": str(args.pvm_config),
            **pvm_meta,
        },
        "entropy_metrics": entropy_metrics,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.append(out_dir / "manifest.json")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-input", default="datasets/all_5_sub_200.jsonl")
    parser.add_argument("--pvm-config", default="configs/training/min_pvm_ppo_500_seed42.yaml")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--entropy-points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args(argv)
    manifest = build_figures(args)
    print(json.dumps({
        "out_dir": manifest["out_dir"],
        "n_outputs": len(manifest["outputs"]),
        "bootstrap": manifest["bootstrap"],
    }, indent=2))


if __name__ == "__main__":
    main()
