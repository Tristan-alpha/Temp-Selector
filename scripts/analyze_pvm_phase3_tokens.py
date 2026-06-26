#!/usr/bin/env python3
"""Analyze phase-3 token rates for high- and low-PVM validation prefixes.

Phase 3 follows the token-level definition used by the Qwen paper's Figure 4:
the final layer perturbs a token when its logit-lens entropy is higher than
the previous layer's entropy, i.e. H_L - H_{L-1} > 0.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import itertools
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.vllm_runner import _cleanup_hidden_states_file, _load_hidden_states_file


DEFAULT_PVM_SCORES = Path("results/analysis_figures_20260623_183844/tables/pvm_prefix_scores.csv")
DEFAULT_CONTINUATIONS = Path("datasets/min_pvm_ppo_500_seed42_20260618/prefix_continuations_val.jsonl")
DEFAULT_VAL_DATASET = Path("datasets/val_5_small_500.jsonl")
DEFAULT_MODEL = Path("/home/data/nas_hdd/jiaxuan/ckpt_ada2/Qwen3-8B")

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "high": "#A3BEFA",
    "high_edge": "#2E4780",
    "low": "#F0986E",
    "low_edge": "#804126",
    "mid": "#FFE15B",
    "neutral": "#7A828F",
    "neutral_dark": "#464C55",
}


@dataclass(frozen=True)
class PrefixRow:
    record_index: int
    problem_id: str
    source_sample_id: str
    prefix_segments: int
    prefix_stage: str
    prefix_token_end: int
    pvm_phi: float
    pvm_bucket: str
    observed_success_rate: float
    n_correct: int
    n_total: int


@dataclass(frozen=True)
class ExtractionJob:
    source_sample_id: str
    prompt_ids: List[int]
    response_ids: List[int]
    max_prefix_token_end: int


@dataclass(frozen=True)
class SourceEntropy:
    prev_entropy: np.ndarray
    last_entropy: np.ndarray
    delta_entropy: np.ndarray
    prev_sampled_logprob: np.ndarray
    last_sampled_logprob: np.ndarray


class _LayerEntropyComputeFn:
    """Compute full-vocabulary entropy for extracted hidden states in vLLM."""

    def __init__(self, hidden_states_cpu: torch.Tensor, token_ids_cpu: torch.Tensor):
        self.hidden_states_cpu = hidden_states_cpu
        self.token_ids_cpu = token_ids_cpu

    def __call__(self, model):
        dev = next(model.parameters()).device
        h = self.hidden_states_cpu.to(dev, non_blocking=True)
        ids = self.token_ids_cpu.to(dev, non_blocking=True)
        outputs = []
        for layer_pos in range(h.shape[1]):
            normed = model.model.norm(h[:, layer_pos, :])
            logits = model.compute_logits(normed)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1)
            sampled = log_probs.gather(1, ids.unsqueeze(1)).squeeze(1)
            outputs.append(torch.stack([entropy, sampled], dim=1))
            del logits, log_probs, probs
        return torch.stack(outputs, dim=1).cpu()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _maybe_int(value: Any) -> int:
    return int(float(value))


def load_prefix_rows(
    pvm_scores_path: Path,
    continuations_path: Path,
    buckets: Sequence[str] = ("low", "high"),
    max_prefixes_per_bucket: int = 0,
    max_prefix_tokens: int = 0,
) -> List[PrefixRow]:
    continuation_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(continuations_path):
        key = (str(row["source_sample_id"]), int(row["prefix_segments"]))
        continuation_by_key[key] = row

    selected: List[PrefixRow] = []
    bucket_counts: Dict[str, int] = defaultdict(int)
    wanted = set(str(bucket) for bucket in buckets)
    with pvm_scores_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            bucket = str(row["pvm_bucket"])
            if bucket not in wanted:
                continue
            if max_prefixes_per_bucket > 0 and bucket_counts[bucket] >= max_prefixes_per_bucket:
                continue
            key = (str(row["source_sample_id"]), int(row["prefix_segments"]))
            if key not in continuation_by_key:
                raise KeyError(f"missing continuation record for {key}")
            continuation = continuation_by_key[key]
            prefix_token_end = int(continuation["prefix_token_end"])
            if max_prefix_tokens > 0:
                prefix_token_end = min(prefix_token_end, int(max_prefix_tokens))
            if prefix_token_end <= 0:
                continue
            selected.append(PrefixRow(
                record_index=_maybe_int(row["record_index"]),
                problem_id=str(row["problem_id"]),
                source_sample_id=str(row["source_sample_id"]),
                prefix_segments=int(row["prefix_segments"]),
                prefix_stage=str(row.get("prefix_stage", "unknown")),
                prefix_token_end=prefix_token_end,
                pvm_phi=float(row["pvm_phi"]),
                pvm_bucket=bucket,
                observed_success_rate=float(row["observed_success_rate"]),
                n_correct=int(row["n_correct"]),
                n_total=int(row["n_total"]),
            ))
            bucket_counts[bucket] += 1

    expected = {
        bucket: min(
            sum(1 for row in selected if row.pvm_bucket == bucket),
            max_prefixes_per_bucket if max_prefixes_per_bucket > 0 else 10**12,
        )
        for bucket in wanted
    }
    missing = sorted(bucket for bucket in wanted if expected.get(bucket, 0) == 0)
    if missing:
        raise RuntimeError(f"no prefixes selected for buckets: {missing}")
    return selected


def load_source_rows(dataset_path: Path, needed_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    needed = set(str(item) for item in needed_ids)
    found: Dict[str, Dict[str, Any]] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not needed:
                break
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            if sid in needed:
                found[sid] = row
                needed.remove(sid)
    if needed:
        preview = ", ".join(sorted(needed)[:5])
        raise KeyError(f"dataset is missing {len(needed)} source rows: {preview}")
    return found


def build_extraction_jobs(
    prefix_rows: Sequence[PrefixRow],
    source_rows: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
) -> List[ExtractionJob]:
    max_by_source: Dict[str, int] = defaultdict(int)
    for row in prefix_rows:
        max_by_source[row.source_sample_id] = max(max_by_source[row.source_sample_id], row.prefix_token_end)

    jobs: List[ExtractionJob] = []
    for source_sample_id in sorted(max_by_source):
        source = source_rows[source_sample_id]
        prompt = source.get("metadata", {}).get("rendered_prompt") or source.get("prompt", "")
        encoded = tokenizer(prompt, add_special_tokens=False)
        prompt_ids = list(encoded.input_ids)
        response_ids = list(source.get("token_ids", []))
        max_end = min(max_by_source[source_sample_id], len(response_ids))
        if max_end <= 0:
            continue
        jobs.append(ExtractionJob(
            source_sample_id=source_sample_id,
            prompt_ids=prompt_ids,
            response_ids=response_ids[:max_end],
            max_prefix_token_end=max_end,
        ))
    return jobs


def phase3_for_prefixes(
    prefix_rows: Sequence[PrefixRow],
    source_entropy: Mapping[str, SourceEntropy],
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in prefix_rows:
        entropy = source_entropy[row.source_sample_id]
        n = min(row.prefix_token_end, int(entropy.delta_entropy.shape[0]))
        if n <= 0:
            raise RuntimeError(f"no entropy values for prefix {row.source_sample_id}")
        delta = entropy.delta_entropy[:n]
        mask = delta > float(threshold)
        phase3_tokens = int(mask.sum())
        out.append({
            "record_index": row.record_index,
            "problem_id": row.problem_id,
            "source_sample_id": row.source_sample_id,
            "prefix_segments": row.prefix_segments,
            "prefix_stage": row.prefix_stage,
            "pvm_bucket": row.pvm_bucket,
            "pvm_phi": row.pvm_phi,
            "observed_success_rate": row.observed_success_rate,
            "n_correct": row.n_correct,
            "n_total": row.n_total,
            "prefix_token_end": n,
            "phase3_tokens": phase3_tokens,
            "non_phase3_tokens": int(n - phase3_tokens),
            "phase3_rate": phase3_tokens / n,
            "mean_delta_entropy": float(np.mean(delta)),
            "median_delta_entropy": float(np.median(delta)),
            "mean_prev_entropy": float(np.mean(entropy.prev_entropy[:n])),
            "mean_last_entropy": float(np.mean(entropy.last_entropy[:n])),
            "mean_prev_sampled_logprob": float(np.mean(entropy.prev_sampled_logprob[:n])),
            "mean_last_sampled_logprob": float(np.mean(entropy.last_sampled_logprob[:n])),
        })
    return out


def bootstrap_mean_diff(
    high: Sequence[float],
    low: Sequence[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    observed = float(np.mean(high_arr) - np.mean(low_arr))
    if n_bootstrap <= 0:
        return {"observed": observed, "ci_low": observed, "ci_high": observed}
    rng = np.random.default_rng(seed)
    high_samples = rng.choice(high_arr, size=(int(n_bootstrap), high_arr.size), replace=True).mean(axis=1)
    low_samples = rng.choice(low_arr, size=(int(n_bootstrap), low_arr.size), replace=True).mean(axis=1)
    diffs = high_samples - low_samples
    return {
        "observed": observed,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def permutation_p_value(
    high: Sequence[float],
    low: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
    max_exact_partitions: int = 50000,
) -> float:
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    observed = abs(float(np.mean(high_arr) - np.mean(low_arr)))
    combined = np.concatenate([high_arr, low_arr])
    n_high = high_arr.size
    total_partitions = math.comb(combined.size, n_high)
    if total_partitions <= max_exact_partitions:
        extreme = 0
        total = 0
        indices = range(combined.size)
        for high_idx in itertools.combinations(indices, n_high):
            mask = np.zeros(combined.size, dtype=bool)
            mask[list(high_idx)] = True
            diff = abs(float(np.mean(combined[mask]) - np.mean(combined[~mask])))
            extreme += int(diff >= observed - 1e-12)
            total += 1
        return float(extreme / max(1, total))

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(combined)
        diff = abs(float(np.mean(permuted[:n_high]) - np.mean(permuted[n_high:])))
        extreme += int(diff >= observed - 1e-12)
    return float((extreme + 1) / (int(n_permutations) + 1))


def _bucket_values(rows: Sequence[Mapping[str, Any]], bucket: str, key: str) -> List[float]:
    return [float(row[key]) for row in rows if row["pvm_bucket"] == bucket]


def assign_length_tertiles(lengths: Sequence[int]) -> List[str]:
    if not lengths:
        return []
    labels = [""] * len(lengths)
    order = np.argsort(np.asarray(lengths, dtype=float), kind="mergesort")
    names = ["short", "mid", "long"]
    for rank, idx in enumerate(order):
        labels[int(idx)] = names[min(2, int(rank * 3 / len(lengths)))]
    return labels


def summarize_results(
    prefix_results: Sequence[Mapping[str, Any]],
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "phase3_definition": "phase3 iff final-layer entropy minus previous-layer entropy is greater than 0",
        "buckets": {},
    }
    for bucket in ("low", "high"):
        rows = [row for row in prefix_results if row["pvm_bucket"] == bucket]
        rates = np.asarray([float(row["phase3_rate"]) for row in rows], dtype=float)
        tokens = int(sum(int(row["prefix_token_end"]) for row in rows))
        phase3_tokens = int(sum(int(row["phase3_tokens"]) for row in rows))
        summary["buckets"][bucket] = {
            "n_prefixes": len(rows),
            "total_tokens": tokens,
            "phase3_tokens": phase3_tokens,
            "token_pooled_phase3_rate": phase3_tokens / max(1, tokens),
            "prefix_mean_phase3_rate": float(np.mean(rates)) if rates.size else 0.0,
            "prefix_median_phase3_rate": float(np.median(rates)) if rates.size else 0.0,
            "prefix_std_phase3_rate": float(np.std(rates, ddof=1)) if rates.size > 1 else 0.0,
            "mean_prefix_token_end": float(np.mean([int(row["prefix_token_end"]) for row in rows])) if rows else 0.0,
            "mean_delta_entropy": float(np.mean([float(row["mean_delta_entropy"]) for row in rows])) if rows else 0.0,
        }

    high_rates = _bucket_values(prefix_results, "high", "phase3_rate")
    low_rates = _bucket_values(prefix_results, "low", "phase3_rate")
    diff = bootstrap_mean_diff(high_rates, low_rates, n_bootstrap=n_bootstrap, seed=seed)
    summary["prefix_mean_difference_high_minus_low"] = {
        **diff,
        "permutation_p_value_two_sided": permutation_p_value(
            high_rates, low_rates, n_permutations=n_permutations, seed=seed,
        ),
    }
    summary["length_strata"] = length_strata_summary(prefix_results)
    summary["relative_position_deciles"] = relative_position_deciles(prefix_results)
    summary["interpretation"] = interpret_difference(diff)
    return summary


def interpret_difference(diff: Mapping[str, float]) -> str:
    observed = float(diff["observed"])
    ci_low = float(diff["ci_low"])
    ci_high = float(diff["ci_high"])
    if ci_high < 0.0:
        return "high PVM prefixes have a lower phase-3 rate than low PVM prefixes"
    if ci_low > 0.0:
        return "high PVM prefixes have a higher phase-3 rate than low PVM prefixes"
    if observed < 0.0:
        return "high PVM prefixes trend lower, but the bootstrap interval overlaps zero"
    if observed > 0.0:
        return "high PVM prefixes trend higher, but the bootstrap interval overlaps zero"
    return "no observed mean phase-3-rate difference"


def length_strata_summary(prefix_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    lengths = [int(row["prefix_token_end"]) for row in prefix_results]
    labels = assign_length_tertiles(lengths)
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row, label in zip(prefix_results, labels):
        grouped[(str(row["pvm_bucket"]), label)].append(row)
    for stratum in ("short", "mid", "long"):
        for bucket in ("low", "high"):
            items = grouped.get((bucket, stratum), [])
            tokens = int(sum(int(row["prefix_token_end"]) for row in items))
            phase3 = int(sum(int(row["phase3_tokens"]) for row in items))
            rows.append({
                "length_stratum": stratum,
                "pvm_bucket": bucket,
                "n_prefixes": len(items),
                "total_tokens": tokens,
                "token_pooled_phase3_rate": phase3 / max(1, tokens),
                "prefix_mean_phase3_rate": float(np.mean([float(row["phase3_rate"]) for row in items])) if items else 0.0,
                "mean_prefix_token_end": float(np.mean([int(row["prefix_token_end"]) for row in items])) if items else 0.0,
            })
    return rows


def relative_position_deciles(prefix_results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Approximate decile summaries from per-prefix rates and token counts.

    Full token-level decile summaries are written during extraction when
    token-level deltas are available. This fallback keeps summary generation
    pure for tests and cached prefix-level runs.
    """
    rows: List[Dict[str, Any]] = []
    for bucket in ("low", "high"):
        items = [row for row in prefix_results if row["pvm_bucket"] == bucket]
        for decile in range(10):
            rows.append({
                "pvm_bucket": bucket,
                "decile": decile,
                "n_prefixes": len(items),
                "prefix_mean_phase3_rate": float(np.mean([float(row["phase3_rate"]) for row in items])) if items else 0.0,
            })
    return rows


def token_decile_summary(
    prefix_rows: Sequence[PrefixRow],
    source_entropy: Mapping[str, SourceEntropy],
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, Any]] = defaultdict(lambda: {
        "phase3_tokens": 0,
        "total_tokens": 0,
        "prefix_rates": [],
    })
    for row in prefix_rows:
        delta = source_entropy[row.source_sample_id].delta_entropy[:row.prefix_token_end]
        n = int(delta.shape[0])
        for decile in range(10):
            start = int(math.floor(decile * n / 10))
            end = int(math.floor((decile + 1) * n / 10))
            if end <= start:
                continue
            part = delta[start:end]
            phase3 = int((part > float(threshold)).sum())
            entry = grouped[(row.pvm_bucket, decile)]
            entry["phase3_tokens"] += phase3
            entry["total_tokens"] += int(part.shape[0])
            entry["prefix_rates"].append(phase3 / max(1, int(part.shape[0])))
    rows: List[Dict[str, Any]] = []
    for bucket in ("low", "high"):
        for decile in range(10):
            item = grouped.get((bucket, decile), {"phase3_tokens": 0, "total_tokens": 0, "prefix_rates": []})
            rows.append({
                "pvm_bucket": bucket,
                "decile": decile,
                "phase3_tokens": int(item["phase3_tokens"]),
                "total_tokens": int(item["total_tokens"]),
                "token_pooled_phase3_rate": int(item["phase3_tokens"]) / max(1, int(item["total_tokens"])),
                "prefix_mean_phase3_rate": float(np.mean(item["prefix_rates"])) if item["prefix_rates"] else 0.0,
            })
    return rows


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "axes.titlecolor": TOKENS["ink"],
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "xtick.color": TOKENS["muted"],
        "ytick.color": TOKENS["muted"],
        "grid.color": TOKENS["grid"],
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": TOKENS["surface"],
        "savefig.bbox": "tight",
    })


def add_chart_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.975, title, ha="left", va="top",
             fontsize=14, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.06, 0.925, subtitle, ha="left", va="top",
             fontsize=9.5, color=TOKENS["muted"])


def plot_phase3_summary(
    prefix_results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    decile_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.77, wspace=0.28)
    add_chart_header(
        fig,
        "Phase-3 token rates separate high- and low-PVM prefixes",
        "Phase 3 is defined as H_L - H_(L-1) > 0 for Qwen3-8B response tokens; intervals are bootstrap 95% CIs across prefixes.",
    )

    ax = axes[0]
    buckets = ["low", "high"]
    x = np.arange(len(buckets))
    means = [summary["buckets"][bucket]["prefix_mean_phase3_rate"] for bucket in buckets]
    colors = [COLORS["low"], COLORS["high"]]
    edges = [COLORS["low_edge"], COLORS["high_edge"]]
    bars = ax.bar(x, means, color=colors, edgecolor=edges, linewidth=1.0, width=0.56)
    for bar, bucket, mean in zip(bars, buckets, means):
        pooled = summary["buckets"][bucket]["token_pooled_phase3_rate"]
        n_prefixes = summary["buckets"][bucket]["n_prefixes"]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{mean:.1%}\npooled {pooled:.1%}\nn={n_prefixes}",
                ha="center", va="bottom", fontsize=8.5, color=TOKENS["ink"])
    diff = summary["prefix_mean_difference_high_minus_low"]
    ax.axhline(0.0, color=TOKENS["axis"], linewidth=1.0)
    ax.set_xticks(x, ["Low PVM", "High PVM"])
    ax.set_ylabel("Mean phase-3 rate per prefix")
    ax.yaxis.set_major_formatter(lambda y, _: f"{100 * y:.0f}%")
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0, -0.22,
        f"High - low: {diff['observed']:.2%} "
        f"[{diff['ci_low']:.2%}, {diff['ci_high']:.2%}], "
        f"p={diff['permutation_p_value_two_sided']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=TOKENS["muted"],
    )

    ax2 = axes[1]
    for bucket, label, color in (
        ("low", "Low PVM", COLORS["low_edge"]),
        ("high", "High PVM", COLORS["high_edge"]),
    ):
        rows = sorted([row for row in decile_rows if row["pvm_bucket"] == bucket], key=lambda r: int(r["decile"]))
        xs = [int(row["decile"]) + 1 for row in rows]
        ys = [float(row.get("prefix_mean_phase3_rate", row.get("token_pooled_phase3_rate", 0.0))) for row in rows]
        ax2.plot(xs, ys, marker="o", linewidth=1.4, markersize=4.2, label=label, color=color)
    ax2.set_xlabel("Relative token-position decile")
    ax2.set_ylabel("Phase-3 rate")
    ax2.yaxis.set_major_formatter(lambda y, _: f"{100 * y:.0f}%")
    ax2.set_xticks(range(1, 11))
    ax2.grid(True, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), max(1, int(batch_size))):
        yield items[start:start + max(1, int(batch_size))]


def extract_source_entropy_with_vllm(
    jobs: Sequence[ExtractionJob],
    model_path: Path,
    max_model_len: int,
    gpu_memory_utilization: float,
    parallel_size: int,
    batch_size: int,
    entropy_chunk_size: int,
    enforce_eager: bool,
) -> Tuple[Dict[str, SourceEntropy], Dict[str, Any]]:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    hf_cfg = AutoConfig.from_pretrained(str(model_path))
    last_layer_id = int(hf_cfg.num_hidden_layers)
    layer_ids = [last_layer_id - 1, last_layer_id]
    hs_tmpdir = tempfile.mkdtemp(prefix="pvm_phase3_hs_", dir="/dev/shm")
    atexit.register(lambda: shutil.rmtree(hs_tmpdir, ignore_errors=True))
    print(f"[phase3] loading vLLM model={model_path} layers={layer_ids} tp={parallel_size}")
    t0 = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=int(parallel_size),
        max_model_len=int(max_model_len),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=bool(enforce_eager),
        enable_chunked_prefill=False,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layer_ids,
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_port": _free_port(),
            "kv_connector_extra_config": {
                "shared_storage_path": hs_tmpdir,
            },
        },
    )
    load_seconds = time.perf_counter() - t0
    print(f"[phase3] vLLM ready in {load_seconds:.1f}s")

    source_entropy: Dict[str, SourceEntropy] = {}
    params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0, temperature=1.0)]
    processed = 0
    for batch in _batched(list(jobs), batch_size):
        full_ids = [job.prompt_ids + job.response_ids for job in batch]
        batch_params = params * len(full_ids)
        outputs = llm.generate(full_ids, batch_params, use_tqdm=False)
        for job, output in zip(batch, outputs):
            hs_path = output.kv_transfer_params.get("hidden_states_path") if output.kv_transfer_params else None
            if hs_path is None:
                raise RuntimeError(f"vLLM did not return hidden_states_path for {job.source_sample_id}")
            try:
                data = _load_hidden_states_file(hs_path)
                hs = data["hidden_states"]
            finally:
                _cleanup_hidden_states_file(hs_path)
            n_resp = len(job.response_ids)
            if n_resp <= 0:
                raise RuntimeError(f"empty response ids for {job.source_sample_id}")
            if hs.ndim != 3 or hs.shape[1] != 2:
                raise RuntimeError(f"expected hidden states [seq, 2, hidden], got {tuple(hs.shape)}")
            if hs.shape[0] < n_resp + 1:
                raise RuntimeError(
                    f"hidden states too short for {job.source_sample_id}: "
                    f"hs={hs.shape[0]} response={n_resp}"
                )
            response_hs = hs[-(n_resp + 1):-1].cpu()
            token_ids = torch.tensor(job.response_ids, dtype=torch.long)
            chunks: List[torch.Tensor] = []
            for start in range(0, n_resp, int(entropy_chunk_size)):
                end = min(start + int(entropy_chunk_size), n_resp)
                raw = llm.apply_model(_LayerEntropyComputeFn(
                    response_hs[start:end],
                    token_ids[start:end],
                ))[0]
                chunks.append(raw)
            stats = torch.cat(chunks, dim=0).float().numpy()
            source_entropy[job.source_sample_id] = SourceEntropy(
                prev_entropy=stats[:, 0, 0],
                last_entropy=stats[:, 1, 0],
                delta_entropy=stats[:, 1, 0] - stats[:, 0, 0],
                prev_sampled_logprob=stats[:, 0, 1],
                last_sampled_logprob=stats[:, 1, 1],
            )
            processed += 1
            print(
                f"[phase3] {processed}/{len(jobs)} {job.source_sample_id} "
                f"tokens={n_resp} phase3_rate={(source_entropy[job.source_sample_id].delta_entropy > 0).mean():.4f}",
                flush=True,
            )
    shutil.rmtree(hs_tmpdir, ignore_errors=True)
    return source_entropy, {
        "model_path": str(model_path),
        "layer_ids": layer_ids,
        "num_hidden_layers": last_layer_id,
        "load_seconds": load_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvm-scores", type=Path, default=DEFAULT_PVM_SCORES)
    parser.add_argument("--continuations", type=Path, default=DEFAULT_CONTINUATIONS)
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_VAL_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-prefixes-per-bucket", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=0)
    parser.add_argument("--phase3-threshold", type=float, default=0.0)
    parser.add_argument("--parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--entropy-chunk-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path("results") / f"pvm_phase3_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    prefix_rows = load_prefix_rows(
        args.pvm_scores,
        args.continuations,
        buckets=("low", "high"),
        max_prefixes_per_bucket=args.max_prefixes_per_bucket,
        max_prefix_tokens=args.max_prefix_tokens,
    )
    needed_ids = sorted({row.source_sample_id for row in prefix_rows})
    source_rows = load_source_rows(args.val_dataset, needed_ids)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    jobs = build_extraction_jobs(prefix_rows, source_rows, tokenizer)
    print(
        f"[phase3] prefixes={len(prefix_rows)} sources={len(jobs)} "
        f"tokens={sum(job.max_prefix_token_end for job in jobs)} out={out_dir}",
        flush=True,
    )
    source_entropy, extraction_meta = extract_source_entropy_with_vllm(
        jobs=jobs,
        model_path=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        parallel_size=args.parallel_size,
        batch_size=args.batch_size,
        entropy_chunk_size=args.entropy_chunk_size,
        enforce_eager=args.enforce_eager,
    )

    prefix_results = phase3_for_prefixes(prefix_rows, source_entropy, threshold=args.phase3_threshold)
    decile_rows = token_decile_summary(prefix_rows, source_entropy, threshold=args.phase3_threshold)
    summary = summarize_results(
        prefix_results,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    summary["relative_position_deciles"] = decile_rows
    summary["runtime"] = {
        "elapsed_seconds": time.time() - started,
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary["extraction"] = extraction_meta
    summary["inputs"] = {
        "pvm_scores": str(args.pvm_scores),
        "continuations": str(args.continuations),
        "val_dataset": str(args.val_dataset),
        "model": str(args.model),
        "phase3_threshold": float(args.phase3_threshold),
        "max_prefixes_per_bucket": int(args.max_prefixes_per_bucket),
        "max_prefix_tokens": int(args.max_prefix_tokens),
    }

    write_csv(out_dir / "prefix_phase3_rates.csv", prefix_results)
    write_csv(out_dir / "phase3_deciles.csv", decile_rows)
    write_csv(out_dir / "phase3_length_strata.csv", summary["length_strata"])
    write_json(out_dir / "phase3_summary.json", summary)
    plot_phase3_summary(prefix_results, summary, decile_rows, out_dir / "figure_phase3_high_vs_low.png")
    write_json(out_dir / "run_manifest.json", {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {
            "prefix_phase3_rates": str(out_dir / "prefix_phase3_rates.csv"),
            "phase3_summary": str(out_dir / "phase3_summary.json"),
            "phase3_deciles": str(out_dir / "phase3_deciles.csv"),
            "phase3_length_strata": str(out_dir / "phase3_length_strata.csv"),
            "figure": str(out_dir / "figure_phase3_high_vs_low.png"),
        },
        "summary": {
            "n_prefixes": len(prefix_results),
            "n_sources": len(jobs),
            "interpretation": summary["interpretation"],
            "high_minus_low": summary["prefix_mean_difference_high_minus_low"],
        },
        "args": vars(args) | {"output_dir": str(out_dir)},
    })

    print("[phase3] done")
    print(json.dumps({
        "out_dir": str(out_dir),
        "n_prefixes": len(prefix_results),
        "summary": summary["prefix_mean_difference_high_minus_low"],
        "interpretation": summary["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
