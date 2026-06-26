#!/usr/bin/env python3
"""Offline phenomenon discovery for PVM logprob-based prefix signals.

This script intentionally uses existing artifacts only. It does not call vLLM,
train models, or regenerate continuations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import textwrap
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PHASE3 = Path("results/pvm_phase3_20260624_170759/prefix_phase3_rates.csv")
DEFAULT_PVM_SCORES = Path("results/analysis_figures_20260623_183844/tables/pvm_prefix_scores.csv")
DEFAULT_CONTINUATIONS = Path("datasets/min_pvm_ppo_500_seed42_20260618/prefix_continuations_val.jsonl")

KEY_FIELDS = ("problem_id", "source_sample_id", "prefix_segments")
LOW_TEMP_MAX = 0.7
HIGH_TEMP_MIN = 0.9

COLORS = {
    "low": "#CC6F47",
    "mid": "#B8A037",
    "high": "#5477C4",
    "weak_or_none": "#7A828F",
    "low_temp_better": "#5477C4",
    "high_temp_better": "#CC6F47",
    "mixed_sensitive": "#B8A037",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--pvm-scores", type=Path, default=DEFAULT_PVM_SCORES)
    parser.add_argument("--continuations", type=Path, default=DEFAULT_CONTINUATIONS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-prefixes", type=int, default=0,
                        help="Restrict to the first N prefix-score rows for smoke tests.")
    parser.add_argument("--high-phi", type=float, default=0.60)
    parser.add_argument("--low-phi", type=float, default=0.40)
    parser.add_argument("--high-success", type=float, default=0.75)
    parser.add_argument("--low-success", type=float, default=0.25)
    parser.add_argument("--weak-sensitivity", type=float, default=0.25,
                        help="Temperature max-min success gap treated as weak/no effect.")
    parser.add_argument("--examples-per-class", type=int, default=5)
    return parser.parse_args()


def key_for(row: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row["problem_id"]),
        str(row["source_sample_id"]),
        int(row["prefix_segments"]),
    )


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_pct(value: float) -> str:
    if value != value:
        return "NA"
    return f"{100.0 * value:.1f}%"


def mean(values: Iterable[float]) -> float:
    clean = [float(v) for v in values if v == v]
    return statistics.fmean(clean) if clean else math.nan


def load_pvm_scores(path: Path, max_prefixes: int = 0) -> Tuple[List[Dict[str, Any]], set[Tuple[str, str, int]]]:
    rows = read_csv_rows(path)
    if max_prefixes > 0:
        rows = rows[:max_prefixes]
    result: List[Dict[str, Any]] = []
    allowed = set()
    for row in rows:
        parsed = dict(row)
        parsed["prefix_segments"] = to_int(row.get("prefix_segments"))
        parsed["source_individual_label"] = to_int(row.get("source_individual_label"))
        parsed["pvm_phi"] = to_float(row.get("pvm_phi"))
        parsed["observed_success_rate"] = to_float(row.get("observed_success_rate"))
        parsed["n_correct"] = to_int(row.get("n_correct"))
        parsed["n_total"] = to_int(row.get("n_total"))
        result.append(parsed)
        allowed.add(key_for(parsed))
    return result, allowed


def parse_phase3_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv_rows(path)
    result: List[Dict[str, Any]] = []
    for row in rows:
        parsed = dict(row)
        parsed["prefix_segments"] = to_int(row.get("prefix_segments"))
        for field in (
            "pvm_phi",
            "observed_success_rate",
            "phase3_rate",
            "mean_delta_entropy",
            "mean_prev_entropy",
            "mean_last_entropy",
            "mean_prev_sampled_logprob",
            "mean_last_sampled_logprob",
        ):
            parsed[field] = to_float(row.get(field))
        for field in ("n_correct", "n_total", "prefix_token_end", "phase3_tokens", "non_phase3_tokens"):
            parsed[field] = to_int(row.get(field))
        result.append(parsed)
    return result


def load_phase3(path: Path, allowed: set[Tuple[str, str, int]]) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], int]:
    rows = parse_phase3_rows(path)
    result: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for parsed in rows:
        k = key_for(parsed)
        if allowed and k not in allowed:
            continue
        result[k] = parsed
    return result, len(rows)


def continuation_temperature_stats(row: Mapping[str, Any]) -> Dict[float, Dict[str, float]]:
    grouped: Dict[float, Dict[str, float]] = defaultdict(lambda: {"n_correct": 0.0, "n_total": 0.0})
    for item in row.get("continuations", []):
        temp = float(item["temperature"])
        grouped[temp]["n_total"] += 1.0
        grouped[temp]["n_correct"] += 1.0 if bool(item.get("correct", False)) else 0.0
    result: Dict[float, Dict[str, float]] = {}
    for temp, values in grouped.items():
        total = values["n_total"]
        result[temp] = {
            "n_correct": values["n_correct"],
            "n_total": total,
            "success_rate": values["n_correct"] / total if total else math.nan,
        }
    return dict(sorted(result.items()))


def classify_temperature_response(rates: Mapping[float, float], sensitivity: float, weak_threshold: float) -> str:
    if sensitivity <= weak_threshold:
        return "weak_or_none"
    low_rates = [rate for temp, rate in rates.items() if temp <= LOW_TEMP_MAX]
    high_rates = [rate for temp, rate in rates.items() if temp >= HIGH_TEMP_MIN]
    low_mean = mean(low_rates)
    high_mean = mean(high_rates)
    if low_mean == low_mean and high_mean == high_mean:
        if low_mean - high_mean >= weak_threshold:
            return "low_temp_better"
        if high_mean - low_mean >= weak_threshold:
            return "high_temp_better"
    return "mixed_sensitive"


def summarize_temperatures(
    rows: Sequence[Mapping[str, Any]],
    allowed: set[Tuple[str, str, int]],
    pvm_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    phase3_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    weak_threshold: float,
) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in rows:
        parsed_key = (
            str(row["problem_id"]),
            str(row["source_sample_id"]),
            int(row["prefix_segments"]),
        )
        if allowed and parsed_key not in allowed:
            continue
        if int(row.get("n_total", 0)) != 32:
            continue
        per_temp = continuation_temperature_stats(row)
        rates = {temp: float(values["success_rate"]) for temp, values in per_temp.items()}
        if not rates:
            continue
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        sensitivity = max_rate - min_rate
        best_temps = [temp for temp, rate in rates.items() if rate == max_rate]
        worst_temps = [temp for temp, rate in rates.items() if rate == min_rate]
        base = pvm_by_key.get(parsed_key, {})
        phase3 = phase3_by_key.get(parsed_key, {})
        row_out: Dict[str, Any] = {
            "problem_id": row["problem_id"],
            "source_sample_id": row["source_sample_id"],
            "prefix_segments": int(row["prefix_segments"]),
            "prefix_stage": row.get("prefix_stage", base.get("prefix_stage", "")),
            "pvm_bucket": base.get("pvm_bucket", phase3.get("pvm_bucket", "")),
            "pvm_phi": to_float(base.get("pvm_phi", phase3.get("pvm_phi"))),
            "observed_success_rate": to_float(base.get("observed_success_rate", phase3.get("observed_success_rate"))),
            "phase3_rate": to_float(phase3.get("phase3_rate")),
            "n_total": int(row.get("n_total", 0)),
            "temperature_sensitivity": sensitivity,
            "best_temperature": ";".join(f"{temp:g}" for temp in best_temps),
            "worst_temperature": ";".join(f"{temp:g}" for temp in worst_temps),
            "temperature_response": classify_temperature_response(rates, sensitivity, weak_threshold),
            "prefix_token_end": int(row.get("prefix_token_end", phase3.get("prefix_token_end", 0))),
            "prefix_text": str(row.get("prefix_text", "")),
        }
        for temp, rate in rates.items():
            row_out[f"temp_{temp:g}_success_rate"] = rate
        result[parsed_key] = row_out
    return result


def length_stratum(prefix_token_end: int) -> str:
    if prefix_token_end < 256:
        return "short"
    if prefix_token_end < 768:
        return "mid"
    return "long"


def group_summary(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str], value_field: str) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    out: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        values = [to_float(row.get(value_field)) for row in items]
        entry = {field: key[idx] for idx, field in enumerate(group_fields)}
        entry.update({
            "n_prefixes": len(items),
            f"mean_{value_field}": mean(values),
            f"median_{value_field}": statistics.median([v for v in values if v == v]) if any(v == v for v in values) else math.nan,
        })
        out.append(entry)
    return out


def merged_phase3_rows(
    pvm_rows: Sequence[Mapping[str, Any]],
    phase3_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for base in pvm_rows:
        k = key_for(base)
        phase3 = phase3_by_key.get(k)
        if not phase3:
            continue
        prefix_token_end = to_int(phase3.get("prefix_token_end"))
        row = {
            "problem_id": base["problem_id"],
            "source_sample_id": base["source_sample_id"],
            "prefix_segments": int(base["prefix_segments"]),
            "prefix_stage": base.get("prefix_stage", ""),
            "pvm_bucket": base.get("pvm_bucket", phase3.get("pvm_bucket", "")),
            "pvm_phi": to_float(base.get("pvm_phi")),
            "observed_success_rate": to_float(base.get("observed_success_rate")),
            "n_correct": to_int(base.get("n_correct")),
            "n_total": to_int(base.get("n_total")),
            "prefix_token_end": prefix_token_end,
            "length_stratum": length_stratum(prefix_token_end),
            "phase3_rate": to_float(phase3.get("phase3_rate")),
            "mean_delta_entropy": to_float(phase3.get("mean_delta_entropy")),
            "phase3_tokens": to_int(phase3.get("phase3_tokens")),
            "non_phase3_tokens": to_int(phase3.get("non_phase3_tokens")),
        }
        rows.append(row)
    return rows


def select_examples(
    pvm_rows: Sequence[Mapping[str, Any]],
    phase3_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    temp_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    categories = {
        "high_phi_high_success": lambda r: to_float(r["pvm_phi"]) >= args.high_phi and to_float(r["observed_success_rate"]) >= args.high_success,
        "low_phi_low_success": lambda r: to_float(r["pvm_phi"]) <= args.low_phi and to_float(r["observed_success_rate"]) <= args.low_success,
        "high_phi_low_success": lambda r: to_float(r["pvm_phi"]) >= args.high_phi and to_float(r["observed_success_rate"]) <= args.low_success,
        "low_phi_high_success": lambda r: to_float(r["pvm_phi"]) <= args.low_phi and to_float(r["observed_success_rate"]) >= args.high_success,
    }
    sorters = {
        "high_phi_high_success": lambda r: (-to_float(r["pvm_phi"]), -to_float(r["observed_success_rate"])),
        "low_phi_low_success": lambda r: (to_float(r["pvm_phi"]), to_float(r["observed_success_rate"])),
        "high_phi_low_success": lambda r: (-to_float(r["pvm_phi"]), to_float(r["observed_success_rate"])),
        "low_phi_high_success": lambda r: (to_float(r["pvm_phi"]), -to_float(r["observed_success_rate"])),
    }
    examples: List[Dict[str, Any]] = []
    for category, predicate in categories.items():
        candidates = [row for row in pvm_rows if predicate(row)]
        candidates = sorted(candidates, key=sorters[category])[:max(0, int(args.examples_per_class))]
        for row in candidates:
            k = key_for(row)
            phase3 = phase3_by_key.get(k, {})
            temp = temp_by_key.get(k, {})
            prefix_text = str(temp.get("prefix_text", ""))
            examples.append({
                "phenomenon_class": category,
                "problem_id": row["problem_id"],
                "source_sample_id": row["source_sample_id"],
                "prefix_segments": int(row["prefix_segments"]),
                "prefix_stage": row.get("prefix_stage", ""),
                "pvm_phi": to_float(row["pvm_phi"]),
                "observed_success_rate": to_float(row["observed_success_rate"]),
                "phase3_rate": to_float(phase3.get("phase3_rate")),
                "temperature_sensitivity": to_float(temp.get("temperature_sensitivity")),
                "temperature_response": temp.get("temperature_response", ""),
                "best_temperature": temp.get("best_temperature", ""),
                "worst_temperature": temp.get("worst_temperature", ""),
                "prefix_excerpt": textwrap.shorten(" ".join(prefix_text.split()), width=360, placeholder="..."),
            })
    return examples


def plot_phase3_vs_phi(path: Path, phase3_rows: Sequence[Mapping[str, Any]]) -> None:
    buckets = [bucket for bucket in ("low", "high") if any(row.get("pvm_bucket") == bucket for row in phase3_rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    fig.suptitle("PVM confidence separates distribution perturbation", fontsize=13, fontweight="bold")

    ax = axes[0]
    means = []
    for idx, bucket in enumerate(buckets):
        vals = [to_float(row["phase3_rate"]) for row in phase3_rows if row.get("pvm_bucket") == bucket]
        means.append(mean(vals))
        ax.scatter([idx] * len(vals), vals, s=12, alpha=0.35, color=COLORS.get(bucket, "#777777"), edgecolors="none")
    ax.bar(range(len(buckets)), means, color=[COLORS.get(bucket, "#777777") for bucket in buckets], alpha=0.55)
    ax.set_xticks(range(len(buckets)), [bucket.title() for bucket in buckets])
    ax.set_ylabel("Phase3 rate")
    ax.set_ylim(0, max(0.45, max([to_float(row["phase3_rate"]) for row in phase3_rows] or [0]) * 1.15))
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Overall")

    ax = axes[1]
    stages = [stage for stage in ("early", "middle", "late") if any(row.get("prefix_stage") == stage for row in phase3_rows)]
    width = 0.36
    xs = range(len(stages))
    for offset, bucket in [(-width / 2, "low"), (width / 2, "high")]:
        vals = []
        for stage in stages:
            vals.append(mean(
                to_float(row["phase3_rate"])
                for row in phase3_rows
                if row.get("prefix_stage") == stage and row.get("pvm_bucket") == bucket
            ))
        ax.bar([x + offset for x in xs], vals, width=width, label=bucket.title(),
               color=COLORS.get(bucket, "#777777"), alpha=0.75)
    ax.set_xticks(list(xs), [stage.title() for stage in stages])
    ax.set_ylabel("Mean phase3 rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("By prefix stage")
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def plot_temperature_sensitivity(path: Path, temp_rows: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    fig.suptitle("Temperature sensitivity is sparse and heterogeneous", fontsize=13, fontweight="bold")

    ax = axes[0]
    with_phase3: List[Mapping[str, Any]] = []
    without_phase3: List[Mapping[str, Any]] = []
    for row in temp_rows:
        phase3 = to_float(row.get("phase3_rate"))
        if phase3 == phase3:
            with_phase3.append(row)
        else:
            without_phase3.append(row)
    if without_phase3:
        ax.scatter(
            [to_float(row.get("pvm_phi")) for row in without_phase3],
            [to_float(row.get("temperature_sensitivity")) for row in without_phase3],
            color="#A5ABB5",
            s=16,
            alpha=0.45,
            edgecolors="none",
            label="No phase3 field",
        )
    scatter = None
    if with_phase3:
        scatter = ax.scatter(
            [to_float(row.get("pvm_phi")) for row in with_phase3],
            [to_float(row.get("temperature_sensitivity")) for row in with_phase3],
            c=[to_float(row.get("phase3_rate")) for row in with_phase3],
            cmap="viridis",
            s=18,
            alpha=0.75,
            edgecolors="none",
            label="With phase3",
        )
    ax.set_xlabel("PVM phi")
    ax.set_ylabel("Temperature sensitivity")
    ax.grid(True, alpha=0.3)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Phase3 rate")
    if without_phase3:
        ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    counts = Counter(str(row.get("temperature_response", "")) for row in temp_rows)
    order = ["weak_or_none", "low_temp_better", "high_temp_better", "mixed_sensitive"]
    labels = [item for item in order if counts.get(item, 0)]
    ax.bar(range(len(labels)), [counts[label] for label in labels],
           color=[COLORS.get(label, "#777777") for label in labels], alpha=0.8)
    ax.set_xticks(range(len(labels)), [label.replace("_", "\n") for label in labels])
    ax.set_ylabel("Prefixes")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Response type")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def rows_to_markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], max_rows: int = 12) -> str:
    selected = list(rows[:max_rows])
    if not selected:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in selected:
        values = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                if "rate" in field or "phi" in field or "sensitivity" in field:
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    phase3_rows: Sequence[Mapping[str, Any]],
    phase3_summary_rows: Sequence[Mapping[str, Any]],
    length_stage_rows: Sequence[Mapping[str, Any]],
    temp_rows: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    pvm_without_phase3: int,
    phase3_available: int,
) -> None:
    low_phase3 = mean(to_float(row["phase3_rate"]) for row in phase3_rows if row.get("pvm_bucket") == "low")
    high_phase3 = mean(to_float(row["phase3_rate"]) for row in phase3_rows if row.get("pvm_bucket") == "high")
    high_sens = [row for row in temp_rows if to_float(row.get("temperature_sensitivity")) > args.weak_sensitivity]
    high_sens_low_phi = mean(1.0 if to_float(row.get("pvm_phi")) <= args.low_phi else 0.0 for row in high_sens)
    high_sens_stages = Counter(str(row.get("prefix_stage", "")) for row in high_sens)
    dominant_stage, dominant_stage_count = high_sens_stages.most_common(1)[0] if high_sens_stages else ("", 0)
    response_counts = Counter(str(row.get("temperature_response", "")) for row in temp_rows)
    example_counts = Counter(str(row.get("phenomenon_class", "")) for row in examples)
    by_length_stage: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
    for row in length_stage_rows:
        n = int(row.get("n_prefixes", 0))
        if n < 10:
            continue
        by_length_stage[(str(row.get("length_stratum", "")), str(row.get("prefix_stage", "")))][str(row.get("pvm_bucket", ""))] = (
            n,
            to_float(row.get("mean_phase3_rate")),
        )
    comparable = [
        values for values in by_length_stage.values()
        if "low" in values and "high" in values
    ]
    supportive = [
        values for values in comparable
        if values["low"][1] > values["high"][1]
    ]

    top_sensitive = sorted(
        temp_rows,
        key=lambda row: (-to_float(row.get("temperature_sensitivity")), to_float(row.get("pvm_phi"))),
    )[:8]

    text = f"""# PVM Logprob Phenomena Summary

This run is offline-only. It uses existing PVM prefix scores, validation continuations, and phase3 entropy artifacts. It does not train, call vLLM, or regenerate continuations.

## Main Findings

1. Low-PVM prefixes show higher distribution perturbation. Mean phase3 rate is {format_pct(low_phase3)} for low-PVM prefixes versus {format_pct(high_phase3)} for high-PVM prefixes. Among populated length/stage cells, {len(supportive)} / {len(comparable)} comparable cells keep the low-PVM > high-PVM direction; sparse cells should be treated as examples, not proof.
2. PVM failures are interpretable cases worth inspecting, not just noise. The selected examples include {example_counts.get('high_phi_low_success', 0)} high-phi/low-success prefixes and {example_counts.get('low_phi_high_success', 0)} low-phi/high-success prefixes.
3. Temperature sensitivity is sparse but sharp. {len(high_sens)} / {len(temp_rows)} prefixes have sensitivity above {args.weak_sensitivity:.2f}. It is not just a low-PVM phenomenon: {format_pct(high_sens_low_phi)} are low-phi prefixes, and the largest group is `{dominant_stage}` ({dominant_stage_count} prefixes).

## Phase3 vs PVM

{rows_to_markdown_table(phase3_summary_rows, ['pvm_bucket', 'n_prefixes', 'mean_phase3_rate', 'median_phase3_rate'])}

## Length/Stage Control

{rows_to_markdown_table(length_stage_rows, ['length_stratum', 'prefix_stage', 'pvm_bucket', 'n_prefixes', 'mean_phase3_rate'])}

## Temperature Response Counts

| response_type | n_prefixes |
| --- | --- |
"""
    for label, count in sorted(response_counts.items()):
        text += f"| {label} | {count} |\n"

    text += f"""
## Most Temperature-Sensitive Prefixes

{rows_to_markdown_table(top_sensitive, ['problem_id', 'prefix_stage', 'pvm_phi', 'observed_success_rate', 'phase3_rate', 'temperature_sensitivity', 'temperature_response', 'best_temperature', 'worst_temperature'])}

## Output Files

- `phenomena_examples.csv`: representative high/low phi and success/failure examples.
- `temperature_sensitivity.csv`: per-prefix temperature response table, restricted to n_total=32 prefixes.
- `fig_phase3_vs_phi.png`: phase3 perturbation split by PVM bucket.
- `fig_temperature_sensitivity.png`: temperature sensitivity against PVM phi and response-type counts.

## Data Quality Checks

- Prefix-score rows used: {len(temp_rows)}
- Phase3 artifact rows available: {phase3_available}
- Phase3 rows matched to this run: {len(phase3_rows)}
- PVM rows without phase3 fields: {pvm_without_phase3} (expected when the phase3 artifact covers only high/low PVM buckets)
- Continuation rows in `temperature_sensitivity.csv` are restricted to `n_total == 32`.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path("results") / f"logprob_phenomena_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pvm_rows, allowed = load_pvm_scores(args.pvm_scores, max_prefixes=args.max_prefixes)
    pvm_by_key = {key_for(row): row for row in pvm_rows}
    phase3_by_key, phase3_available = load_phase3(args.phase3, allowed)
    continuation_rows = read_jsonl(args.continuations)
    temp_by_key = summarize_temperatures(
        continuation_rows,
        allowed,
        pvm_by_key,
        phase3_by_key,
        weak_threshold=float(args.weak_sensitivity),
    )

    phase3_rows = merged_phase3_rows(pvm_rows, phase3_by_key)
    pvm_without_phase3 = max(0, len(pvm_rows) - len(phase3_rows))
    temp_rows = [temp_by_key[k] for k in pvm_by_key if k in temp_by_key]
    examples = select_examples(pvm_rows, phase3_by_key, temp_by_key, args)

    phase3_summary_rows = group_summary(phase3_rows, ["pvm_bucket"], "phase3_rate")
    length_stage_rows = group_summary(phase3_rows, ["length_stratum", "prefix_stage", "pvm_bucket"], "phase3_rate")
    temp_response_rows = group_summary(temp_rows, ["temperature_response"], "temperature_sensitivity")

    write_csv(out_dir / "phase3_bucket_summary.csv", phase3_summary_rows)
    write_csv(out_dir / "phase3_length_stage_summary.csv", length_stage_rows)
    write_csv(out_dir / "temperature_response_summary.csv", temp_response_rows)
    write_csv(out_dir / "phenomena_examples.csv", examples)
    write_csv(out_dir / "temperature_sensitivity.csv", temp_rows)

    plot_phase3_vs_phi(out_dir / "fig_phase3_vs_phi.png", phase3_rows)
    plot_temperature_sensitivity(out_dir / "fig_temperature_sensitivity.png", temp_rows)

    manifest = {
        "started_at": timestamp,
        "inputs": {
            "phase3": str(args.phase3),
            "pvm_scores": str(args.pvm_scores),
            "continuations": str(args.continuations),
        },
        "parameters": {
            "max_prefixes": int(args.max_prefixes),
            "high_phi": float(args.high_phi),
            "low_phi": float(args.low_phi),
            "high_success": float(args.high_success),
            "low_success": float(args.low_success),
            "weak_sensitivity": float(args.weak_sensitivity),
            "examples_per_class": int(args.examples_per_class),
        },
        "outputs": {
            "summary": str(out_dir / "summary.md"),
            "phenomena_examples": str(out_dir / "phenomena_examples.csv"),
            "temperature_sensitivity": str(out_dir / "temperature_sensitivity.csv"),
            "phase3_vs_phi": str(out_dir / "fig_phase3_vs_phi.png"),
            "temperature_sensitivity_figure": str(out_dir / "fig_temperature_sensitivity.png"),
        },
        "counts": {
            "pvm_prefixes": len(pvm_rows),
            "phase3_available": phase3_available,
            "phase3_matched": len(phase3_rows),
            "pvm_without_phase3": pvm_without_phase3,
            "temperature_rows": len(temp_rows),
            "examples": len(examples),
        },
    }
    write_json(out_dir / "run_manifest.json", manifest)
    write_summary(
        out_dir / "summary.md",
        args,
        phase3_rows,
        phase3_summary_rows,
        length_stage_rows,
        temp_rows,
        examples,
        pvm_without_phase3,
        phase3_available,
    )

    print(f"[logprob_phenomena] wrote {out_dir}")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
