#!/usr/bin/env python3
"""Audit and summarize the 200-prompt online evaluation outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.jsonl import sample_prefix


METRICS = [
    "majority_accuracy",
    "pass_at_1_accuracy",
    "individual_accuracy",
    "ece",
    "brier",
    "nll",
    "mean_confidence",
    "mean_answer_entropy",
    "average_tokens",
    "total_tokens",
    "wall_seconds",
]


def dataset_audit(input_path: str, train_path: str) -> Dict[str, Any]:
    train_ids = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_ids.add(sample_prefix(str(json.loads(line).get("sample_id", ""))))

    ids = set()
    temps = Counter()
    vote_ids = Counter()
    rows = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            ids.add(sample_prefix(str(row.get("sample_id", ""))))
            if "temperature" in row:
                temps[str(float(row["temperature"]))] += 1
            metadata = row.get("metadata") or {}
            if "vote_id" in metadata:
                vote_ids[str(metadata["vote_id"])] += 1

    return {
        "input_path": input_path,
        "train_path": train_path,
        "n_rows": rows,
        "n_prompts": len(ids),
        "n_temperatures": len(temps),
        "temperature_counts": dict(sorted(temps.items(), key=lambda item: float(item[0]))),
        "n_vote_ids": len(vote_ids),
        "vote_id_counts": dict(sorted(vote_ids.items(), key=lambda item: int(item[0]))),
        "train_overlap_count": len(ids & train_ids),
        "train_overlap_examples": sorted(ids & train_ids)[:10],
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for key in METRICS:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        result[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def _temperature_totals(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        counter.update({str(k): int(v) for k, v in row.get("selected_temperature_distribution", {}).items()})
    return dict(sorted(counter.items(), key=lambda item: float(item[0])))


def _q_segment_temperature_distribution(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    segment_counter: Dict[int, Counter] = defaultdict(Counter)
    stage_counter: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for prediction in row.get("predictions", []):
            for decision in prediction.get("q_decisions", []):
                temp = str(float(decision["temperature"]))
                segment_counter[int(decision["segment_index"])][temp] += 1
                stage_counter[str(decision.get("stage", "unknown"))][temp] += 1
    return {
        "by_segment_index": {
            str(idx): dict(sorted(counter.items(), key=lambda item: float(item[0])))
            for idx, counter in sorted(segment_counter.items())
        },
        "by_stage": {
            stage: dict(sorted(counter.items(), key=lambda item: float(item[0])))
            for stage, counter in sorted(stage_counter.items())
        },
    }


def summarize_result_dir(result_dir: str, seeds: Sequence[int],
                         input_path: str, train_path: str) -> Dict[str, Any]:
    root = Path(result_dir)
    by_method: Dict[str, List[Dict[str, Any]]] = {
        "fixed_temperature_1.0": [],
        "prefix_q_argmax_selector": [],
    }
    missing: List[str] = []
    for seed in seeds:
        specs = [
            ("fixed_temperature_1.0", root / f"fixed_t1_seed{seed}.json"),
            ("prefix_q_argmax_selector", root / f"q_selector_seed{seed}.json"),
        ]
        for method, path in specs:
            if not path.exists():
                missing.append(str(path))
                continue
            by_method[method].append(_load_json(path))

    audit = dataset_audit(input_path, train_path)
    methods: Dict[str, Any] = {}
    for method, rows in by_method.items():
        methods[method] = {
            "per_seed": [
                {key: row.get(key) for key in ["seed", "n_prompts", "num_votes", *METRICS]}
                for row in sorted(rows, key=lambda item: int(item.get("seed", 0)))
            ],
            "aggregate": _aggregate(rows),
            "selected_temperature_distribution_total": _temperature_totals(rows),
        }
        if method == "prefix_q_argmax_selector":
            methods[method]["q_temperature_distribution"] = _q_segment_temperature_distribution(rows)

    return {
        "result_dir": result_dir,
        "seeds": [int(seed) for seed in seeds],
        "dataset_audit": audit,
        "missing_outputs": missing,
        "methods": methods,
    }


def markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# Eval200 Online Summary",
        "",
        f"- result_dir: `{summary['result_dir']}`",
        f"- prompts: {summary['dataset_audit']['n_prompts']}",
        f"- train_overlap_count: {summary['dataset_audit']['train_overlap_count']}",
        "",
    ]
    if summary["missing_outputs"]:
        lines.extend(["## Missing Outputs", ""])
        lines.extend(f"- `{path}`" for path in summary["missing_outputs"])
        lines.append("")
    for method, data in summary["methods"].items():
        lines.extend([
            f"## {method}",
            "",
            "| seed | n | maj_acc | pass@1 | ind_acc | ECE | Brier | NLL | avg_tokens | wall_s |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in data["per_seed"]:
            lines.append(
                f"| {row.get('seed')} | {row.get('n_prompts')} | "
                f"{float(row.get('majority_accuracy', 0.0)):.4f} | "
                f"{float(row.get('pass_at_1_accuracy', 0.0)):.4f} | "
                f"{float(row.get('individual_accuracy', 0.0)):.4f} | "
                f"{float(row.get('ece', 0.0)):.4f} | "
                f"{float(row.get('brier', 0.0)):.4f} | "
                f"{float(row.get('nll', 0.0)):.4f} | "
                f"{float(row.get('average_tokens', 0.0)):.1f} | "
                f"{float(row.get('wall_seconds', 0.0)):.1f} |"
            )
        lines.extend(["", "Mean +/- std:", ""])
        for key in ["majority_accuracy", "pass_at_1_accuracy", "individual_accuracy", "ece", "brier", "nll", "average_tokens"]:
            agg = data["aggregate"].get(key)
            if agg:
                lines.append(f"- {key}: {agg['mean']:.6f} +/- {agg['std']:.6f}")
        lines.extend(["", "Temperature total:", "", "```json"])
        lines.append(json.dumps(data["selected_temperature_distribution_total"], indent=2))
        lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--input", default="datasets/all_5_sub_200.jsonl")
    parser.add_argument("--train", default="datasets/train_5_small_500.jsonl")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--audit-output", default=None)
    args = parser.parse_args()

    summary = summarize_result_dir(args.result_dir, args.seeds, args.input, args.train)
    root = Path(args.result_dir)
    output = Path(args.output) if args.output else root / "summary.json"
    markdown_output = Path(args.markdown_output) if args.markdown_output else root / "summary.md"
    audit_output = Path(args.audit_output) if args.audit_output else root / "dataset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_output.write_text(markdown_summary(summary), encoding="utf-8")
    audit_output.write_text(json.dumps(summary["dataset_audit"], indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "markdown_output": str(markdown_output),
        "audit_output": str(audit_output),
        "missing_outputs": summary["missing_outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
