#!/usr/bin/env python3
"""Evaluate fixed-temperature self-consistency calibration from dataset JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.answer_verifier import extract_answer, verify_answer_by_value
from utils.calibration import (
    answer_entropy,
    binary_nll,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    selective_risk_curve,
)
from utils.jsonl import load_jsonl, sample_prefix


NO_ANSWER = "<NO_ANSWER>"


def vote_index(row: Dict[str, Any]) -> int:
    metadata = row.get("metadata", {})
    if "vote_id" in metadata:
        return int(metadata["vote_id"])
    sample_id = str(row.get("sample_id", ""))
    if "_v" in sample_id:
        try:
            return int(sample_id.rsplit("_v", 1)[1])
        except ValueError:
            return 0
    return 0


def extracted_vote_answer(row: Dict[str, Any]) -> str:
    cached = row.get("_calibration_extracted_answer")
    if cached is not None:
        return str(cached)
    response = row.get("response")
    if response is not None:
        answer = extract_answer(str(response))
        result = answer if answer else NO_ANSWER
        row["_calibration_extracted_answer"] = result
        return result
    metadata_answer = row.get("metadata", {}).get("extracted_answer")
    result = str(metadata_answer) if metadata_answer else NO_ANSWER
    row["_calibration_extracted_answer"] = result
    return result


def row_individual_correct(row: Dict[str, Any]) -> int:
    if "individual_label" in row:
        return 1 - int(row["individual_label"])
    metadata = row.get("metadata", {})
    if "individual_correct" in metadata:
        return int(bool(metadata["individual_correct"]))
    answer = extracted_vote_answer(row)
    gold = str(metadata.get("gold_answer", ""))
    return int(answer != NO_ANSWER and verify_answer_by_value(answer, gold))


def majority_vote_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=vote_index)
    answers = [extracted_vote_answer(row) for row in ordered]
    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0] if counts else (NO_ANSWER, 0)
    gold = str(ordered[0].get("metadata", {}).get("gold_answer", "")) if ordered else ""
    majority_correct = int(
        majority_answer != NO_ANSWER and verify_answer_by_value(majority_answer, gold)
    )
    token_counts = [len(row.get("token_ids", [])) for row in ordered]
    individual_correct = [row_individual_correct(row) for row in ordered]
    return {
        "problem_id": sample_prefix(str(ordered[0].get("sample_id", ""))) if ordered else "",
        "temperature": float(ordered[0].get("temperature", 0.0)) if ordered else 0.0,
        "num_votes": len(ordered),
        "majority_answer": majority_answer,
        "majority_count": int(majority_count),
        "majority_correct": majority_correct,
        "sc_confidence": majority_count / max(1, len(ordered)),
        "answer_entropy": answer_entropy(answers),
        "answers": answers,
        "individual_correct": individual_correct,
        "token_counts": token_counts,
        "total_tokens": int(sum(token_counts)),
        "first_vote_correct": individual_correct[0] if individual_correct else 0,
    }


def group_dataset_votes(rows: Iterable[Dict[str, Any]]) -> Dict[tuple[str, float], List[Dict[str, Any]]]:
    grouped: Dict[tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pid = sample_prefix(str(row.get("sample_id", "")))
        temp = float(row.get("temperature", 0.0))
        grouped[(pid, temp)].append(row)
    return dict(grouped)


def metrics_from_vote_summaries(summaries: Sequence[Dict[str, Any]],
                                n_bins: int = 10) -> Dict[str, Any]:
    confidences = [float(item["sc_confidence"]) for item in summaries]
    correctness = [int(item["majority_correct"]) for item in summaries]
    token_counts = [
        int(token)
        for item in summaries
        for token in item.get("token_counts", [])
    ]
    individual_correct = [
        int(value)
        for item in summaries
        for value in item.get("individual_correct", [])
    ]
    num_votes = [int(item["num_votes"]) for item in summaries]
    if not summaries:
        return {
            "pass_at_1_accuracy": 0.0,
            "first_vote_accuracy": 0.0,
            "majority_vote_accuracy": 0.0,
            "self_consistency_confidence": 0.0,
            "self_consistency_ece": 0.0,
            "self_consistency_brier": 0.0,
            "self_consistency_nll": 0.0,
            "answer_entropy": 0.0,
            "mean_tokens": 0.0,
            "median_tokens": 0.0,
            "num_samples": 0,
            "n_groups": 0,
            "num_votes": 0,
            "mean_num_votes": 0.0,
            "reliability_bins": reliability_bins([], [], n_bins=n_bins),
            "selective_risk_curve": [],
        }

    unique_vote_counts = sorted(set(num_votes))
    return {
        "pass_at_1_accuracy": sum(individual_correct) / max(1, len(individual_correct)),
        "first_vote_accuracy": sum(int(item["first_vote_correct"]) for item in summaries) / len(summaries),
        "majority_vote_accuracy": sum(correctness) / len(correctness),
        "self_consistency_confidence": sum(confidences) / len(confidences),
        "self_consistency_ece": expected_calibration_error(confidences, correctness, n_bins=n_bins),
        "self_consistency_brier": brier_score(confidences, correctness),
        "self_consistency_nll": binary_nll(confidences, correctness),
        "answer_entropy": sum(float(item["answer_entropy"]) for item in summaries) / len(summaries),
        "mean_tokens": sum(token_counts) / max(1, len(token_counts)),
        "median_tokens": float(statistics.median(token_counts)) if token_counts else 0.0,
        "num_samples": len(individual_correct),
        "n_groups": len(summaries),
        "num_votes": unique_vote_counts[0] if len(unique_vote_counts) == 1 else None,
        "mean_num_votes": sum(num_votes) / len(num_votes),
        "reliability_bins": reliability_bins(confidences, correctness, n_bins=n_bins),
        "selective_risk_curve": selective_risk_curve(confidences, correctness),
    }


def temperature_sweep_from_rows(rows: Sequence[Dict[str, Any]],
                                temperatures: Sequence[float] | None = None,
                                n_bins: int = 10) -> Dict[str, Any]:
    grouped = group_dataset_votes(rows)
    available_temps = sorted({temp for _, temp in grouped})
    selected_temps = [float(t) for t in temperatures] if temperatures is not None else available_temps
    by_temperature: List[Dict[str, Any]] = []
    reliability_by_temp: Dict[str, Any] = {}
    for temp in selected_temps:
        summaries = [
            majority_vote_summary(bucket)
            for (pid, group_temp), bucket in sorted(grouped.items())
            if group_temp == float(temp)
        ]
        metrics = metrics_from_vote_summaries(summaries, n_bins=n_bins)
        reliability_by_temp[str(float(temp))] = metrics.pop("reliability_bins")
        by_temperature.append({
            "temperature": float(temp),
            **metrics,
        })
    return {
        "temperatures": by_temperature,
        "reliability_bins": reliability_by_temp,
    }


def markdown_table(result: Dict[str, Any]) -> str:
    lines = [
        "| Temperature | Pass@1 | Majority acc | SC confidence | ECE | Brier | NLL | Entropy | Mean tokens | Median tokens |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["temperatures"]:
        lines.append(
            f"| {row['temperature']:.1f} | "
            f"{row['pass_at_1_accuracy']:.4f} | "
            f"{row['majority_vote_accuracy']:.4f} | "
            f"{row['self_consistency_confidence']:.4f} | "
            f"{row['self_consistency_ece']:.4f} | "
            f"{row['self_consistency_brier']:.4f} | "
            f"{row['self_consistency_nll']:.4f} | "
            f"{row['answer_entropy']:.4f} | "
            f"{row['mean_tokens']:.1f} | "
            f"{row['median_tokens']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_temperature_sweep(config_path: str, split: str = "test",
                               input_path: str | None = None,
                               temperatures: Sequence[float] | None = None,
                               n_bins: int = 10) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_path = input_path or cfg["paths"][f"{split}_dataset"]
    if temperatures is None:
        temperatures = cfg.get("inference", {}).get("temperature_grid")
    rows = load_jsonl(data_path)
    result = temperature_sweep_from_rows(rows, temperatures=temperatures, n_bins=n_bins)
    result.update({
        "config": config_path,
        "split": split,
        "input_path": data_path,
        "n_bins": n_bins,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="results/temperature_sweep_calibration.json")
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--temperatures", nargs="*", type=float, default=None)
    args = parser.parse_args()

    result = evaluate_temperature_sweep(
        args.config,
        split=args.split,
        input_path=args.input,
        temperatures=args.temperatures,
        n_bins=args.n_bins,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_output = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(markdown_table(result), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "markdown": str(markdown_output),
        "n_temperatures": len(result["temperatures"]),
    }, indent=2))


if __name__ == "__main__":
    main()
