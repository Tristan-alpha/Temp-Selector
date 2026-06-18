#!/usr/bin/env python3
"""Validate the fixed 500-problem, 60,000-trajectory experimental split."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.jsonl import load_jsonl, sample_prefix


EXPECTED_TEMPERATURES = [round(0.1 * i, 1) for i in range(1, 16)]


def validate_split(train_path: str, val_path: str, test_path: str) -> dict:
    split_paths = {"train": train_path, "val": val_path, "test": test_path}
    expected_problems = {"train": 400, "val": 50, "test": 50}
    expected_rows = {"train": 48000, "val": 6000, "test": 6000}
    problem_sets = {}
    summary = {}
    for split, path in split_paths.items():
        rows = load_jsonl(path)
        grouped = defaultdict(list)
        for row in rows:
            grouped[sample_prefix(str(row.get("sample_id", "")))].append(row)
        if len(rows) != expected_rows[split]:
            raise ValueError(f"{split}: expected {expected_rows[split]} rows, got {len(rows)}")
        if len(grouped) != expected_problems[split]:
            raise ValueError(
                f"{split}: expected {expected_problems[split]} problems, got {len(grouped)}"
            )
        for problem, bucket in grouped.items():
            if len(bucket) != 120:
                raise ValueError(f"{split}/{problem}: expected 120 trajectories, got {len(bucket)}")
            counts = Counter(round(float(row["temperature"]), 1) for row in bucket)
            if sorted(counts) != EXPECTED_TEMPERATURES:
                raise ValueError(f"{split}/{problem}: incomplete temperature grid")
            if any(counts[temp] != 8 for temp in EXPECTED_TEMPERATURES):
                raise ValueError(f"{split}/{problem}: expected 8 votes per temperature")
        problem_sets[split] = set(grouped)
        summary[split] = {"rows": len(rows), "problems": len(grouped)}
    if problem_sets["train"] & problem_sets["val"] or \
       problem_sets["train"] & problem_sets["test"] or \
       problem_sets["val"] & problem_sets["test"]:
        raise ValueError("problem-level leakage detected across train/val/test")
    summary["total"] = {
        "rows": sum(item["rows"] for item in summary.values()),
        "problems": sum(item["problems"] for item in summary.values()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/train_5_small_500.jsonl")
    parser.add_argument("--val", default="datasets/val_5_small_500.jsonl")
    parser.add_argument("--test", default="datasets/test_5_small_500.jsonl")
    args = parser.parse_args()
    print(validate_split(args.train, args.val, args.test))


if __name__ == "__main__":
    main()
