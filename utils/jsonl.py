"""Shared JSONL helpers used by split_jsonl and subsample_jsonl."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

GROUP_BY_CHOICES = ["none", "sample_id", "sample_prefix", "question", "prompt"]


def add_groupby_arg(parser: argparse.ArgumentParser, default: str = "sample_prefix") -> None:
    parser.add_argument("--group-by", choices=GROUP_BY_CHOICES, default=default)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_vote_suffix(sample_id: str) -> str:
    """Remove '_vN' suffix from sample_id."""
    return re.sub(r"_v\d+$", "", sample_id)


def sample_prefix(sample_id: str) -> str:
    """Extract the base prompt id from a sample id.

    ``"q1_t0.2_v0"`` → ``"q1"``,  ``"q1_t0.2"`` → ``"q1"``.
    """
    sid = strip_vote_suffix(sample_id)
    marker = "_t"
    if marker in sid:
        base, suffix = sid.rsplit(marker, 1)
        try:
            float(suffix)
            return base
        except ValueError:
            return sample_id
    return sample_id


def row_group_key(row: dict, group_by: str) -> str:
    if group_by == "none":
        return row.get("sample_id", "") + "#row"
    if group_by == "sample_id":
        return str(row.get("sample_id", ""))
    if group_by == "sample_prefix":
        return sample_prefix(str(row.get("sample_id", "")))
    if group_by == "question":
        return str(row.get("question", row.get("problem", row.get("prompt", ""))))
    if group_by == "prompt":
        return str(row.get("prompt", row.get("question", row.get("problem", ""))))
    raise ValueError(f"Unsupported group_by: {group_by}")


def group_rows(rows: List[dict], group_by: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row_group_key(row, group_by)].append(row)
    return grouped


def split_by_group(rows: List[dict], eval_ratio: float, seed: int, group_by: str) -> Tuple[List[dict], List[dict]]:
    if not rows:
        return [], []

    grouped = group_rows(rows, group_by)
    keys = list(grouped.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)

    n_eval_groups = max(1, int(len(keys) * eval_ratio))
    n_eval_groups = min(n_eval_groups, len(keys) - 1) if len(keys) > 1 else 1

    eval_key_set = set(keys[:n_eval_groups])

    train_rows: List[dict] = []
    eval_rows: List[dict] = []

    for key, bucket in grouped.items():
        if key in eval_key_set:
            eval_rows.extend(bucket)
        else:
            train_rows.extend(bucket)

    return train_rows, eval_rows
