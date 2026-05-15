#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

# Allow direct execution: python scripts/subsample_jsonl.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.jsonl import add_groupby_arg, group_rows, load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsample a JSONL dataset for quick experiments.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=None, help="Keep this fraction of rows/groups.")
    parser.add_argument("--max-rows", type=int, default=None, help="Hard cap for output rows after sampling.")
    add_groupby_arg(parser)
    args = parser.parse_args()

    if args.ratio is None and args.max_rows is None:
        raise ValueError("Set at least one of --ratio or --max-rows")

    rows = load_jsonl(args.input)
    if not rows:
        raise RuntimeError("Input dataset is empty")

    rnd = random.Random(args.seed)

    grouped = group_rows(rows, args.group_by)

    keys = list(grouped.keys())
    rnd.shuffle(keys)

    if args.ratio is not None:
        if not (0.0 < args.ratio <= 1.0):
            raise ValueError("--ratio must be in (0, 1]")
        keep_n = max(1, int(len(keys) * args.ratio))
    else:
        keep_n = len(keys)

    selected_keys = keys[:keep_n]
    sampled: List[dict] = []
    for k in selected_keys:
        sampled.extend(grouped[k])

    rnd.shuffle(sampled)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        sampled = sampled[: args.max_rows]

    write_jsonl(args.output, sampled)

    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(sampled)}")
    print(f"group_by={args.group_by}")
    print(f"ratio={args.ratio}")
    print(f"max_rows={args.max_rows}")


if __name__ == "__main__":
    main()
