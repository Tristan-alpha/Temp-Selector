#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.jsonl import add_groupby_arg, load_jsonl, split_by_group, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL into train/val/test with group-aware split.")
    parser.add_argument("--config", default=None, help="Optional YAML config for default paths")
    parser.add_argument("--input", default=None, help="Override paths.all_dataset from config")
    parser.add_argument("--train-output", default=None, help="Override paths.train_dataset from config")
    parser.add_argument("--val-output", default=None, help="Override paths.val_dataset from config")
    parser.add_argument("--test-output", default=None, help="Override paths.test_dataset from config")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    add_groupby_arg(parser)
    args = parser.parse_args()

    input_path = args.input
    train_out = args.train_output
    val_out = args.val_output
    test_out = args.test_output
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        paths = cfg.get("paths", {})
        input_path = input_path or paths.get("all_dataset")
        train_out = train_out or paths.get("train_dataset")
        val_out = val_out or paths.get("val_dataset")
        test_out = test_out or paths.get("test_dataset")
    if not all([input_path, train_out, val_out, test_out]):
        raise ValueError("--input, --train-output, --val-output, --test-output required "
                         "(or use --config for defaults)")

    for r, name in [(args.val_ratio, "val"), (args.test_ratio, "test")]:
        if not (0.0 < r < 1.0):
            raise ValueError(f"--{name}-ratio must be in (0, 1), got {r}")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1.0, "
                         f"got {args.val_ratio} + {args.test_ratio}")

    rows = load_jsonl(input_path)

    # Stage 1: split off test from the full dataset
    train_val_rows, test_rows = split_by_group(rows, args.test_ratio, args.seed, args.group_by)

    # Stage 2: split val from the train+val pool.
    # val_ratio is relative to the FULL dataset, so we adjust for the
    # already-removed test portion: val_frac = val_ratio / (1 - test_ratio).
    val_frac = args.val_ratio / (1.0 - args.test_ratio)
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_ratio={args.val_ratio} + test_ratio={args.test_ratio} must sum to < 1.0")
    train_rows, val_rows = split_by_group(train_val_rows, val_frac, args.seed + 1, args.group_by)

    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(
            f"Split failed: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}. "
            "Try larger dataset or smaller ratios."
        )

    write_jsonl(train_out, train_rows)
    write_jsonl(val_out, val_rows)
    write_jsonl(test_out, test_rows)

    total = len(rows)
    print(f"input_rows={total}")
    print(f"train_rows={len(train_rows)} ({len(train_rows)/total*100:.1f}%)")
    print(f"val_rows={len(val_rows)} ({len(val_rows)/total*100:.1f}%)")
    print(f"test_rows={len(test_rows)} ({len(test_rows)/total*100:.1f}%)")
    print(f"group_by={args.group_by}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
