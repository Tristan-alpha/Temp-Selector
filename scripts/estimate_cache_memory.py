#!/usr/bin/env python3
"""Estimate system RAM needed for MIL segment feature cache under different
pooling modes and (instance_dim, segment_size) combinations.

Reads train + val JSONL files, counts response tokens per row, and computes:
  - mean  mode: per segment = instance_dim × 4 bytes
  - concat mode: per segment = segment_size × instance_dim × 4 bytes

Usage:
  python scripts/estimate_cache_memory.py
  python scripts/estimate_cache_memory.py --dims 64,128,256 --segments 64,128,256,512
"""

from __future__ import annotations

import argparse
import json
import math
from typing import List, Tuple


def load_token_counts(path: str) -> List[int]:
    """Return list of response token counts from a JSONL file."""
    counts: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts.append(len(row.get("token_ids", [])))
    return counts


def estimate(
    token_counts: List[int],
    instance_dim: int,
    segment_size: int,
    mode: str,
) -> Tuple[int, int, float]:
    """Return (n_segments_total, max_segments_per_row, cache_gb) for one config."""
    total_segments = 0
    max_segments = 0
    for n_tok in token_counts:
        k = max(1, math.ceil(n_tok / segment_size)) if n_tok > 0 else 1
        total_segments += k
        max_segments = max(max_segments, k)

    if mode == "mean":
        dims_per_segment = instance_dim
    else:  # concat
        dims_per_segment = segment_size * instance_dim

    total_bytes = total_segments * dims_per_segment * 4  # float32
    cache_gb = total_bytes / (1024 ** 3)
    return total_segments, max_segments, cache_gb


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate MIL segment cache memory")
    parser.add_argument("--train", default="datasets/train.jsonl", help="Path to train JSONL")
    parser.add_argument("--val", default="datasets/val.jsonl", help="Path to val JSONL")
    parser.add_argument("--dims", default="8,16,32,64,128,256,512,4098",
                        help="Comma-separated instance_dim values to test")
    parser.add_argument("--segments", default="64,128,256,512",
                        help="Comma-separated segment_size values to test")
    args = parser.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    segs = [int(x) for x in args.segments.split(",")]

    # Load data
    train_counts = load_token_counts(args.train)
    val_counts = load_token_counts(args.val)
    all_counts = train_counts + val_counts

    n_rows = len(all_counts)
    n_train = len(train_counts)
    n_val = len(val_counts)
    total_tokens = sum(all_counts)
    max_tokens = max(all_counts)
    avg_tokens = total_tokens / n_rows

    print(f"{'='*70}")
    print(f"Dataset: {args.train} ({n_train} rows) + {args.val} ({n_val} rows)")
    print(f"Total rows: {n_rows:,}  |  Total tokens: {total_tokens:,}")
    print(f"Tokens per row: min={min(all_counts)}, max={max_tokens}, avg={avg_tokens:.0f}")
    print(f"{'='*70}")

    # Header
    print()
    print(f"{'mode':>6}  {'dim':>5}  {'seg':>4}  "
          f"{'dim/seg':>8}  {'segs_total':>10}  {'max_seg':>7}  "
          f"{'cache (GB)':>10}  {'vs mean@4098':>13}")
    print(f"{'-'*6}  {'-'*5}  {'-'*4}  "
          f"{'-'*8}  {'-'*10}  {'-'*7}  "
          f"{'-'*10}  {'-'*13}")

    # Baseline: mean @ 4098
    _, _, baseline_gb = estimate(all_counts, 4098, 512, "mean")

    for mode in ("mean", "concat"):
        for dim in dims:
            for seg in segs:
                n_segs, max_seg, cache_gb = estimate(all_counts, dim, seg, mode)
                ratio = cache_gb / baseline_gb if baseline_gb > 0 else 0

                # Highlight: concat dims close to mean@4098 per-segment size
                marker = ""
                if mode == "concat" and abs(seg * dim - 4098) <= 10:
                    marker = " ←"

                print(f"{mode:>6}  {dim:>5}  {seg:>4}  "
                      f"{seg*dim:>8}  {n_segs:>10,}  {max_seg:>7}  "
                      f"{cache_gb:>10.2f}  {ratio:>12.2f}x{marker}")

    print()
    print("← = concat per-segment dims ≈ mean@4098 (similar memory/segment)")
    print(f"\nBaseline (mean, dim=4098, seg=512): {baseline_gb:.2f} GB")


if __name__ == "__main__":
    main()
