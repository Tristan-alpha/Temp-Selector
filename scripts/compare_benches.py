#!/usr/bin/env python3
"""Compare outputs from the five benchmark conditions (greedy decoding).

Usage:
    python scripts/compare_benches.py \
        --baseline        results/baseline.json \
        --hidden          results/hidden_states.json \
        --segment         results/segment.json \
        --segment-token   results/segment_token_ids.json \
        --segment-fixed   results/segment_fixed.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare greedy benchmark results across 5 conditions")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hidden", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--segment-token", required=True)
    parser.add_argument("--segment-fixed", default=None, help="segment_fixed JSON (optional)")
    parser.add_argument("--output", default=None, help="Save diff report JSON here")
    args = parser.parse_args()

    data = {
        "baseline": _load(args.baseline),
        "hidden_states": _load(args.hidden),
        "segment": _load(args.segment),
        "segment_token_ids": _load(args.segment_token),
    }
    if args.segment_fixed:
        data["segment_fixed"] = _load(args.segment_fixed)

    # ── Summary table ──
    print("=" * 90)
    print("GREEDY BENCHMARK COMPARISON")
    print("=" * 90)
    header = f"{'Condition':<22} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Time(s)':>10} {'Δ vs baseline':>14}"
    print(header)
    print("-" * 90)

    base_acc = data["baseline"]["summary"]["accuracy"]
    base_n = data["baseline"]["summary"]["n_total"]

    results_map: Dict[str, Dict[str, Any]] = {}
    for name, d in data.items():
        s = d["summary"]
        acc = s["accuracy"]
        n_c = s["n_correct"]
        n_t = s["n_total"]
        t = s["gen_time_s"]
        delta = acc - base_acc
        results_map[name] = {"accuracy": acc, "n_correct": n_c, "n_total": n_t, "delta": delta}
        print(f"{name:<22} {acc:>10.4f} {n_c:>10} {n_t:>8} {t:>10.1f} {delta:>+14.4f}")

    # ── Exact-match analysis (greedy = deterministic with same seed) ──
    print("\n" + "=" * 90)
    print("OUTPUT DIVERGENCE ANALYSIS (greedy → any diff is a bug or path change)")
    print("=" * 90)

    base_results = {r["prompt_idx"]: r for r in data["baseline"]["results"]}
    for name in ["hidden_states", "segment", "segment_token_ids", "segment_fixed"]:
        if name not in data:
            continue
        other_results = {r["prompt_idx"]: r for r in data[name]["results"]}
        common_indices = sorted(set(base_results.keys()) & set(other_results.keys()))

        n_text_match = 0
        n_correct_match = 0
        n_text_diff = 0
        n_correct_flip = 0
        first_diff_idx: List[int] = []

        for idx in common_indices:
            br = base_results[idx]
            ort = other_results[idx]
            text_same = br["response"] == ort["response"]
            correct_same = br["correct"] == ort["correct"]

            if text_same:
                n_text_match += 1
            else:
                n_text_diff += 1
                if len(first_diff_idx) < 5:
                    first_diff_idx.append(idx)

            if correct_same:
                n_correct_match += 1
            else:
                n_correct_flip += 1

        print(f"\n--- {name} vs baseline ---")
        print(f"  Text identical:    {n_text_match}/{len(common_indices)} "
              f"({100 * n_text_match / max(1, len(common_indices)):.1f}%)")
        print(f"  Text differ:       {n_text_diff}/{len(common_indices)}")
        print(f"  Correctness match: {n_correct_match}/{len(common_indices)}")
        print(f"  Correctness flip:  {n_correct_flip}")

        if first_diff_idx:
            print(f"  First {len(first_diff_idx)} diverging samples:")
            for idx in first_diff_idx:
                br = base_results[idx]
                ort = other_results[idx]
                base_text = br["response"]
                other_text = ort["response"]
                # Find first diff position
                min_len = min(len(base_text), len(other_text))
                diff_pos = next((j for j in range(min_len)
                                 if base_text[j] != other_text[j]), min_len)
                ctx = 30
                print(f"    prompt_idx={idx}  q={br['question'][:50]}...")
                print(f"      baseline_correct={br['correct']}  {name}_correct={ort['correct']}")
                print(f"      first diff at char {diff_pos}:")
                print(f"        base: ...{base_text[max(0,diff_pos-ctx):diff_pos+ctx]!r}...")
                print(f"        other: ...{other_text[max(0,diff_pos-ctx):diff_pos+ctx]!r}...")

    # ── Cross-condition comparison ──
    print("\n" + "=" * 90)
    print("CROSS-CONDITION TOKEN-LEVEL COMPARISON")
    print("=" * 90)

    # Count total tokens per condition
    for name, d in data.items():
        total_tok = sum(len(r.get("token_ids", [])) for r in d["results"])
        avg_tok = total_tok / max(1, len(d["results"]))
        print(f"  {name:<22} total_tokens={total_tok:>8}  avg_tokens={avg_tok:>8.1f}")

    # ── Save ──
    report = {
        "accuracy": {name: r["accuracy"] for name, r in results_map.items()},
        "deltas": {name: r["delta"] for name, r in results_map.items()},
    }
    if args.output:
        json.dump(report, open(args.output, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
