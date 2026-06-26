#!/usr/bin/env python3
"""Analyze segment-level latent reasoning path candidates."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.segment_records import (  # noqa: E402
    PVM_GROUP_ORDER,
    delta_label,
    opportunity_by_pvm_group,
    prefix_summary_rows,
    proposal_yield_by_temperature,
    read_jsonl,
    selection_gain,
    standardize_segment_records,
    top_level_summary,
    write_csv,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "segment_latent_paths"


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(path), "read_error": True}
    return data if isinstance(data, dict) else {"path": str(path), "data": data}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "segment_candidate_records.jsonl",
        help="Raw generated segment candidate records.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    parser.add_argument("--default-delta", type=float, default=0.05)
    return parser.parse_args()


def _ordered_groups(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda row: PVM_GROUP_ORDER.get(str(row.get("pvm_group", "")), 99))


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_figures(
    *,
    opportunity_rows: Sequence[Mapping[str, Any]],
    temperature_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    default_delta: float,
) -> None:
    label = delta_label(default_delta)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ordered = _ordered_groups(opportunity_rows)
    groups = [str(row["pvm_group"]) for row in ordered]
    rates = [float(row.get(f"opportunity_rate_delta_{label}") or 0.0) for row in ordered]
    ax.bar(groups, rates, color=["#4c78a8", "#72b7b2", "#f58518"][: len(groups)])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Prefix PVM group")
    ax.set_ylabel("Segment opportunity rate")
    ax.set_title(f"Opportunity Rate by PVM Group (delta={label})")
    _save(fig, output_dir / "fig_opportunity_by_pvm_group.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    temps = [float(row["temperature"]) for row in temperature_rows]
    yields = [float(row.get("yield") or 0.0) for row in temperature_rows]
    best = [float(row.get("best_of_n_yield") or 0.0) for row in temperature_rows]
    ax.plot(temps, yields, marker="o", label="sample yield", color="#4c78a8")
    ax.plot(temps, best, marker="s", label="best-of-n yield", color="#f58518")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Proposal yield")
    ax.set_title(f"Temperature Proposal Yield (delta={label})")
    ax.legend()
    _save(fig, output_dir / "fig_temperature_yield.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    metric_names = ["Acc_greedy", "Acc_random_sampled", "Acc_PVM_best"]
    labels = ["greedy", "random sampled", "PVM-best"]
    rows = [row for row in selection_rows if row.get("scope") in {"all", "low_pvm"}]
    width = 0.35
    xs = list(range(len(metric_names)))
    for idx, row in enumerate(rows):
        offset = (idx - (len(rows) - 1) / 2.0) * width
        values = [float(row.get(name) or 0.0) for name in metric_names]
        ax.bar([x + offset for x in xs], values, width=width, label=str(row["scope"]))
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Final correctness / success rate")
    ax.set_title("Greedy vs Random vs PVM-Best")
    if rows:
        ax.legend()
    _save(fig, output_dir / "fig_selection_gain.png")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    raw_rows = read_jsonl(args.candidates)
    records = standardize_segment_records(raw_rows)
    opportunity_rows = opportunity_by_pvm_group(records, deltas=args.deltas)
    temperature_rows = proposal_yield_by_temperature(records, delta=args.default_delta)
    selection_rows = selection_gain(records)
    prefix_rows = prefix_summary_rows(records, default_delta=args.default_delta)
    summary = top_level_summary(
        records,
        opportunity_rows=opportunity_rows,
        temperature_rows=temperature_rows,
        selection_rows=selection_rows,
        default_delta=args.default_delta,
    )
    summary.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "raw_candidates": str(args.candidates.resolve()),
        "output_dir": str(output_dir),
        "delta_thresholds": [float(delta) for delta in args.deltas],
        "outputs": {
            "standard_records": str(output_dir / "segment_records.jsonl"),
            "prefix_summaries": str(output_dir / "prefix_summaries.csv"),
            "opportunity_by_pvm_group": str(output_dir / "opportunity_by_pvm_group.csv"),
            "proposal_yield_by_temperature": str(output_dir / "proposal_yield_by_temperature.csv"),
            "selection_gain": str(output_dir / "selection_gain.csv"),
            "fig_opportunity_by_pvm_group": str(output_dir / "fig_opportunity_by_pvm_group.png"),
            "fig_temperature_yield": str(output_dir / "fig_temperature_yield.png"),
            "fig_selection_gain": str(output_dir / "fig_selection_gain.png"),
        },
    })

    write_jsonl(output_dir / "segment_records.jsonl", records)
    write_csv(output_dir / "prefix_summaries.csv", prefix_rows)
    write_csv(output_dir / "opportunity_by_pvm_group.csv", opportunity_rows)
    write_csv(output_dir / "proposal_yield_by_temperature.csv", temperature_rows)
    write_csv(output_dir / "selection_gain.csv", selection_rows)
    write_json(output_dir / "summary.json", summary)
    generation_manifest = _read_optional_json(output_dir / "generation_manifest.json")
    scoring_manifest = _read_optional_json(output_dir / "scoring_manifest.json")
    write_json(output_dir / "run_manifest.json", {
        "summary": summary,
        "generation_manifest": generation_manifest,
        "scoring_manifest": scoring_manifest,
        "counts": {
            "raw_candidates": len(raw_rows),
            "standard_records": len(records),
            "sample_records": sum(len(row.get("samples", [])) for row in records),
        },
    })
    plot_figures(
        opportunity_rows=opportunity_rows,
        temperature_rows=temperature_rows,
        selection_rows=selection_rows,
        output_dir=output_dir,
        default_delta=args.default_delta,
    )
    print(f"loaded raw_candidates={len(raw_rows)} prefixes={len(records)}")
    print(f"wrote segment latent path analysis to {output_dir}")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
