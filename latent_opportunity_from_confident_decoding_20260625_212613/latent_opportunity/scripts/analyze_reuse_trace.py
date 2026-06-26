#!/usr/bin/env python3
"""Analyze latent opportunity from saved layer traces and PVM target rows."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "latent_opportunity" / "src"
PVM_SRC = ROOT / "pvm_value" / "src"
for path in (SRC, PVM_SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from latent_opportunity.candidate_builder import build_records_from_trace_targets
from latent_opportunity.io_utils import load_config, read_jsonl, resolve_path, write_csv, write_json, write_jsonl
from latent_opportunity.opportunity_metrics import (
    best_candidate_source_table,
    build_prefix_summaries,
    high_value_rank_table,
    negative_control_table,
    opportunity_table_by_group,
    per_prefix_temperature_rows,
    temperature_aggregate_table,
    top_level_summary,
)
from latent_opportunity.plotting import plot_all
from latent_opportunity.pvm_scoring import assign_pvm_groups, score_prefixes_with_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "latent_opportunity/configs/latent_opportunity_reuse_trace.yaml"))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--limit-prefixes", type=int, default=0)
    parser.add_argument(
        "--prefix-score-source",
        choices=["config", "teacher", "trace_field"],
        default="config",
        help="Override config prefix scoring. trace_field avoids loading the PVM teacher for smoke tests.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _config_repo_root(config: dict[str, Any]) -> Path:
    return Path(config.get("experiment", {}).get("repo_root", ROOT)).resolve()


def _run_dir(config: dict[str, Any], args: argparse.Namespace, repo_root: Path) -> Path:
    if args.run_dir:
        return resolve_path(args.run_dir, repo_root)
    configured = config.get("outputs", {}).get("run_dir")
    if configured:
        return resolve_path(configured, repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "latent_opportunity" / "runs" / f"reuse_trace_{timestamp}"


def _filter_targets(target_rows: list[dict[str, Any]], trace_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in target_rows if str(row.get("trace_id")) in trace_ids]


def _prefix_score_source(config: dict[str, Any], args: argparse.Namespace) -> str:
    if args.prefix_score_source != "config":
        return args.prefix_score_source
    return str(config.get("analysis", {}).get("prefix_score_source", "teacher"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo_root = _config_repo_root(config)
    run_dir = _run_dir(config, args, repo_root)
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs = config["inputs"]
    traces_path = resolve_path(inputs["traces"], repo_root)
    targets_path = resolve_path(inputs["targets"], repo_root)
    trace_rows = read_jsonl(traces_path)
    if args.limit_prefixes:
        trace_rows = trace_rows[: int(args.limit_prefixes)]
    trace_ids = {str(row["trace_id"]) for row in trace_rows}
    target_rows = _filter_targets(read_jsonl(targets_path), trace_ids)
    print(f"loaded {len(trace_rows)} trace rows and {len(target_rows)} target rows")

    analysis_cfg = config.get("analysis", {})
    deltas = [float(x) for x in analysis_cfg.get("delta_thresholds", [0.01, 0.03, 0.05, 0.10])]
    default_delta = float(analysis_cfg.get("default_delta", 0.03))
    temperatures = [float(x) for x in analysis_cfg.get("temperatures", [0.0, 0.2, 0.4, 0.7, 1.0])]
    n_bootstrap = int(analysis_cfg.get("num_bootstrap", 1000))
    seed = int(analysis_cfg.get("seed", 42))
    score_tolerance = float(analysis_cfg.get("score_tolerance", 1e-6))

    model_name = config.get("model", {}).get("name_or_path")
    prefix_records, candidate_records, source_records = build_records_from_trace_targets(
        trace_rows,
        target_rows,
        model_name_or_path=model_name,
        score_tolerance=score_tolerance,
    )
    print(
        "built "
        f"{len(prefix_records)} prefixes, {len(candidate_records)} unique candidates, "
        f"{len(source_records)} source records"
    )

    score_source = _prefix_score_source(config, args)
    if score_source == "teacher":
        teacher_cfg = config["pvm_teacher"]
        prefix_records = score_prefixes_with_teacher(
            prefix_records,
            repo_root=repo_root,
            tf_mil_root=resolve_path(teacher_cfg["tf_mil_root"], repo_root),
            pvm_checkpoint=resolve_path(teacher_cfg["checkpoint"], repo_root),
            model_name_or_path=model_name,
            teacher_top_k=int(teacher_cfg.get("top_k", 64)),
            batch_size=int(teacher_cfg.get("batch_size", 32)),
            parallel_size=teacher_cfg.get("parallel_size", 1),
            gpu_memory_utilization=float(teacher_cfg.get("gpu_memory_utilization", 0.70)),
            enable_prefix_caching=bool(teacher_cfg.get("enable_prefix_caching", False)),
        )
        print("rescored prefix PVM values with frozen teacher")
    elif score_source == "trace_field":
        print("using trace pvm_score_prefix values for prefix grouping")
    else:
        raise ValueError(f"unsupported prefix_score_source: {score_source}")
    prefix_records = assign_pvm_groups(prefix_records)

    prefix_summaries = build_prefix_summaries(
        prefix_records,
        candidate_records,
        deltas=deltas,
        default_delta=default_delta,
    )
    candidate_top_k = max((int(row.get("candidate_top_k", 0)) for row in prefix_records), default=None)
    opportunity_by_group = opportunity_table_by_group(
        prefix_summaries,
        group_key="pvm_group",
        deltas=deltas,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    opportunity_by_decile = opportunity_table_by_group(
        prefix_summaries,
        group_key="relative_position_decile",
        deltas=deltas,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    best_source = best_candidate_source_table(
        prefix_summaries,
        candidate_records,
        default_delta=default_delta,
    )
    rank_table = high_value_rank_table(source_records, default_delta=default_delta)
    temp_prefix_rows = per_prefix_temperature_rows(
        prefix_records,
        source_records,
        temperatures=temperatures,
        default_delta=default_delta,
    )
    temp_table = temperature_aggregate_table(temp_prefix_rows)
    neg_controls = negative_control_table(
        prefix_summaries,
        source_records,
        temp_prefix_rows,
        default_delta=default_delta,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = top_level_summary(
        prefix_summaries,
        deltas=deltas,
        default_delta=default_delta,
        candidate_top_k=candidate_top_k,
    )
    summary.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "run_dir": str(run_dir),
        "inputs": {
            "traces": str(traces_path),
            "targets": str(targets_path),
        },
        "prefix_score_source": score_source,
        "outputs": {
            "prefix_records": str(run_dir / "prefix_records.jsonl"),
            "candidate_records": str(run_dir / "candidate_records.jsonl"),
            "candidate_source_records": str(run_dir / "candidate_source_records.jsonl"),
            "prefix_summaries": str(run_dir / "prefix_summaries.jsonl"),
            "summary": str(run_dir / "summary.json"),
            "tables": str(tables_dir),
            "figures": str(figures_dir),
        },
    })

    write_jsonl(run_dir / "prefix_records.jsonl", prefix_records)
    write_jsonl(run_dir / "candidate_records.jsonl", candidate_records)
    write_jsonl(run_dir / "candidate_source_records.jsonl", source_records)
    write_jsonl(run_dir / "prefix_summaries.jsonl", prefix_summaries)
    write_jsonl(run_dir / "temperature_prefix_records.jsonl", temp_prefix_rows)
    write_csv(tables_dir / "table_1_opportunity_rate_by_pvm_group.csv", opportunity_by_group)
    write_csv(tables_dir / "table_1b_opportunity_rate_by_position_decile.csv", opportunity_by_decile)
    write_csv(tables_dir / "table_2_best_candidate_source.csv", best_source)
    write_csv(tables_dir / "table_3_high_value_candidate_rank_distribution.csv", rank_table)
    write_csv(tables_dir / "table_4_temperature_elicitation_potential.csv", temp_table)
    write_csv(tables_dir / "table_5_negative_controls.csv", neg_controls)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "run_manifest.json", {
        "summary": summary,
        "counts": {
            "trace_rows": len(trace_rows),
            "target_rows": len(target_rows),
            "prefix_records": len(prefix_records),
            "candidate_records": len(candidate_records),
            "candidate_source_records": len(source_records),
            "temperature_prefix_records": len(temp_prefix_rows),
        },
    })
    if not args.no_plots:
        plot_all(
            prefix_summaries=prefix_summaries,
            opportunity_by_group=opportunity_by_group,
            best_source=best_source,
            rank_table=rank_table,
            temperature_table=temp_table,
            out_dir=figures_dir,
            default_delta=default_delta,
        )
    print(f"wrote latent-opportunity outputs to {run_dir}")
    print(f"summary: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
