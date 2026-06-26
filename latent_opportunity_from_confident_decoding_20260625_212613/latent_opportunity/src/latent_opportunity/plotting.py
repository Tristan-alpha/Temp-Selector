"""Matplotlib figures for latent-opportunity diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latent_opportunity.opportunity_metrics import delta_label


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_all(
    *,
    prefix_summaries: Sequence[Mapping[str, Any]],
    opportunity_by_group: Sequence[Mapping[str, Any]],
    best_source: Sequence[Mapping[str, Any]],
    rank_table: Sequence[Mapping[str, Any]],
    temperature_table: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    default_delta: float,
) -> None:
    out = Path(out_dir)
    label = delta_label(default_delta)

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [row["pvm_group"] for row in opportunity_by_group]
    rates = [row.get(f"overall_opportunity_rate_delta_{label}", 0.0) or 0.0 for row in opportunity_by_group]
    ax.bar(groups, rates, color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Opportunity rate")
    ax.set_title(f"Opportunity Rate by PVM Group (delta={label})")
    _save(fig, out / "figure_1_opportunity_rate_by_pvm_group.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    by_decile: dict[int, list[float]] = {}
    for row in prefix_summaries:
        decile = row.get("relative_position_decile")
        if decile is None:
            continue
        by_decile.setdefault(int(decile), []).append(1.0 if row.get(f"overall_opportunity_{label}") else 0.0)
    xs = sorted(by_decile)
    ys = [sum(by_decile[x]) / len(by_decile[x]) for x in xs]
    ax.plot(xs, ys, marker="o", color="#f58518")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative token-position decile")
    ax.set_ylabel("Opportunity rate")
    ax.set_title("Opportunity Rate by Token Position")
    _save(fig, out / "figure_2_opportunity_rate_by_position.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    source_names = [row["source_type"] for row in best_source]
    counts = [row["count"] for row in best_source]
    ax.bar(source_names, counts, color="#54a24b")
    ax.set_ylabel("Best-candidate count")
    ax.set_title("Best Candidate Source Distribution")
    _save(fig, out / "figure_3_best_candidate_source_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for row in rank_table:
        xs = [3, 5, 10, 20]
        ys = [
            row.get("p_rank_le_3") or 0.0,
            row.get("p_rank_le_5") or 0.0,
            row.get("p_rank_le_10") or 0.0,
            row.get("p_rank_le_20") or 0.0,
        ]
        ax.plot(xs, ys, marker="o", label=str(row["source_type"]))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Rank threshold")
    ax.set_ylabel("CDF")
    if rank_table:
        ax.legend()
    ax.set_title("Rank CDF of High-Value Candidates")
    _save(fig, out / "figure_4_rank_cdf_high_value_candidates.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    xs = [float(row["V_g"]) for row in prefix_summaries]
    ys = [float(row["V_best_all"]) for row in prefix_summaries if row.get("V_best_all") is not None]
    xs2 = [float(row["V_g"]) for row in prefix_summaries if row.get("V_best_all") is not None]
    ax.scatter(xs2, ys, s=10, alpha=0.55, color="#4c78a8")
    if xs:
        lo = min(xs)
        hi = max(max(xs), max(ys) if ys else max(xs))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.plot([lo, hi], [lo + default_delta, hi + default_delta], color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("V_g")
    ax.set_ylabel("V_best_all")
    ax.set_title("Final Greedy Value vs Best Alternative")
    _save(fig, out / "figure_5_greedy_vs_best_alt.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for source_family in sorted({str(row["source_family"]) for row in temperature_table}):
        rows = [row for row in temperature_table if row["source_family"] == source_family]
        rows = sorted(rows, key=lambda row: (str(row["pvm_group"]), float(row["temperature"])))
        for group in sorted({str(row["pvm_group"]) for row in rows}):
            sub = sorted([row for row in rows if row["pvm_group"] == group], key=lambda row: float(row["temperature"]))
            ax.plot(
                [float(row["temperature"]) for row in sub],
                [float(row.get("P_T_A_plus") or 0.0) for row in sub],
                marker="o",
                label=f"{source_family}:{group}:A+",
            )
    ax.set_xlabel("Temperature")
    ax.set_ylabel("P_T(A_plus)")
    ax.legend(fontsize=7)
    ax.set_title("Temperature Elicitation Curve")
    _save(fig, out / "figure_6_temperature_elicitation_curve.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [
        float(row["near_final_advantage"])
        for row in prefix_summaries
        if row.get("near_final_advantage") is not None
    ]
    ax.hist(vals, bins=30, color="#b279a2")
    ax.set_xlabel("V_best_near_final - V_best_final_alt")
    ax.set_ylabel("Prefix count")
    ax.set_title("Near-Final Advantage Distribution")
    _save(fig, out / "figure_7_near_final_advantage_distribution.png")
